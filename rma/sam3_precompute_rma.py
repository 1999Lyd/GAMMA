#!/usr/bin/env python
"""RMA stage 2: SAM-3 detection cache on the agent-1 frame grid (rma_sft_v1,
stride 5 + final frame), r5 concept table with PER-TASK vocabulary (word bag =
dataset dir name + subtask words), threshold 0.25.
Vision embeddings are computed ONCE per 16-frame batch and reused across the
task's concept phrases (Sam3Model.forward(vision_embeds=...)); verified
identical to the full forward, 1.9x faster.
Output: $OUT/task{T}_seed{S}.json = {step: [[name, row, col, score, x0,y0,x1,y1], ...]}
  (row, col) = box centre in the RoboMME <row, col> convention; all hits kept
  (generic + specific concepts may overlap; the corpus generator resolves).
Workers self-balance through .lock files: any number of GPUs, same queue.
Env: OUT, SEEDS_PER_TASK (default 40 -> seeds 100..139), BATCH (16), THR (0.25)
"""
import os, sys, glob, json, time, re
import torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor

HERE = os.path.dirname(os.path.abspath(__file__))
F = os.path.expandvars("${GAMMA_DATA}/data/rma_sft_v1")
RAW = os.path.expandvars("${GAMMA_DATA}/data/RoboMemArena")
OUT = os.environ.get("OUT", f"{F}/dets_sam3")
NSEED = int(os.environ.get("SEEDS_PER_TASK", "40"))
BATCH = int(os.environ.get("BATCH", "16"))
os.environ.setdefault("HF_HOME", os.path.expandvars("${HF_HOME}"))
os.makedirs(f"{OUT}/.locks", exist_ok=True)
cfg = json.load(open(os.environ.get("PROMPTS", f"{HERE}/prompts_rma_r6.json")))
TOPK = int(os.environ.get("TOPK", cfg.get("topk", 3)))
CONCEPTS = [tuple(c) for c in cfg["concepts"]]; ALWAYS = [tuple(a) for a in cfg["always"]]
THR = float(os.environ.get("THR", cfg.get("thr", 0.25)))

def dir_words():
    bags = {}
    for ds in glob.glob(f"{RAW}/*/*_dataset"):
        base = os.path.basename(ds); tid = int(base.split("_")[0])
        bags[tid] = set(w for w in base.lower().split("_") if w)
    return bags
DIRW = dir_words()

def cands_for(man):
    bag = set(DIRW.get(man["task"], set()))
    for st in man["subtasks"]:
        for w in re.split(r"[_\d]+", st["words"].lower()):
            if w: bag.add(w)
    for w in re.split(r"[^a-z]+", man["instruction"].lower()):
        if w: bag.add(w)
    c = [(p, n) for p, n, trig in CONCEPTS if any(t in bag for t in trig)]
    return c + [(p, n) for p, n in ALWAYS]

DEV = "cuda:0"
proc = Sam3Processor.from_pretrained("facebook/sam3")
model = Sam3Model.from_pretrained("facebook/sam3", dtype=torch.bfloat16).to(DEV).eval()
print(f"[sam3-rma] ready; {len(CONCEPTS)} concepts, thr {THR}, seeds/task {NSEED}", flush=True)

@torch.no_grad()
def detect(pils, cands):
    res = [[] for _ in pils]
    inp0 = proc(images=pils, text=[cands[0][0]] * len(pils), return_tensors="pt").to(DEV)
    ve = model.get_vision_features(pixel_values=inp0["pixel_values"])
    sizes = inp0["original_sizes"].tolist()
    for phrase, name in cands:
        tin = proc(images=pils, text=[phrase] * len(pils), return_tensors="pt").to(DEV)
        out = model(vision_embeds=ve, input_ids=tin["input_ids"], attention_mask=tin["attention_mask"])
        pp = proc.post_process_instance_segmentation(out, threshold=THR, mask_threshold=0.5, target_sizes=sizes)
        for i, r in enumerate(pp):
            order = sorted(range(len(r["scores"])), key=lambda j: -float(r["scores"][j]))[:TOPK]
            for box, sc in [(r["boxes"][j], r["scores"][j]) for j in order]:
                x0, y0, x1, y1 = [float(v) for v in box.tolist()]
                res[i].append([name, int(round((y0 + y1) / 2)), int(round((x0 + x1) / 2)), round(float(sc), 3),
                               int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))])
    return res

mans = sorted(glob.glob(f"{F}/manifests/task*_seed*.json"),
              key=lambda p: (int(re.search(r"task(\d+)_", p).group(1)), int(re.search(r"seed(\d+)", p).group(1))))
done = skipped = 0; t0 = time.time()
for mp in mans:
    man = json.load(open(mp)); T, S = man["task"], man["seed"]
    if S >= 100 + NSEED: continue
    op, lk = f"{OUT}/task{T}_seed{S}.json", f"{OUT}/.locks/task{T}_seed{S}"
    if os.path.exists(op): skipped += 1; continue
    try:
        fd = os.open(lk, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.close(fd)
    except FileExistsError:
        continue
    cands = cands_for(man); grid = man["grid"]; out = {}; te = time.time()
    for i in range(0, len(grid), BATCH):
        ts = grid[i:i + BATCH]
        pils = [Image.open(f"{F}/frames/task{T}_seed{S}/f{t}.jpg").convert("RGB") for t in ts]
        for t, d in zip(ts, detect(pils, cands)): out[str(t)] = d
    json.dump({"task": T, "seed": S, "concepts": [n for _, n in cands], "thr": THR, "dets": out}, open(op + ".tmp", "w"))
    os.replace(op + ".tmp", op); done += 1
    print(f"[sam3-rma] task{T} seed{S}: {len(grid)} frames x {len(cands)} concepts in {time.time()-te:.0f}s "
          f"(done {done}, {(time.time()-t0)/60:.1f} min)", flush=True)
print(f"[sam3-rma] finished: {done} episodes written, {skipped} pre-existing", flush=True)

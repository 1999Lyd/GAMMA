#!/usr/bin/env python
"""SAM-3 prompt probe round 3: PER-TASK vocabulary.

Each sample frame is probed only with the concepts its task actually
involves (word bag = dataset dir name + subtask filenames), killing the
cross-task false positives of rounds 1-2. Food items get their specific
round-1 phrases; receptacles keep the round-2 winners. Overlays ->
probe_overlays_r5/ for review; the surviving (phrase, name, triggers)
table is the draft prompts_rma.json for the full precompute."""
import collections, glob, json, os, re
import torch
from PIL import Image, ImageDraw

os.environ.setdefault("HF_HOME", os.path.expandvars("${HF_HOME}"))
from transformers import Sam3Model, Sam3Processor

DEV = "cuda:0"
THR = 0.25
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.expandvars("${GAMMA_DATA}/data/RoboMemArena")
OUT = f"{HERE}/probe_overlays_r5"
os.makedirs(OUT, exist_ok=True)

# (phrase, name, trigger words)
CONCEPTS = [
    ("a small box on the table", "smallbox",
     ["butter", "chocolate", "cookies", "cookie", "cream", "popcorn", "pudding"]),
    ("a small snack box on the dining table", "snackbox",
     ["butter", "chocolate", "cookies", "cookie", "cream", "popcorn", "pudding"]),
    ("an open microwave oven", "microwave_open2", ["microwave"]),
    ("a small white and blue carton on the table", "butter", ["butter"]),
    ("a dark brown box on the table", "chocolate", ["chocolate"]),
    ("a dark brown box on the table", "pudding", ["pudding"]),
    ("a yellow box with a red lid", "cookies", ["cookies", "cookie"]),
    ("an orange snack box with a white label", "popcorn", ["popcorn"]),
    ("a small white carton on the table", "cream", ["cream"]),
    ("a white milk carton", "milk", ["milk"]),
    ("a red bottle on the table", "tomato_sauce", ["tomato"]),
    ("an orange juice bottle", "orange_juice", ["orange"]),
    ("a dark green bottle", "wine", ["wine"]),
    ("a mug on the table", "mug", ["mug", "cup"]),
    ("a black frying pan", "frypan", ["frypan", "pan"]),
    ("a white dish rack with slats", "drainer", ["drainer"]),
    ("a wooden basket with slatted sides", "basket", ["basket"]),
    ("a dark wooden cabinet standing on the table", "cabinet", ["drawer", "cabinet"]),
    ("a black microwave with a door", "microwave", ["microwave"]),
    ("an open microwave with its door open", "microwave_open", ["microwave"]),
    ("an open drawer of the cabinet", "drawer_open", ["drawer"]),
    ("a small box inside the open drawer", "drawer_content", ["drawer"]),
    ("a plate on the table", "plate", ["plate"]),
]
ALWAYS = [("robot arm", "arm")]
PALETTE = [(255,80,80),(80,220,80),(90,140,255),(255,200,40),(255,120,255),
           (60,220,220),(255,150,60),(180,255,120),(200,120,255),(120,200,255)]


def task_bags():
    bags = collections.defaultdict(set)
    for ds in glob.glob(f"{DATA}/Multi-*/*_dataset"):
        base = os.path.basename(ds)
        tid = int(base.split("_")[0])
        for w in base.lower().split("_"):
            bags[tid].add(w)
        for f in glob.glob(f"{ds}/subtask_data/*.hdf5"):
            words = os.path.basename(f).rsplit("_seed", 1)[0].lower()
            for w in re.split(r"[_\d]+", words):
                if w:
                    bags[tid].add(w)
    return bags


bags = task_bags()
proc = Sam3Processor.from_pretrained("facebook/sam3")
model = Sam3Model.from_pretrained("facebook/sam3", dtype=torch.bfloat16).to(DEV).eval()
print("[r3] model ready", flush=True)

vocab_used = {}
for f in sorted(glob.glob(f"{HERE}/sample_frames/*.png") + glob.glob(f"{HERE}/sample_frames_r4/*.jpg")):
    m = re.search(r"task(\d+)_", os.path.basename(f))
    tid = int(m.group(1))
    bag = bags[tid]
    cands = [(p, n) for p, n, trig in CONCEPTS if any(t in bag for t in trig)]
    cands += ALWAYS
    vocab_used[os.path.basename(f)] = [n for _, n in cands]
    pil = Image.open(f).convert("RGB")
    hits = []
    with torch.no_grad():
        for phrase, name in cands:
            inp = proc(images=[pil], text=[phrase], return_tensors="pt").to(DEV)
            out = model(**inp)
            pp = proc.post_process_instance_segmentation(
                out, threshold=THR, mask_threshold=0.5,
                target_sizes=inp.get("original_sizes").tolist())[0]
            for box, sc in zip(pp["boxes"], pp["scores"]):
                hits.append((name, float(sc), *box.tolist()))
    im = pil.resize((512, 512), Image.NEAREST)
    d = ImageDraw.Draw(im)
    names = sorted({h[0] for h in hits})
    for name, sc, x0, y0, x1, y1 in sorted(hits, key=lambda h: -h[1]):
        c = PALETTE[names.index(name) % len(PALETTE)]
        d.rectangle([x0*2, y0*2, x1*2, y1*2], outline=c, width=2)
        d.text((x0*2+2, max(y0*2-11, 0)), f"{name} {sc:.2f}", fill=c)
    im.save(f"{OUT}/{os.path.basename(f)}")
    print(f"[r3] {os.path.basename(f)}: {len(cands)} concepts, "
          f"{len(hits)} hits -> {sorted({h[0] for h in hits})}", flush=True)
json.dump({"concepts": CONCEPTS, "always": ALWAYS,
           "per_frame_vocab": vocab_used, "thr": THR},
          open(f"{HERE}/prompts_rma_r5_draft.json", "w"), indent=1)
print("[r3] overlays ->", OUT, flush=True)

#!/usr/bin/env python
"""SAM 3 detection cache on the agent-1 frame grid (stride 4).

Replaces the Grounding-DINO cache: SAM 3 assigns ONE label per physical site,
which the dino cache could not do (it emitted button+target+container at the
same coordinates, and agent-2 could not bind them -- the direct cause of the
PickXtimes closed-loop failure).

Concepts are phrase-tuned against known button/target positions:
"a grey push button on the table" hits 0.69-0.84 where "round button" missed.

Workers self-balance through .lock files, so any number of GPUs can be added
to the same queue at any time.
"""
import glob, io, json, os, time
import pyarrow.parquet as pq
import torch
from PIL import Image

D = os.path.expandvars("${GAMMA_DATA}/data/robomme_lerobot")
OUT = os.environ.get("OUT_DIR",
                     os.path.expandvars("${GAMMA_DATA}/data/wam_sft_v8")) + "/dets_sam3"
FSTEP, BATCH = 4, 16
THR = float(os.environ.get("THR", "0.35"))
os.makedirs(OUT, exist_ok=True)
os.makedirs(f"{OUT}/.locks", exist_ok=True)
t0 = time.time()

os.environ.setdefault("HF_HOME", os.path.expandvars("${HF_HOME}"))
from transformers import Sam3Model, Sam3Processor
DEV = "cuda:0"
proc = Sam3Processor.from_pretrained("facebook/sam3")
model = Sam3Model.from_pretrained("facebook/sam3", dtype=torch.bfloat16).to(DEV).eval()
# phrasings chosen by scoring candidates against oracle-derived truths
# (10 per class, hit@14px): cubes/button/bin 1.00, target/container 0.90,
# peg 0.50 (best available).  "bin" as a bare word scored 0.00 -- "a white
# open box" 1.00 -- so the naive prompt would have lost binfill entirely.
CON = [("a small red block on the table", "red_cube"),
       ("a small green block on the table", "green_cube"),
       ("a small blue block on the table", "blue_cube"),
       ("white container box", "container"),
       ("a grey push button on the table", "button"),
       ("a flat square target marker on the table", "target"),
       ("a white open box", "bin"),
       ("a long cylindrical peg on the table", "peg"),
       ("a thin stick standing on the table", "stick"),
       ("robot arm", "arm")]
# SAM3 cannot see the ph highlight decal (0/10 on nine phrasings), so that
# one channel stays on the deterministic HSV white-blob detector, which
# verified at 95% and runs live at eval time too.
import numpy as np
from scipy import ndimage


def white_blobs(rgb, min_area=40):
    """HSV white-decal detector, numpy/scipy only (same rule as the verified
    cv2 version): high value, low saturation, arm zone and borders rejected."""
    a = rgb.astype(np.float32) / 255.0
    mx, mn = a.max(-1), a.min(-1)
    v = mx * 255.0
    s = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0) * 255.0
    m = (v > 185) & (s < 50)
    lab, n = ndimage.label(m)
    out = []
    for i in range(1, n + 1):
        ys, xs = np.where(lab == i)
        if len(ys) < min_area:
            continue
        wx, wy = int(round(ys.mean())), int(round(xs.mean()))
        if wx > 45 and 14 < wy < 242:
            out.append((wx, wy))
    return out
print(f"[sam3] ready ({(time.time()-t0)/60:.1f} min), {len(CON)} concepts", flush=True)


@torch.no_grad()
def detect(pils):
    res = [[] for _ in pils]
    for phrase, name in CON:
        inp = proc(images=pils, text=[phrase]*len(pils), return_tensors="pt").to(DEV)
        out = model(**inp)
        pp = proc.post_process_instance_segmentation(
            out, threshold=THR, mask_threshold=0.5,
            target_sizes=inp.get("original_sizes").tolist())
        for i, r in enumerate(pp):
            for box, sc in zip(r["boxes"], r["scores"]):
                x0, y0, x1, y1 = box.tolist()
                x, y = int(round((y0+y1)/2)), int(round((x0+x1)/2))
                if not (5 < x < 251 and 5 < y < 251):
                    continue
                res[i].append((name, x, y, float(sc)))
    # one label per site: strongest wins within 10px across classes
    clean = []
    for dets in res:
        dets.sort(key=lambda d: -d[3])
        keep = []
        for n, x, y, s in dets:
            if any((x-a)**2 + (y-b)**2 <= 10**2 for _, a, b in keep):
                continue
            keep.append((n, x, y))
        clean.append(keep)
    return clean


files = sorted(glob.glob(f"{D}/data/chunk-*/*.parquet"))
done = 0
for f in files:
    e = int(pq.read_table(f, columns=["episode_index"])["episode_index"][0].as_py())
    op, lk = f"{OUT}/e{e}.json", f"{OUT}/.locks/e{e}"
    if os.path.exists(op):
        continue
    try:                                   # atomic claim
        fd = os.open(lk, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError:
        continue
    tbl = pq.read_table(f)
    si = tbl["step_idx"].to_pylist()
    idxs = sorted(range(len(si)), key=lambda i: si[i])
    imgs = tbl["image"].to_pylist()
    T = len(idxs)
    grid = sorted(set(list(range(0, T, FSTEP)) + [T-1]))
    out = {}
    for i in range(0, len(grid), BATCH):
        ts = grid[i:i+BATCH]
        pils = []
        for t in ts:
            raw = imgs[idxs[t]]
            raw = raw["bytes"] if isinstance(raw, dict) else raw
            pils.append(Image.open(io.BytesIO(raw)).convert("RGB"))
        for t, pil, d in zip(ts, pils, detect(pils)):
            # highlight is NEVER deduped against object detections: the decal
            # surrounds a cube, so its centroid always collides with the
            # cube's SAM3 hit (measured: dedup cut display-window recall
            # 1.00 -> 0.20 on ph val).
            for hx, hy in white_blobs(np.array(pil)):
                d.append(("highlight", hx, hy))
            out[str(t)] = d
    json.dump(out, open(op, "w"))
    done += 1
    if done % 10 == 0:
        el = (time.time()-t0)/60
        n = len(glob.glob(f"{OUT}/e*.json"))
        print(f"[sam3] this worker {done} eps | cache {n}/1600 | {el:.1f} min "
              f"| {el/done:.2f} min/ep", flush=True)
print(f"[sam3] worker DONE {done} eps; {(time.time()-t0)/60:.1f} min", flush=True)

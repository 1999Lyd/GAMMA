#!/usr/bin/env python
"""Verify every patched grounded coordinate, per task ("no toxic labels").

Checks, per (task, instruction template):
  n            segments carrying this instruction across all seeds
  med (r,c)    median coordinate
  IQR          spread (objects are randomised per seed -> moderate spread is
               healthy for picks; fixed receptacles should cluster)
  border%      coordinates within 2 px of the frame border = projection left
               the image -> TOXIC
  fallback%    segments where the anchor list was shorter than the segment
               list (patcher used segment-end ee instead) -> flag
Also renders one overlay sheet per task (2 seeds, every segment's coordinate
drawn on its anchor-nearest extracted frame).

Outputs:
  results/rma/grounded_coord_report.md
  pipeline_rma/coord_check/task{N}.png
"""
import collections, json, os, pickle, re

import numpy as np
from PIL import Image, ImageDraw

O = os.path.expandvars("${ROBOMME_ROOT}")
D = f"{O}/data/rma_preprocessed_data"
V = os.path.expandvars("${GAMMA_DATA}/data/rma_sft_v1")
HERE = os.path.dirname(os.path.abspath(__file__))
OUTMD = os.path.expandvars("${GAMMA_DATA}/results/rma/grounded_coord_report.md")
os.makedirs(f"{HERE}/coord_check", exist_ok=True)

kf = json.load(open(f"{D}/meta/keyframes.json"))
sg = json.load(open(f"{D}/meta/segments.json"))
et = json.load(open(f"{D}/meta/episode_lengths.json"))
etask = json.load(open(f"{D}/meta/episode_task.json"))
C = re.compile(r"<\s*(-?\d+)\s*,\s*(-?\d+)\s*>")

offs, acc = {}, 0
for e in sorted(et, key=int):
    offs[e] = acc
    acc += et[e]

stats = collections.defaultdict(list)  # (task, instr) -> [(r, c, fallback)]
missing = 0
for e in sorted(kf, key=lambda x: int(x) if x.isdigit() else 10**9):
    if not e.isdigit():
        continue
    task = etask[e]["task"]
    seg = sg[e]
    anchors = kf[e]
    for si, s0 in enumerate(seg["seg_start"]):
        p = pickle.load(open(f"{D}/data/{offs[e] + s0}.pkl", "rb"))
        gsg = p.get("grounded_subgoal", "")
        m = C.search(gsg)
        if not m:
            missing += 1
            continue
        r, c = int(m.group(1)), int(m.group(2))
        fb = si >= len(anchors)
        instr = seg["instructions"][si].strip().lower()
        stats[(task, instr)].append((r, c, fb))

lines = ["# Grounded-coordinate verification (all tasks, all seeds)\n",
         f"segments without a coordinate: {missing}\n",
         "| task | instruction | n | med r,c | IQR r,c | border% | fallback% | flag |",
         "|---|---|---|---|---|---|---|---|"]
flags = 0
for (task, instr), rows in sorted(stats.items(),
                                  key=lambda kv: (int(kv[0][0][4:]), kv[0][1])):
    a = np.array([(r, c) for r, c, _ in rows], dtype=float)
    fb = 100 * sum(1 for *_, f in rows if f) / len(rows)
    border = 100 * np.mean((a <= 2).any(1) | (a >= 253).any(1))
    med = np.median(a, 0)
    iqr = np.percentile(a, 75, 0) - np.percentile(a, 25, 0)
    flag = []
    if border > 0:
        flag.append("BORDER")
    if fb > 0:
        flag.append("FALLBACK")
    if ("place" in instr or "pour" in instr or "close" in instr) and max(iqr) > 60:
        flag.append("SPREAD")
    if flag:
        flags += 1
    lines.append(f"| {task} | {instr[:44]} | {len(rows)} "
                 f"| {med[0]:.0f},{med[1]:.0f} | {iqr[0]:.0f},{iqr[1]:.0f} "
                 f"| {border:.1f} | {fb:.1f} | {' '.join(flag)} |")
lines.append(f"\nflagged rows: {flags}\n")
open(OUTMD, "w").write("\n".join(lines))
print(f"[verify] report -> {OUTMD}; flagged {flags} rows; missing {missing}")

# overlay sheets: 2 seeds per task, all segment coords on anchor frames
bytask = collections.defaultdict(list)
for e in kf:
    if e.isdigit():
        bytask[etask[e]["task"]].append(e)
for task, eps in sorted(bytask.items(), key=lambda kv: int(kv[0][4:])):
    tiles = []
    for e in eps[:2]:
        seed = etask[e]["seed"]
        man_p = f"{V}/manifests/{task}_seed{seed}.json"
        if not os.path.exists(man_p):
            continue
        man = json.load(open(man_p))
        for si, s0 in enumerate(sg[e]["seg_start"]):
            a = kf[e][si] if si < len(kf[e]) else s0 + sg[e]["seg_len"][si] - 1
            p = pickle.load(open(f"{D}/data/{offs[e] + s0}.pkl", "rb"))
            m = C.search(p.get("grounded_subgoal", ""))
            if not m:
                continue
            r, c = int(m.group(1)), int(m.group(2))
            g = min(man["grid"], key=lambda x: abs(x - a))
            fp = f"{V}/frames/{task}_seed{seed}/f{g}.jpg"
            if not os.path.exists(fp):
                continue
            im = Image.open(fp).resize((256, 256))
            d = ImageDraw.Draw(im)
            d.ellipse([c - 4, r - 4, c + 4, r + 4], outline=(255, 0, 0), width=2)
            d.text((2, 2), sg[e]["instructions"][si][:34], fill=(255, 255, 0))
            d.text((2, 244), f"s{seed}", fill=(0, 255, 255))
            tiles.append(im)
    if not tiles:
        continue
    cols = min(6, len(tiles))
    rows_n = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 258, rows_n * 258), (25, 25, 25))
    for i, im in enumerate(tiles):
        sheet.paste(im, ((i % cols) * 258, (i // cols) * 258))
    sheet.save(f"{HERE}/coord_check/{task}.png")
print("[verify] sheets ->", f"{HERE}/coord_check/")

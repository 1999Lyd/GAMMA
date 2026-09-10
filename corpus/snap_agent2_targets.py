#!/usr/bin/env python
"""Snap agent-2 target coordinates onto coordinates that actually exist in
that record's own memory bank (the SAM-derived numbers), so the target is
exactly copyable from the input rather than requiring the model to invent
sub-pixel precision it cannot read.

Keeps the untouched oracle string as `target_oracle` for reporting.
Radius default 15px: pi0.5 was trained with +/-8px grounding noise, and 96%
of snaps move the coordinate by <=5px.
"""
import json, os, re, sys
import numpy as np

SRC = sys.argv[1]
RAD = int(os.environ.get("SNAP_RADIUS", "15"))
COORD = re.compile(r"<\s*(\d+)\s*,\s*(\d+)\s*>")

for name in ("agent2_val", "agent2_train", "agent2_pred_val", "agent2_pred_train"):
    p = f"{SRC}/{name}.jsonl"
    if not os.path.exists(p):
        continue
    n = snapped = far = 0
    moves = []
    out = []
    for l in open(p):
        r = json.loads(l)
        # coordinates may be copied from the bank OR the current frame's
        # detections (agent-2 now sees both)
        bank_xy = [(int(a), int(b)) for a, b in COORD.findall(r.get("frame_det", ""))]
        bank_xy += [(int(a), int(b)) for ln in r["bank"]
                   for a, b in COORD.findall(ln)]
        r["target_oracle"] = r["target"]
        if bank_xy and COORD.search(r["target"]):
            def rep(m):
                global snapped, far
                x, y = int(m.group(1)), int(m.group(2))
                bx, by = min(bank_xy, key=lambda c: (c[0]-x)**2 + (c[1]-y)**2)
                d = float(np.hypot(bx-x, by-y))
                if d <= RAD:
                    if d > 0:
                        snapped += 1
                        moves.append(d)
                    return f"<{bx}, {by}>"
                far += 1
                return m.group(0)
            r["target"] = COORD.sub(rep, r["target"])
        n += 1
        out.append(json.dumps(r))
    with open(p, "w") as w:
        w.write("\n".join(out) + "\n")
    md = f"median move {np.median(moves):.1f}px" if moves else "-"
    print(f"{name}: {n} records, snapped {snapped} coords ({md}), "
          f"left alone (>{RAD}px) {far}")

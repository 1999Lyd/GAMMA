#!/usr/bin/env python3
"""Aggregate eval progress.json files (task -> {ep_idx: bool}).

Usage: agg_progress.py <glob-or-dir> [...]
Each arg may be a result dir (wam_0p8b_off_laneA) or a direct progress.json path.
Dirs are resolved via <dir>/ckpt79999/seed7/oracle/progress.json.
"""
import json, os, sys, glob

EVAL = os.path.expandvars("${ROBOMME_ROOT}/examples/robomme/runs/evaluation")

def resolve(a):
    if a.endswith(".json"):
        return [a]
    hits = []
    for d in sorted(glob.glob(os.path.join(EVAL, a)) or glob.glob(a)):
        f = os.path.join(d, "ckpt79999", "seed7", "oracle", "progress.json")
        if os.path.exists(f):
            hits.append(f)
    return hits

merged = {}
for arg in sys.argv[1:]:
    for f in resolve(arg):
        label = f.split("/evaluation/")[-1].split("/")[0]
        p = json.load(open(f))
        for task, eps in p.items():
            merged.setdefault(task, {})
            for i, ok in eps.items():
                if not isinstance(ok, bool):
                    continue   # "error" records are re-run on resume, never scored
                key = (label, i)
                merged[task][key] = ok

tot = suc = 0
for task in sorted(merged):
    eps = merged[task]
    s = sum(eps.values()); e = len(eps)
    tot += e; suc += s
    print(f"  {task:20s} {s:3d}/{e:<3d} {100*s/e:5.1f}%")
print(f"  {'TOTAL':20s} {suc:3d}/{tot:<3d} {100*suc/tot:5.1f}%" if tot else "  (no data)")

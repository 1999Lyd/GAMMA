#!/usr/bin/env python
"""Per-object grasp width prior from the recordings: gripper q6 at the pick
predicate's fire step, aggregated by object word. Writes grasp_widths.json."""
import os, sys, json, re, collections, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.expandvars('${ROBOMME_ROOT}/src/mme_vla_suite/dataset_builder'))
import rma_progress_predicates as pp
from rma_h5_utils import discover_task_dirs, discover_episodes, load_episode
pp.SEQUENTIAL = True
RAW = os.path.expandvars('${GAMMA_DATA}/data/RoboMemArena')
OBJ = ['cookies', 'tomato sauce', 'butter', 'popcorn', 'cream', 'chocolate', 'pudding', 'milk', 'wine', 'orange', 'sauce']
def obj_of(ins):
    i = ins.lower()
    for o in OBJ:
        if o in i: return o
    return 'other'
acc = collections.defaultdict(list); nseed = int(sys.argv[1]) if len(sys.argv) > 1 else 6
for td in discover_task_dirs(RAW):
    for ep in discover_episodes(td['path'])[:nseed]:
        e = load_episode(ep, load_images=False); st = e['state']
        segs = list(zip(e['seg_start'], e['seg_len'], e['instructions']))
        for k, ins, a, b, p in pp.predict(st, segs):
            if k == 'pick' and p is not None:
                q = st[p, 6]; acc[obj_of(ins)].append(float(q))
            if k == 'place' and p is not None:
                # width during the carry (20 steps before release)
                acc['carry:' + obj_of(ins)].append(float(np.median(st[max(0, p - 25):p - 3, 6])))
out = {k: {'median': round(float(np.median(v)), 4), 'p10': round(float(np.percentile(v, 10)), 4), 'p90': round(float(np.percentile(v, 90)), 4), 'n': len(v)} for k, v in acc.items()}
for k, v in sorted(out.items()): print(f'{k:22s} median {v["median"]:.4f}  p10 {v["p10"]:.4f}  p90 {v["p90"]:.4f}  n {v["n"]}')
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'grasp_widths.json'), 'w'), indent=1)

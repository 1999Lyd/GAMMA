#!/usr/bin/env python3
"""Paired-episode diagnosis: 0.8B harness-off vs harness-on (same seed 7,
same episode indices). Classifies each episode pair and, for episodes the
harness LOSES (on-fail & off-success), prints stall/premature evidence from
the on-arm trace."""
import json, glob, os, sys
from collections import defaultdict

EVAL = os.path.expandvars("${ROBOMME_ROOT}/examples/robomme/runs/evaluation")
FQ = os.path.expandvars("${GAMMA_DATA}/traces")

def load_progress(pattern):
    out = defaultdict(dict)
    for d in glob.glob(os.path.join(EVAL, pattern)):
        f = os.path.join(d, "ckpt79999", "seed7", "oracle", "progress.json")
        if not os.path.exists(f):
            continue
        for task, eps in json.load(open(f)).items():
            for i, ok in eps.items():
                out[task][int(i)] = bool(ok)
    return out

off = load_progress("wam_0p8b_off_lane*")
on  = load_progress("wam_0p8b_on_lane*")      # v18 partial
on.update({})  # keep v18 only; v19 handled separately where present
on19 = load_progress("wam_0p8b_on19_lane*")

def trace_stalls(task):
    """per-ep max consecutive oracle/agent2 content mismatch in ON traces."""
    stalls = {}
    for f in glob.glob(os.path.join(FQ, "wam_live_bank_0p8b_on*_lane*.jsonl")):
        cur = None; run = 0
        for line in open(f):
            if f'"{task}"' not in line:
                continue
            r = json.loads(line)
            if r.get("task") != task:
                continue
            if r["kind"] == "reset":
                cur = r["ep"]; run = 0
            elif r["kind"] == "tick" and cur is not None:
                o = (r.get("oracle") or "").split(" at <")[0]
                a = (r.get("agent2_out") or "").split(" at <")[0]
                if o and a and o != a:
                    run += 1
                    stalls[cur] = max(stalls.get(cur, 0), run)
                else:
                    run = 0
    return stalls

tasks = sys.argv[1:] or sorted(set(off) | set(on) | set(on19))
for task in tasks:
    o, h = off.get(task, {}), {**on.get(task, {}), **on19.get(task, {})}
    common = sorted(set(o) & set(h))
    if not common:
        print(f"== {task}: no paired episodes"); continue
    lose = [e for e in common if o[e] and not h[e]]
    win  = [e for e in common if h[e] and not o[e]]
    both = [e for e in common if not o[e] and not h[e]]
    stalls = trace_stalls(task)
    print(f"== {task}  ({len(common)} paired) off={sum(o[e] for e in common)} "
          f"on={sum(h[e] for e in common)}  harness-los={len(lose)} "
          f"harness-wins={len(win)} both-fail={len(both)}")
    if lose:
        print(f"   lost eps {lose}")
        print(f"   max-stall in lost eps: "
              f"{[(e, stalls.get(e, 0)) for e in lose]}")

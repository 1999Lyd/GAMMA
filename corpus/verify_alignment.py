#!/usr/bin/env python
"""Gates G1+G2 for EVERY task and EVERY event type.

For each target line, check
  IDENTITY: the coordinate is supported by a detection of a compatible class
            inside this record's own 5 frames (or the bank, for lines that
            reference an object hidden in this window);
  DYNAMICS: the window actually shows the change the line claims --
            appear -> newly present, cover -> cube gone/container there,
            move -> a container off-site, reach -> proximity, done -> the
            named object or the arm moved during the window.
"""
import json, os, re, sys, collections
import numpy as np

SRC = sys.argv[1]
SPLIT = os.environ.get("SPLIT", "val")
DET = os.environ.get("DET_DIR",
                     os.path.expandvars("${GAMMA_DATA}/data/wam_sft_v8/dets_sam3"))
COORD = re.compile(r"<\s*(\d+)\s*,\s*(\d+)\s*>")
TOL = 14.0

_cache = {}
def dets_of(ep):
    if ep not in _cache:
        with open(f"{DET}/e{ep}.json") as f:
            _cache[ep] = {int(k): [tuple(x) for x in v] for k, v in json.load(f).items()}
    return _cache[ep]

def kind_of(line):
    l = line.lower()
    if "initial scene" in l or "execution phase begins" in l: return "SCENE"
    if "appeared at" in l and "highlight" not in l:          return "APPEAR"
    if "highlight appeared" in l:                            return "HIGHLIGHT"
    if "were placed over the cubes" in l:                    return "COVER"
    if ("is being moved" in l or "moved to" in l
            or "moved from" in l):                           return "MOVE"
    if "reached the target" in l:                            return "REACH"
    if l.startswith("[event] completed") or "demo showed" in l: return "DONE"
    return "OTHER"

def win_dets(rec):
    out = []
    for d in rec["frame_dets"]:
        out.append([(m.group(1), int(m.group(2)), int(m.group(3)))
                    for m in re.finditer(r"(\w+)<\s*(\d+),\s*(\d+)>", d)])
    return out

per = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0, 0]))
for l in open(f"{SRC}/agent1_{SPLIT}.jsonl"):
    r = json.loads(l)
    if r.get("offset", 0) != 0 or r["target"].strip() == "NONE":
        continue
    fr = win_dets(r)
    if not fr: continue
    allxy = [(x, y) for f in fr for _, x, y in f]
    bankxy = [(int(a), int(b)) for ln in r["bank"] for a, b in COORD.findall(ln)]
    a, b = r["span"]
    D = dets_of(r["ep"])
    prev = [t for t in D if a - 16 <= t < a]
    prevxy = [(n, x, y) for t in prev for n, x, y in D[t]]
    for line in r["target"].split("\n"):
        k = kind_of(line)
        cs = [(int(u), int(v)) for u, v in COORD.findall(line)]
        c = per[r["family"]][k]
        c[0] += 1
        # IDENTITY
        ident = all(any((x-u)**2 + (y-v)**2 <= TOL**2 for u, v in allxy + bankxy)
                    for x, y in cs) if cs else True
        c[1] += ident
        # DYNAMICS
        dyn = True
        if k == "APPEAR" and cs:
            x, y = cs[0]
            was = any((x-u)**2 + (y-v)**2 <= TOL**2 for _, u, v in prevxy)
            now = any((x-u)**2 + (y-v)**2 <= TOL**2 for u, v in allxy)
            dyn = now and not was
        elif k == "HIGHLIGHT" and cs:
            x, y = cs[0]
            dyn = any(n == "highlight" and (x-u)**2 + (y-v)**2 <= (TOL+8)**2
                      for f in fr for n, u, v in f)
        elif k == "MOVE":
            # displacement test (matches gen): some container >=6px from its
            # nearest previous-window container -- the old >18px off-site rule
            # dropped every vus move line (7-18px per 8-step demo window)
            pc = [(u, v) for n_, u, v in prevxy if "container" in n_]
            dyn = any("container" in n and
                      min((u-su)**2 + (v-sv)**2 for su, sv in pc) >= 6**2
                      for f in fr for n, u, v in f) if pc else True
        elif k == "REACH" and cs:
            tx, ty = cs[0]
            dyn = any("cube" in n and (u-tx)**2 + (v-ty)**2 <= 22**2
                      for f in fr for n, u, v in f)
        elif k == "COVER":
            early = max((sum(1 for n, _, _ in f if n.endswith("_cube"))
                         for f in fr[:2]), default=0)
            late = min((sum(1 for n, _, _ in f if n.endswith("_cube"))
                        for f in fr[-2:]), default=0)
            pear = sum(1 for n, _, _ in prevxy if n.endswith("_cube"))
            dyn = late < early or (pear > 0 and late == 0)
        elif k == "DONE":
            mv = 0.0
            for i in range(len(fr)-1):
                A = [(x, y) for n, x, y in fr[i] if n == "arm"]
                B = [(x, y) for n, x, y in fr[i+1] if n == "arm"]
                if A and B:
                    mv += min(np.hypot(A[0][0]-q[0], A[0][1]-q[1]) for q in B)
            dyn = mv >= 3.0 or not cs
        c[2] += dyn

print(f"source={SRC} split={SPLIT}\n")
print(f"{'task':9s} {'event':10s} {'n':>5s} {'IDENTITY':>9s} {'DYNAMICS':>9s}")
tot = collections.defaultdict(lambda: [0, 0, 0])
for fam in sorted(per):
    for k in sorted(per[fam]):
        n, i, d = per[fam][k]
        t = tot[k]; t[0] += n; t[1] += i; t[2] += d
        flag = "  <-- LOW" if (i/n < 0.9 or d/n < 0.9) else ""
        print(f"{fam:9s} {k:10s} {n:5d} {i/n:9.2f} {d/n:9.2f}{flag}")
print(f"\n{'ALL':9s} {'event':10s} {'n':>5s} {'IDENTITY':>9s} {'DYNAMICS':>9s}")
for k in sorted(tot):
    n, i, d = tot[k]
    print(f"{'':9s} {k:10s} {n:5d} {i/n:9.2f} {d/n:9.2f}")
g = [sum(v[0] for v in tot.values()), sum(v[1] for v in tot.values()), sum(v[2] for v in tot.values())]
print(f"\nOVERALL identity {g[1]/g[0]:.3f}  dynamics {g[2]/g[0]:.3f}  (n={g[0]})")

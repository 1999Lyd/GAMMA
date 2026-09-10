#!/usr/bin/env python
"""Derivability audit: is every agent-1 target readable from that record's
own inputs — the 5 frames' SAM detections and the past memory bank?

Reports, per task, the fraction of event targets whose every coordinate is
matched (<=TOL px) by a detection in one of the record's 5 frames, or by a
coordinate already in the bank. Splits authored (observed:/initial scene/
execution phase) from annotation-derived (completed:/demo showed:) lines.
"""
import json, os, re, sys, collections
import numpy as np

SRC = sys.argv[1] if len(sys.argv) > 1 else \
    os.path.expandvars("${GAMMA_DATA}/data/wam_sft_v9")
SPLIT = os.environ.get("SPLIT", "val")
TOL = int(os.environ.get("TOL", "12"))
COORD = re.compile(r"<\s*(\d+)\s*,\s*(\d+)\s*>")

per = collections.defaultdict(lambda: collections.Counter())
bad = []
for l in open(f"{SRC}/agent1_{SPLIT}.jsonl"):
    r = json.loads(l)
    if r.get("offset", 0) != 0 or r["target"].strip() == "NONE":
        continue
    frame_xy = [(int(a), int(b)) for d in r["frame_dets"]
                for a, b in COORD.findall(d)]
    bank_xy = [(int(a), int(b)) for ln in r["bank"]
               for a, b in COORD.findall(ln)]
    for line in r["target"].split("\n"):
        tgt = [(int(a), int(b)) for a, b in COORD.findall(line)]
        kind = ("authored" if re.search(r"observed:|initial scene|execution phase begins", line)
                else "annotated")
        c = per[r["family"]]
        if not tgt:
            c[f"{kind}_nocoord"] += 1
            continue
        in_f = all(any((x-u)**2 + (y-v)**2 <= TOL*TOL for u, v in frame_xy)
                   for x, y in tgt)
        in_fb = all(any((x-u)**2 + (y-v)**2 <= TOL*TOL for u, v in frame_xy + bank_xy)
                    for x, y in tgt)
        c[f"{kind}_n"] += 1
        c[f"{kind}_frames"] += in_f
        c[f"{kind}_frames_or_bank"] += in_fb
        if not in_fb and len(bad) < 10:
            miss = [t for t in tgt
                    if not any((t[0]-u)**2 + (t[1]-v)**2 <= TOL*TOL
                               for u, v in frame_xy + bank_xy)]
            bad.append((r["family"], r["ep"], r["span"][1], kind, miss, line[:80]))

print(f"source={SRC}  split={SPLIT}  TOL={TOL}px")
print(f"{'task':10s} | {'authored n':>10s} {'frames':>7s} {'+bank':>7s} | "
      f"{'annot n':>8s} {'frames':>7s} {'+bank':>7s}")
tot = collections.Counter()
for f in sorted(per):
    c = per[f]
    for k in c:
        tot[k] += c[k]
    an, af, ab = c["authored_n"], c["authored_frames"], c["authored_frames_or_bank"]
    nn, nf, nb = c["annotated_n"], c["annotated_frames"], c["annotated_frames_or_bank"]
    g = lambda n, a, b: (f"{n:>10d} {a/n:7.3f} {b/n:7.3f}" if n else
                         f"{'-':>10s} {'-':>7s} {'-':>7s}")
    print(f"{f:10s} | {g(an, af, ab)} | {g(nn, nf, nb).strip():>24s}")
an, af, ab = tot["authored_n"], tot["authored_frames"], tot["authored_frames_or_bank"]
nn, nf, nb = tot["annotated_n"], tot["annotated_frames"], tot["annotated_frames_or_bank"]
print(f"\nAUTHORED  lines: n={an}  in-frames {af/max(an,1):.3f}  frames+bank {ab/max(an,1):.3f}")
print(f"ANNOTATED lines: n={nn}  in-frames {nf/max(nn,1):.3f}  frames+bank {nb/max(nn,1):.3f}")
tn = an + nn
print(f"ALL            : n={tn}  frames+bank {(ab+nb)/max(tn,1):.3f}")
if bad:
    print("\nstill not derivable:")
    for f, e, t, k, m, s in bad:
        print(f"  [{f} ep{e} t{t} {k}] missing {m}: {s}")

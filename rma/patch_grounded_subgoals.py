#!/usr/bin/env python
"""Rewrite grounded_subgoal fields of the RMA preprocessed dataset.

For every episode segment: coordinate = the segment's interaction-anchor ee
position (anchor pkl state[:3], the gripper grasp/release moment) projected
through the fixed agentview camera (rma_agentview_camera.json, verified by
overlay 2026-09-09). Every pkl in the segment gets
    grounded_subgoal        = f"{instruction} at <row, col>"
    grounded_subgoal_online = same
(simple_subgoal fields untouched). Convention <row, col> matches RoboMME's
name<x, y> transposed-centers convention.

Idempotent: a marker field _grounded_v1 skips already-patched pkls.
Env: WORKERS (default 24).
"""
import json, os, pickle
from multiprocessing import Pool

import numpy as np

O = os.path.expandvars("${ROBOMME_ROOT}")
D = f"{O}/data/rma_preprocessed_data"
HERE = os.path.dirname(os.path.abspath(__file__))
M = np.array(json.load(open(f"{HERE}/rma_agentview_camera.json"))["matrix"])

kf = json.load(open(f"{D}/meta/keyframes.json"))
sg = json.load(open(f"{D}/meta/segments.json"))
et = json.load(open(f"{D}/meta/episode_lengths.json"))


def project_rc(ee):
    """(row, col) exactly as robosuite camera_utils computes it."""
    # verified 2026-09-09 against robosuite camera_utils output on a live
    # point: (row, col) = (pix[1], pix[0]) directly; the saved matrix already
    # carries the row orientation (no flip).
    world = np.array([ee[0], ee[1], ee[2], 1.0])
    cam = M @ world
    pix = cam[:2] / cam[2]
    return int(np.clip(round(pix[1]), 0, 255)), int(np.clip(round(pix[0]), 0, 255))


offs, acc = {}, 0
for e in sorted(et, key=int):
    offs[e] = acc
    acc += et[e]


def release_step(base, s0, sl):
    """First step where the gripper gap opens beyond its early-segment hold
    level (+0.007) -- the object is at its final site there. None if the
    gap never opens (not a place-like segment)."""
    gaps = []
    for t in range(s0, s0 + sl):
        st = pickle.load(open(f"{D}/data/{base + t}.pkl", "rb"))["state"]
        gaps.append(float(st[6]) - float(st[7]))
    hold = float(np.median(gaps[: max(3, sl // 5)]))
    for i, g in enumerate(gaps):
        if i > sl // 4 and g > hold + 0.007:
            return s0 + i
    return None


def do_episode(e):
    base = offs[e]
    anchors = kf[e]
    seg = sg[e]
    n = 0
    for si, (s0, sl) in enumerate(zip(seg["seg_start"], seg["seg_len"])):
        a = anchors[si] if si < len(anchors) else s0 + sl - 1
        # v2 (2026-09-09): place-type segments anchor at the measured gripper
        # RELEASE, not the raw keyframe -- task20/21 first-place keyframes
        # were lift inflections pointing at the PICKUP location (toxic).
        if "place" in seg["instructions"][si].lower():
            rs = release_step(base, s0, sl)
            if rs is not None:
                a = rs
        ap = pickle.load(open(f"{D}/data/{base + a}.pkl", "rb"))
        row, col = project_rc(np.asarray(ap["state"][:3], dtype=float))
        instr = seg["instructions"][si].strip().lower()
        gs = f"{instr} at <{row}, {col}>"
        for t in range(s0, s0 + sl):
            fp = f"{D}/data/{base + t}.pkl"
            r = pickle.load(open(fp, "rb"))
            if r.get("_grounded_v2") == gs:
                continue
            r["grounded_subgoal"] = gs
            r["grounded_subgoal_online"] = gs
            r["_grounded_v2"] = gs
            with open(fp, "wb") as w:
                pickle.dump(r, w)
            n += 1
    return e, n


if __name__ == "__main__":
    eps = [e for e in kf if e.isdigit()]
    print(f"[patch] {len(eps)} episodes")
    tot = 0
    with Pool(int(os.environ.get("WORKERS", "24"))) as p:
        for i, (e, n) in enumerate(p.imap_unordered(do_episode, eps, chunksize=4)):
            tot += n
            if (i + 1) % 100 == 0:
                print(f"[patch] {i+1}/{len(eps)} episodes, {tot} pkls", flush=True)
    print(f"[patchv2] DONE {tot} pkls rewritten")

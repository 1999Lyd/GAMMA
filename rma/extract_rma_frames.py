#!/usr/bin/env python
"""RMA stage 1: reconstruct episodes and extract the agent-1 frame grid.

Per (task, seed): chain subtask hdf5s in planIndex order (filename contract
{words}_{planIndex}_seed{S}_task{T}.hdf5), verify ee continuity at every
join (gap <= GAP_MM mm, else quarantine the episode), then dump
agentview_rgb on the stride-5 grid (+ final frame) as jpgs and write a
manifest with structural subtask boundaries and offset keyframes.

Output: $OUT/frames/task{T}_seed{S}/f{step}.jpg
        $OUT/manifests/task{T}_seed{S}.json
        $OUT/inventory.json  (per-task seed census + quarantines)
Idempotent: episodes with an existing manifest are skipped.
Env: OUT (default rma_sft_v1), WORKERS (default 16), STRIDE (default 5).
"""
import collections, glob, json, math, os, re, sys
from multiprocessing import Pool

import h5py
import numpy as np
from PIL import Image

D = os.path.expandvars("${GAMMA_DATA}/data/RoboMemArena")
OUT = os.environ.get("OUT", os.path.expandvars("${GAMMA_DATA}/data/rma_sft_v1"))
STRIDE = int(os.environ.get("STRIDE", "5"))
WORKERS = int(os.environ.get("WORKERS", "16"))
GAP_MM = 5.0
PAT = re.compile(r"^(?P<words>.+)_(?P<pi>\d+)_seed(?P<seed>\d+)_task(?P<task>\d+)\.hdf5$")

os.makedirs(f"{OUT}/frames", exist_ok=True)
os.makedirs(f"{OUT}/manifests", exist_ok=True)


def episodes():
    eps = collections.defaultdict(dict)  # (task, seed) -> {pi: (words, path)}
    meta = {}
    for fam in ("Multi-Counting", "Multi-Occlusion", "Multi-Sequence",
                "Multi-Transferring"):
        for ds in sorted(glob.glob(f"{D}/{fam}/*_dataset")):
            base = os.path.basename(ds)
            tid = int(base.split("_")[0])
            instr = " ".join(base.split("_")[1:-1])
            meta[tid] = dict(family=fam, instruction=instr, dataset=base)
            for f in glob.glob(f"{ds}/subtask_data/*.hdf5"):
                m = PAT.match(os.path.basename(f))
                if not m:
                    print("[extract] UNPARSED filename:", f)
                    continue
                eps[(int(m["task"]), int(m["seed"]))][int(m["pi"])] = (
                    m["words"], f)
    return eps, meta


def do_episode(args):
    (task, seed), parts, meta = args
    name = f"task{task}_seed{seed}"
    man_path = f"{OUT}/manifests/{name}.json"
    if os.path.exists(man_path):
        return (task, seed, "done", None)
    pis = sorted(parts)
    if pis != list(range(len(pis))):
        return (task, seed, "quarantine", f"planIndex gap: {pis}")
    fdir = f"{OUT}/frames/{name}"
    os.makedirs(fdir, exist_ok=True)
    subs, off, prev_end = [], 0, None
    grid_all = []
    try:
        for pi in pis:
            words, path = parts[pi]
            with h5py.File(path, "r") as h:
                d = h["data/demo_0"]
                T = d["actions"].shape[0]
                ee = d["obs/ee_pos"]
                if prev_end is not None:
                    gap = float(np.linalg.norm(np.array(ee[0]) - prev_end))
                    if gap > GAP_MM / 1000.0:
                        return (task, seed, "quarantine",
                                f"chain gap {gap*1000:.1f}mm at pi={pi}")
                prev_end = np.array(ee[-1])
                kf = int(d["keyframe_indices"][0]) if "keyframe_indices" in d else None
                grid = sorted(set(list(range(0, T, STRIDE)) + [T - 1]))
                rgb = d["obs/agentview_rgb"]
                for t in grid:
                    Image.fromarray(rgb[t]).save(f"{fdir}/f{off + t}.jpg",
                                                 quality=90)
                grid_all += [off + t for t in grid]
                subs.append(dict(planIndex=pi, words=words, start=off,
                                 end=off + T,
                                 keyframe=(off + kf) if kf is not None else None))
                off += T
    except Exception as e:
        return (task, seed, "quarantine", f"read error: {e}")
    json.dump(dict(task=task, seed=seed, family=meta["family"],
                   instruction=meta["instruction"], length=off,
                   stride=STRIDE, grid=grid_all, subtasks=subs),
              open(man_path, "w"))
    return (task, seed, "ok", None)


if __name__ == "__main__":
    eps, meta = episodes()
    jobs = [((t, s), parts, meta[t]) for (t, s), parts in sorted(eps.items())]
    print(f"[extract] {len(jobs)} episodes over {len(meta)} tasks; "
          f"workers={WORKERS}", flush=True)
    inv = collections.defaultdict(lambda: dict(ok=0, done=0, quarantine=[]))
    with Pool(WORKERS) as p:
        for i, (task, seed, st, why) in enumerate(
                p.imap_unordered(do_episode, jobs, chunksize=1)):
            if st == "quarantine":
                inv[task]["quarantine"].append(dict(seed=seed, why=why))
                print(f"[extract] QUARANTINE task{task} seed{seed}: {why}",
                      flush=True)
            else:
                inv[task][st] += 1
            if (i + 1) % 100 == 0:
                print(f"[extract] {i+1}/{len(jobs)}", flush=True)
    for t in sorted(inv):
        q = inv[t]["quarantine"]
        print(f"[extract] task{t}: ok={inv[t]['ok']} prior={inv[t]['done']} "
              f"quarantined={len(q)}", flush=True)
    json.dump({str(t): v for t, v in inv.items()},
              open(f"{OUT}/inventory.json", "w"), indent=1)
    print("[extract] DONE", flush=True)

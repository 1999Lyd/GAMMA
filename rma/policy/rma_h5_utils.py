"""RoboMemArena (RMA) raw-HDF5 helpers: episode reconstruction from subtask segments.

The RMA release ships NO full_trajectory/ dirs; each task only has
``subtask_data/`` of ordered per-subtask segments.  Concatenating one seed's
segments in subtask order reconstructs the full episode (verified: inter-segment
ee_pos gap < 0.002).

Filename convention::

    <subtask_slug>_<order>_seed<seed>_task<N>.hdf5
        pick_wine_bottle_0_seed100_task10.hdf5        -> order 0
        pour_wine_into_mug_1st_1_seed100_task10.hdf5  -> order 1

The integer token immediately before ``_seed`` is the subtask order index.

DIFFERENCE vs ``robomme_setup/scratch/rma_common.py``
----------------------------------------------------
``rma_common.discover_episodes`` builds ONE GLOBAL ``{order: name}`` dict for the
whole task, so the last file scanned wins.  For tasks whose subtask identity
depends on the seed (task4 / task5: 3 distinct name triples at orders 6/7/8
across the 100 seeds) that mislabels ~2/3 of the episodes.  Here the per-episode
subtask names come from each seed's OWN files, and the per-segment natural
language subgoal is read from each segment file's ``language_instruction``
attribute (authoritative; filenames are only used for ordering).
"""

from __future__ import annotations

import glob
import os
import re
from collections import defaultdict

import h5py
import numpy as np

ADIM = 7
STATE_DIM = 8  # ee_states (6) + gripper_states (2)

SUITES = (
    "Multi-Counting",
    "Multi-Occlusion",
    "Multi-Sequence",
    "Multi-Transferring",
)

_FNAME_RE = re.compile(r"_seed(\d+)_task(\d+)\.hdf5$")
_STRIP_RE = re.compile(r"_seed\d+_task\d+\.hdf5$")


def parse_filename(fname: str) -> tuple[int, int, int] | None:
    """``(seed, task_number, subtask_order)`` from an RMA segment filename."""
    b = os.path.basename(fname)
    m = _FNAME_RE.search(b)
    if m is None:
        return None
    seed = int(m.group(1))
    task_num = int(m.group(2))
    order = int(b[: m.start()].split("_")[-1])
    return seed, task_num, order


def subtask_slug(fname: str) -> str:
    """``pick_wine_bottle_0`` from ``pick_wine_bottle_0_seed100_task10.hdf5``."""
    return _STRIP_RE.sub("", os.path.basename(fname))


def task_number_from_dir(task_dir: str) -> int:
    """``10`` from ``.../10_pour_wine_bottle_into_mug_dataset``."""
    base = os.path.basename(os.path.normpath(task_dir))
    m = re.match(r"(\d+)_", base)
    if m is None:
        raise ValueError(f"cannot parse task number from {task_dir!r}")
    return int(m.group(1))


def task_tag_from_dir(task_dir: str) -> str:
    return f"task{task_number_from_dir(task_dir)}"


def discover_task_dirs(root: str) -> list[dict]:
    """All 26 RMA task dirs under ``root``, sorted by task number.

    Each entry: ``{task_num, tag, suite, name, path}``.
    """
    out = []
    for suite in SUITES:
        suite_dir = os.path.join(root, suite)
        if not os.path.isdir(suite_dir):
            continue
        for name in sorted(os.listdir(suite_dir)):
            path = os.path.join(suite_dir, name)
            if not os.path.isdir(os.path.join(path, "subtask_data")):
                continue
            n = task_number_from_dir(path)
            out.append(
                dict(task_num=n, tag=f"task{n}", suite=suite, name=name, path=path)
            )
    out.sort(key=lambda d: d["task_num"])
    return out


def demo_key(h: h5py.File) -> str:
    keys = list(h["data"].keys())
    assert len(keys) == 1, (h.filename, keys)
    return keys[0]


def discover_episodes(task_dir: str) -> list[dict]:
    """Episodes for one task, ordered by ascending seed.

    Each episode: ``{seed, files, orders, slugs}`` where ``files`` is the seed's
    segment paths sorted by subtask order.  Subtask identity is resolved
    PER SEED (see module docstring).
    """
    sub = os.path.join(task_dir, "subtask_data")
    by_seed: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for f in glob.glob(os.path.join(sub, "*.hdf5")):
        p = parse_filename(f)
        if p is None:
            continue
        seed, _task_num, order = p
        by_seed[seed].append((order, f))

    episodes = []
    for seed in sorted(by_seed):
        segs = sorted(by_seed[seed], key=lambda x: x[0])
        orders = [o for o, _ in segs]
        assert orders == list(range(len(orders))), (task_dir, seed, orders)
        files = [f for _, f in segs]
        episodes.append(
            dict(
                seed=seed,
                files=files,
                orders=orders,
                slugs=[subtask_slug(f) for f in files],
            )
        )
    return episodes


def read_segment_meta(path: str) -> tuple[int, str]:
    """``(num_frames, language_instruction)`` for one segment file."""
    with h5py.File(path, "r") as h:
        g = h["data"][demo_key(h)]
        instr = g.attrs["language_instruction"]
        if isinstance(instr, bytes):
            instr = instr.decode()
        return int(g["actions"].shape[0]), str(instr)


def load_segment(path: str, load_images: bool = True) -> dict:
    """Full per-segment payload used by the converter."""
    with h5py.File(path, "r") as h:
        g = h["data"][demo_key(h)]
        instr = g.attrs["language_instruction"]
        if isinstance(instr, bytes):
            instr = instr.decode()
        out = {
            "actions": np.asarray(g["actions"], dtype=np.float64),
            "ee_states": np.asarray(g["obs"]["ee_states"], dtype=np.float64),
            "gripper_states": np.asarray(g["obs"]["gripper_states"], dtype=np.float64),
            "instruction": str(instr),
        }
        if load_images:
            # RMA frames are already display-oriented -> NO vertical flip here.
            out["agentview_rgb"] = np.asarray(g["obs"]["agentview_rgb"], dtype=np.uint8)
            out["eye_in_hand_rgb"] = np.asarray(
                g["obs"]["eye_in_hand_rgb"], dtype=np.uint8
            )
    return out


def load_episode(ep: dict, load_images: bool = True) -> dict:
    """Concatenate a seed's ordered segments into one episode.

    Returns ``actions (L,7) float64 CLIPPED to [-1,1]``, ``state (L,8) float64``
    (= ee_states | gripper_states), optionally the two ``(L,256,256,3) uint8``
    image stacks, plus ``seg_start`` / ``seg_len`` / ``instructions``.
    """
    acts, states, agv, eih = [], [], [], []
    seg_start, seg_len, instructions = [], [], []
    acc = 0
    for path in ep["files"]:
        s = load_segment(path, load_images=load_images)
        a = s["actions"]
        assert a.shape[1] == ADIM, (path, a.shape)
        n = a.shape[0]
        assert s["ee_states"].shape == (n, 6), (path, s["ee_states"].shape)
        assert s["gripper_states"].shape == (n, 2), (path, s["gripper_states"].shape)
        acts.append(a)
        states.append(np.concatenate([s["ee_states"], s["gripper_states"]], axis=1))
        if load_images:
            assert s["agentview_rgb"].shape == (n, 256, 256, 3)
            assert s["eye_in_hand_rgb"].shape == (n, 256, 256, 3)
            agv.append(s["agentview_rgb"])
            eih.append(s["eye_in_hand_rgb"])
        seg_start.append(acc)
        seg_len.append(n)
        instructions.append(s["instruction"])
        acc += n

    out = {
        # Native 7-d delta-OSC. A handful of gripper strays live outside the
        # nominal +/-1 (task10 has a 2.0; tasks 20-24 have 0.5) -> clip.
        "actions": np.clip(np.concatenate(acts, axis=0), -1.0, 1.0),
        "state": np.concatenate(states, axis=0),
        "seg_start": seg_start,
        "seg_len": seg_len,
        "instructions": instructions,
        "length": acc,
    }
    if load_images:
        out["agentview_rgb"] = np.concatenate(agv, axis=0)
        out["eye_in_hand_rgb"] = np.concatenate(eih, axis=0)
    return out

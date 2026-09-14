#!/usr/bin/env python
"""Tier-1 RoboMemArena (RMA) -> trainer-format conversion.

Produces the layout ``RoboMMEDataset`` reads::

    <out>/data/{i}.pkl                              flat, dense 0..N-1
    <out>/features/episode_{ep}/token_emb_{t}.npy   {image_emb_4x4, state_emb}
    <out>/meta/stats.json                           {execution_samples, total_samples}
    <out>/meta/episode_lengths.json                 {epis_idx: L}
    <out>/meta/episode_task.json                    {epis_idx: {task, suite, seed}}
    <out>/meta/segments.json                        {epis_idx: {seg_start, seg_len, instructions}}
    <out>/meta/val_episodes.json                    held-out episodes (10 highest seeds/task)
    <out>/meta/build_index.json                     pass-1 index (lets passes 2/3 rerun alone)

Three passes:

  pass 1  index      CPU  walk subtask_data/, per-seed segment order/lengths/subgoals
  pass 2  features   GPU  SigLIP -> pool_tokens_to_size(16) -> image_emb_4x4 (+ state_emb)
                          ONLY runs with --extract-features
  pass 3  samples    CPU  per-frame pkls (JPEG-encoded views) + meta json, multiprocess

Design decisions baked in (see the conversion spec):
  * actions: native 7-d delta-OSC clipped to [-1,1]; NO DeltaActions transform.
  * state:   8-d = ee_states(6) | gripper_states(2), matching the RMA eval
             harness's ``build_eval26_policy_input`` / ``_extract_state``.
  * images:  JPEG (quality 90) bytes in the pkl; the loader-side decode shim
             lives in ``mme_vla_suite.policies.rma_policy._parse_image``.
  * features: ONLY image_emb_4x4 + state_emb are stored; pos_emb_4x4 is
             recomputed on the fly by ``MemoryBuffer._load_pos_emb``.
  * exec_start_idx = 0 and is_demo = False everywhere (RMA has no video-demo
    prefix), so total_samples == execution_samples.

Examples::

    # counts + disk estimate only, no writes
    python -m mme_vla_suite.dataset_builder.build_rma_dataset --dry-run

    # full CPU conversion for one task
    python -m mme_vla_suite.dataset_builder.build_rma_dataset --tasks task10

    # feature extraction later, on ONE gpu
    CUDA_VISIBLE_DEVICES=<uuid> python -m mme_vla_suite.dataset_builder.build_rma_dataset \
        --extract-features --skip-samples
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from mme_vla_suite.dataset_builder import rma_h5_utils as rh
from mme_vla_suite.dataset_builder.rma_task_prompts import TASK_PROMPTS

DEFAULT_RMA_ROOT = os.path.expandvars("${DATA_ROOT}/rma_data/RoboMemArena")
DEFAULT_OUT = "data/rma_preprocessed_data"

ACTION_CHUNK_HORIZON = 20
JPEG_QUALITY = 90
N_VAL_SEEDS = 10
FEATURE_BATCH = 128          # same batch as scratch/rma_extract_feats.py
POOL_TOKENS = 16             # 4x4 -> image_emb_4x4
NUM_VIEWS = 1                # agentview only, matching the RoboMME feature layout

# rough per-frame fixed pickle overhead (keys, small arrays, strings), bytes
_PKL_FIXED_OVERHEAD = 1400


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# JPEG codec
# ---------------------------------------------------------------------------

def encode_jpeg(img: np.ndarray, quality: int = JPEG_QUALITY) -> bytes:
    """RGB uint8 (H,W,3) -> JPEG bytes."""
    import cv2

    ok, buf = cv2.imencode(
        ".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    )
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return buf.tobytes()


# ---------------------------------------------------------------------------
# pass 1: index
# ---------------------------------------------------------------------------

def build_index(rma_root: str, tasks: list[str] | None, threads: int = 64) -> list[dict]:
    """Walk the RMA tree; return one record per episode in (task, seed) order.

    ``epis_idx`` is DENSE over the SELECTED tasks (0..M-1), assigned in
    ascending task number then ascending seed, which is exactly the order the
    pkls are written in -- ``build_exec_index_map`` relies on that.
    """
    task_dirs = rh.discover_task_dirs(rma_root)
    if tasks:
        want = set(tasks)
        task_dirs = [t for t in task_dirs if t["tag"] in want]
        missing = want - {t["tag"] for t in task_dirs}
        if missing:
            raise ValueError(f"unknown/absent tasks: {sorted(missing)}")
    if not task_dirs:
        raise ValueError(f"no RMA task dirs under {rma_root}")

    records: list[dict] = []
    epis_idx = 0
    for td in task_dirs:
        t0 = time.time()
        episodes = rh.discover_episodes(td["path"])
        # read (length, language_instruction) for every segment in parallel:
        # NFS latency dominates, so threads (not processes) are the right tool.
        flat = [(i, j, f) for i, ep in enumerate(episodes) for j, f in enumerate(ep["files"])]
        with ThreadPoolExecutor(min(threads, max(4, len(flat)))) as pool:
            metas = list(pool.map(lambda x: rh.read_segment_meta(x[2]), flat, chunksize=8))
        per_ep: dict[int, dict[int, tuple[int, str]]] = {}
        for (i, j, _f), m in zip(flat, metas):
            per_ep.setdefault(i, {})[j] = m

        prompt = TASK_PROMPTS[td["tag"]]
        for i, ep in enumerate(episodes):
            lens = [per_ep[i][j][0] for j in range(len(ep["files"]))]
            instrs = [per_ep[i][j][1] for j in range(len(ep["files"]))]
            seg_start = np.concatenate([[0], np.cumsum(lens)[:-1]]).astype(int).tolist()
            records.append(
                dict(
                    epis_idx=epis_idx,
                    task=td["tag"],
                    task_num=td["task_num"],
                    suite=td["suite"],
                    task_name=td["name"],
                    seed=int(ep["seed"]),
                    files=ep["files"],
                    slugs=ep["slugs"],
                    seg_start=seg_start,
                    seg_len=[int(x) for x in lens],
                    instructions=instrs,
                    length=int(sum(lens)),
                    prompt=prompt,
                )
            )
            epis_idx += 1
        n = sum(r["length"] for r in records if r["task"] == td["tag"])
        log(
            f"  [{td['tag']:>6}] {td['suite']:<18} {len(episodes):>4} eps  "
            f"{n:>8} frames  ({time.time() - t0:.1f}s)"
        )
    return records


def assign_offsets(records: list[dict]) -> None:
    """Attach the first global pkl index of each episode (in place)."""
    g = 0
    for r in records:
        r["pkl_start"] = g
        g += r["length"]


# ---------------------------------------------------------------------------
# meta writing
# ---------------------------------------------------------------------------

def write_metas(records: list[dict], meta_dir: str) -> dict:
    os.makedirs(meta_dir, exist_ok=True)
    total = sum(r["length"] for r in records)

    stats = {"execution_samples": total, "total_samples": total}
    with open(os.path.join(meta_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    with open(os.path.join(meta_dir, "episode_lengths.json"), "w") as f:
        json.dump({str(r["epis_idx"]): r["length"] for r in records}, f)

    with open(os.path.join(meta_dir, "episode_task.json"), "w") as f:
        json.dump(
            {
                str(r["epis_idx"]): {
                    "task": r["task"],
                    "suite": r["suite"],
                    "seed": r["seed"],
                }
                for r in records
            },
            f,
        )

    with open(os.path.join(meta_dir, "segments.json"), "w") as f:
        json.dump(
            {
                str(r["epis_idx"]): {
                    "seg_start": r["seg_start"],
                    "seg_len": r["seg_len"],
                    "instructions": r["instructions"],
                }
                for r in records
            },
            f,
        )

    # held out: the N_VAL_SEEDS highest seed values of every task
    by_task: dict[str, list[dict]] = {}
    for r in records:
        by_task.setdefault(r["task"], []).append(r)
    per_task = {}
    for task, rs in by_task.items():
        top = sorted(rs, key=lambda r: r["seed"], reverse=True)[:N_VAL_SEEDS]
        per_task[task] = sorted(int(r["epis_idx"]) for r in top)
    val = sorted(e for v in per_task.values() for e in v)
    with open(os.path.join(meta_dir, "val_episodes.json"), "w") as f:
        json.dump({"n_val_per_task": N_VAL_SEEDS, "val_episodes": val, "per_task": per_task}, f)

    with open(os.path.join(meta_dir, "build_index.json"), "w") as f:
        json.dump(records, f)

    return stats


# ---------------------------------------------------------------------------
# pass 3: per-frame pkls (multiprocess, one episode per task unit)
# ---------------------------------------------------------------------------

_W: dict = {}


def _worker_init(data_dir: str, horizon: int, quality: int) -> None:
    import cv2

    cv2.setNumThreads(1)  # we parallelize across episodes, not inside cv2
    _W["data_dir"] = data_dir
    _W["horizon"] = horizon
    _W["quality"] = quality


def _write_episode(record: dict) -> tuple[int, int, int]:
    """Write one episode's per-frame pkls. Returns (epis_idx, n_frames, n_bytes)."""
    ep = rh.load_episode(record, load_images=True)
    L = ep["length"]
    assert L == record["length"], (record["epis_idx"], L, record["length"])

    horizon = _W["horizon"]
    quality = _W["quality"]
    data_dir = _W["data_dir"]

    actions = ep["actions"]                                   # (L,7) f64, clipped
    state = ep["state"]                                       # (L,8) f64
    idx = np.minimum(np.arange(L)[:, None] + np.arange(horizon)[None, :], L - 1)
    chunks = actions[idx]                                     # (L,H,7) edge-padded

    # per-frame segment id -> per-segment language_instruction
    seg_id = np.concatenate(
        [np.full(n, i, dtype=np.int32) for i, n in enumerate(record["seg_len"])]
    )
    instrs = [s.strip().lower() for s in record["instructions"]]
    prompt = record["prompt"].strip().lower()

    epis = int(record["epis_idx"])
    start = int(record["pkl_start"])
    nbytes = 0
    for t in range(L):
        sg = instrs[int(seg_id[t])]
        frame = {
            "image": encode_jpeg(ep["agentview_rgb"][t], quality),
            "wrist_image": encode_jpeg(ep["eye_in_hand_rgb"][t], quality),
            "state": state[t],
            "actions": chunks[t],
            "is_demo": np.array([False], dtype=np.bool_),
            "exec_start_idx": np.array([0], dtype=np.int32),
            "step_idx": np.array([t], dtype=np.int32),
            "epis_idx": np.array([epis], dtype=np.int32),
            "prompt": prompt,
            # RMA has no grounded (pixel-referenced) subgoals -> mirror the simple one.
            "simple_subgoal": sg,
            "grounded_subgoal": sg,
            "simple_subgoal_online": sg,
            "grounded_subgoal_online": sg,
        }
        path = os.path.join(data_dir, f"{start + t}.pkl")
        with open(path, "wb") as f:
            pickle.dump(frame, f, protocol=pickle.HIGHEST_PROTOCOL)
        nbytes += os.path.getsize(path)
    return epis, L, nbytes


def run_samples(records: list[dict], out: str, workers: int, horizon: int, quality: int) -> None:
    data_dir = os.path.join(out, "data")
    os.makedirs(data_dir, exist_ok=True)
    t0 = time.time()
    done_f = done_b = 0
    total_f = sum(r["length"] for r in records)
    ctx = mp.get_context("fork")
    with ctx.Pool(workers, initializer=_worker_init, initargs=(data_dir, horizon, quality)) as pool:
        for k, (_epis, L, nb) in enumerate(
            pool.imap_unordered(_write_episode, records, chunksize=1)
        ):
            done_f += L
            done_b += nb
            if (k + 1) % 25 == 0 or k + 1 == len(records):
                el = time.time() - t0
                log(
                    f"  pkl {k + 1}/{len(records)} eps  {done_f}/{total_f} frames  "
                    f"{done_b / 2**30:.2f} GiB  {done_f / max(el, 1e-6):.0f} f/s  "
                    f"eta {el * (total_f - done_f) / max(done_f, 1):.0f}s"
                )
    log(
        f"  pkl DONE {done_f} frames, {done_b / 2**30:.2f} GiB, "
        f"{done_b / max(done_f, 1):.0f} B/frame, {time.time() - t0:.0f}s"
    )


# ---------------------------------------------------------------------------
# pass 2: SigLIP features (GPU) -- gated behind --extract-features
# ---------------------------------------------------------------------------

def run_features(records: list[dict], out: str) -> None:
    """Encode agentview frames -> image_emb_4x4 (1,16,2048) bf16 + state_emb (8,) f64.

    Identical encoder route to scratch/rma_extract_feats.py: uint8 -> /255*2-1
    -> resize_with_pad(224) -> SigLIP So400m/14 -> pool_tokens_to_size(16).
    """
    import jax
    import jax.numpy as jnp

    from openpi.shared import image_tools
    from mme_vla_suite.shared.data_utils import pool_tokens_to_size
    from mme_vla_suite.shared.siglip_tokenizer import SigLipTokenizer

    feat_root = os.path.join(out, "features")
    os.makedirs(feat_root, exist_ok=True)
    tok = SigLipTokenizer()

    @jax.jit
    def encode_batch(images_u8):                 # (B, 256, 256, 3) uint8
        x = images_u8.astype(jnp.float32) / 255.0 * 2.0 - 1.0
        x = image_tools.resize_with_pad(x, 224, 224)
        x = x[:, None]                           # (B, 1, 224, 224, 3)
        emb = tok(x)                             # (B, 1, P, 2048) bf16
        return pool_tokens_to_size(emb, POOL_TOKENS)  # (B, 1, 16, 2048)

    t0 = time.time()
    n_done = 0
    total = sum(r["length"] for r in records)
    for k, rec in enumerate(records):
        epis = int(rec["epis_idx"])
        ep_dir = os.path.join(feat_root, f"episode_{epis}")
        os.makedirs(ep_dir, exist_ok=True)
        L = rec["length"]
        last = os.path.join(ep_dir, f"token_emb_{L - 1}.npy")
        if os.path.exists(last):                 # resumable: episode already done
            n_done += L
            continue

        ep = rh.load_episode(rec, load_images=True)
        assert ep["length"] == L
        imgs = ep["agentview_rgb"]
        state = ep["state"]
        out_chunks = []
        for b0 in range(0, L, FEATURE_BATCH):
            b1 = min(b0 + FEATURE_BATCH, L)
            chunk = imgs[b0:b1]
            if b1 - b0 < FEATURE_BATCH:          # pad to fixed batch (avoid re-jit)
                pad = np.repeat(chunk[-1:], FEATURE_BATCH - (b1 - b0), axis=0)
                chunk = np.concatenate([chunk, pad], axis=0)
            res = jax.device_get(encode_batch(jnp.asarray(chunk)))
            out_chunks.append(res[: b1 - b0])
        embs = np.concatenate(out_chunks, axis=0)  # (L, 1, 16, 2048) bfloat16
        assert embs.shape == (L, NUM_VIEWS, POOL_TOKENS, 2048), embs.shape

        for t in range(L):
            np.save(
                os.path.join(ep_dir, f"token_emb_{t}.npy"),
                {"image_emb_4x4": embs[t], "state_emb": state[t]},
            )
        n_done += L
        if (k + 1) % 10 == 0 or k + 1 == len(records):
            el = time.time() - t0
            log(
                f"  feat {k + 1}/{len(records)} eps  {n_done}/{total} frames  "
                f"{n_done / max(el, 1e-6):.0f} f/s  elapsed {el:.0f}s"
            )
    log(f"  feat DONE {n_done} frames in {time.time() - t0:.0f}s -> {feat_root}")


# ---------------------------------------------------------------------------
# dry run
# ---------------------------------------------------------------------------

def _sample_jpeg_bytes(records: list[dict], n_tasks_sample: int, quality: int) -> tuple[float, float]:
    """Mean JPEG size of (agentview, wrist) over a few frames per sampled task."""
    seen: dict[str, dict] = {}
    for r in records:
        seen.setdefault(r["task"], r)
    picks = list(seen.values())
    if len(picks) > n_tasks_sample:
        step = max(1, len(picks) // n_tasks_sample)
        picks = picks[::step][:n_tasks_sample]

    a_sizes, w_sizes = [], []
    for r in picks:
        s = rh.load_segment(r["files"][0], load_images=True)
        n = s["agentview_rgb"].shape[0]
        for t in np.linspace(0, n - 1, min(4, n)).astype(int):
            a_sizes.append(len(encode_jpeg(s["agentview_rgb"][t], quality)))
            w_sizes.append(len(encode_jpeg(s["eye_in_hand_rgb"][t], quality)))
    return float(np.mean(a_sizes)), float(np.mean(w_sizes))


def dry_run(records: list[dict], out: str, horizon: int, quality: int) -> None:
    total = sum(r["length"] for r in records)
    n_eps = len(records)
    tasks = sorted({r["task"] for r in records}, key=lambda t: int(t[4:]))
    lens = np.array([r["length"] for r in records])
    nseg = np.array([len(r["seg_len"]) for r in records])

    a_mean, w_mean = _sample_jpeg_bytes(records, n_tasks_sample=8, quality=quality)
    per_frame_pkl = a_mean + w_mean + horizon * 7 * 8 + 8 * 8 + _PKL_FIXED_OVERHEAD
    # features: image_emb_4x4 (1,16,2048) bf16 + state_emb (8,) f64 + npy/pickle hdr
    per_frame_feat = NUM_VIEWS * POOL_TOKENS * 2048 * 2 + 8 * 8 + 512

    log("")
    log("=========================== DRY RUN ===========================")
    log(f"output root        : {os.path.abspath(out)}")
    log(f"tasks              : {len(tasks)}  ({tasks[0]} .. {tasks[-1]})")
    log(f"episodes           : {n_eps}")
    log(f"frames (= samples) : {total}")
    log(f"exec samples       : {total}   (exec_start_idx=0, is_demo=False everywhere)")
    log(
        f"episode length     : min {lens.min()}  p50 {int(np.median(lens))}  "
        f"mean {lens.mean():.1f}  max {lens.max()}"
    )
    log(f"segments/episode   : min {nseg.min()}  max {nseg.max()}  mean {nseg.mean():.2f}")
    log(f"held-out episodes  : {N_VAL_SEEDS * len(tasks)}  ({N_VAL_SEEDS} highest seeds x {len(tasks)} tasks)")
    log("")
    log(f"JPEG q{quality} mean bytes : agentview {a_mean:,.0f}   wrist {w_mean:,.0f}")
    log(f"pkl  est / frame   : {per_frame_pkl:,.0f} B")
    log(f"feat est / frame   : {per_frame_feat:,.0f} B  (image_emb_4x4 bf16 + state_emb)")
    log("")
    log(f"data/   estimate   : {total * per_frame_pkl / 2**30:8.1f} GiB   ({total:,} files)")
    log(f"features/ estimate : {total * per_frame_feat / 2**30:8.1f} GiB   ({total:,} files)")
    log(
        f"TOTAL estimate     : {total * (per_frame_pkl + per_frame_feat) / 2**30:8.1f} GiB   "
        f"({2 * total:,} files)"
    )
    log("===============================================================")
    log("")
    log("per-task breakdown:")
    for t in tasks:
        rs = [r for r in records if r["task"] == t]
        f = sum(r["length"] for r in rs)
        log(
            f"  {t:>6}  {rs[0]['suite']:<18} {len(rs):>4} eps  {f:>8} frames  "
            f"{f * (per_frame_pkl + per_frame_feat) / 2**30:6.1f} GiB"
        )


# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rma-root", default=DEFAULT_RMA_ROOT)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--tasks", default="", help="comma list e.g. task1,task10 (default: all 26)")
    p.add_argument("--dry-run", action="store_true", help="counts + disk estimate only, no writes")
    p.add_argument("--extract-features", action="store_true", help="run pass 2 (GPU)")
    p.add_argument("--skip-samples", action="store_true", help="skip pass 3 (pkls)")
    p.add_argument("--skip-meta", action="store_true", help="skip writing meta/*.json")
    p.add_argument("--workers", type=int, default=32, help="pkl-writing processes")
    p.add_argument("--action-horizon", type=int, default=ACTION_CHUNK_HORIZON)
    p.add_argument("--jpeg-quality", type=int, default=JPEG_QUALITY)
    p.add_argument("--index-threads", type=int, default=64)
    p.add_argument("--force", action="store_true",
                   help="allow writing over an output root built from a DIFFERENT --tasks selection")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()] or None

    log(f"[pass 1] indexing {args.rma_root} ...")
    t0 = time.time()
    records = build_index(args.rma_root, tasks, threads=args.index_threads)
    assign_offsets(records)
    log(
        f"[pass 1] {len(records)} episodes, "
        f"{sum(r['length'] for r in records)} frames in {time.time() - t0:.1f}s"
    )

    if args.dry_run:
        dry_run(records, args.out, args.action_horizon, args.jpeg_quality)
        return 0

    # ``epis_idx`` and the flat pkl indices are DENSE OVER THE SELECTED TASKS, so
    # a task10-only build is NOT a prefix of the all-26 build. Refuse to mix two
    # different selections in one output root -- stale pkls would silently
    # corrupt the flat index.
    prev_path = os.path.join(args.out, "meta", "build_index.json")
    if os.path.exists(prev_path) and not args.force:
        with open(prev_path) as f:
            prev = json.load(f)
        prev_tasks = sorted({r["task"] for r in prev}, key=lambda t: int(t[4:]))
        cur_tasks = sorted({r["task"] for r in records}, key=lambda t: int(t[4:]))
        if prev_tasks != cur_tasks or len(prev) != len(records):
            raise SystemExit(
                f"REFUSING to write into {args.out}: it already holds a build of "
                f"{len(prev)} episodes over {prev_tasks}, but this run selects "
                f"{len(records)} episodes over {cur_tasks}. epis_idx / pkl indices "
                f"are dense over the SELECTED tasks, so mixing selections corrupts "
                f"the flat index. Delete data/ + meta/ + features/ first (or pass "
                f"--force if you know the old contents are gone)."
            )

    os.makedirs(args.out, exist_ok=True)
    if not args.skip_meta:
        stats = write_metas(records, os.path.join(args.out, "meta"))
        log(f"[meta] {stats}")

    if not args.skip_samples:
        log(f"[pass 3] writing pkls with {args.workers} workers ...")
        run_samples(records, args.out, args.workers, args.action_horizon, args.jpeg_quality)

    if args.extract_features:
        log("[pass 2] extracting SigLIP features (GPU) ...")
        run_features(records, args.out)

    log("[done]")
    return 0


if __name__ == "__main__":
    sys.exit(main())

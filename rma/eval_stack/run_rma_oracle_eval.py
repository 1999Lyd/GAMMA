"""RoboMemArena 1-26 sweep with the privileged grounded-subgoal oracle feeding
the subgoal-conditioned S1 policy (rma_ground_sg). Same arguments, loop,
scoring and output files as their run_all_tasks1_26.py; the only additions
are the oracle env hook and the "grounded_subgoal" field on every infer.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

BENCH = os.path.expandvars("${RMA_BENCH_ROOT}")
STACK = os.path.dirname(os.path.abspath(__file__))
for p in (f"{BENCH}/scripts", f"{BENCH}/libero_fork", STACK):
    if p not in sys.path:
        sys.path.insert(0, p)

import eval_common as ec  # noqa: E402
import eval_tasks2_26 as tasks26  # noqa: E402
import run_all_tasks1_26 as r  # noqa: E402
from policy_adapter import load_policy_adapter  # noqa: E402
import mme_rma_adapter  # noqa: E402
import rma_oracle_subgoal as oracle_mod  # noqa: E402


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = r.build_argparser().parse_args()
    if args.task_start < 1 or args.task_end > 26 or args.task_start > args.task_end:
        raise ValueError(f"Invalid task range: {args.task_start}..{args.task_end}")
    if args.num_trials_per_task < 1:
        raise ValueError("--num-trials-per-task must be >= 1.")

    oracle = oracle_mod.ORACLE
    oracle.num_steps_wait = args.num_steps_wait
    oracle_mod.install(oracle)          # must precede _patch_env_resolution
    mme_rma_adapter.SUBGOAL_PROVIDER = oracle.current

    adapter_kwargs = ec.parse_adapter_kwargs(args.adapter_kwargs)
    adapter = load_policy_adapter(args.adapter_spec, **adapter_kwargs)
    adapter.subgoal_provider = oracle.current          # bind on the loaded instance
    sys.modules[type(adapter).__module__].SUBGOAL_PROVIDER = oracle.current
    out_root = Path(args.out_root)
    video_root = out_root / "videos"
    video_root.mkdir(parents=True, exist_ok=True)
    os.environ["RMA_SUBGOAL_LOG"] = str(out_root / "subgoals.jsonl")
    assert getattr(adapter, "subgoal_provider", None) is not None

    tasks26._patch_env_resolution()
    results: list[dict[str, Any]] = []
    transitions_log = out_root / "oracle_transitions.jsonl"
    try:
        for task_id in range(args.task_start, args.task_end + 1):
            _, task_key = ec._resolve_task_id(task_id)
            bddl_path = ec._resolve_bddl_path(task_id)
            prompt = ec.get_prompt(task_key, bddl_path.stem)
            video_dir = video_root / f"task{task_id}"
            oracle.task_id = task_id
            oracle.episode = -1
            logging.info("task=%s seed_start=%s trials=%s prompt=%s [oracle plan: %s]",
                         task_id, args.seed, args.num_trials_per_task, prompt,
                         " | ".join(s["instr"] for s in oracle_mod.PLANS[task_id]))
            res = r._run_task(
                task_id=task_id,
                adapter=adapter,
                num_trials_per_task=args.num_trials_per_task,
                resize_size=args.resize_size,
                replan_steps=args.replan_steps,
                num_steps_wait=args.num_steps_wait,
                max_steps=args.max_steps,
                post_goal_steps=args.post_goal_steps,
                fail_on_extra_pour=args.fail_on_extra_pour,
                extra_pour_monitor_steps=args.extra_pour_monitor_steps,
                video_dir=video_dir,
                seed=args.seed,
            )
            results.append(res)
            with transitions_log.open("a") as fh:
                fh.write(json.dumps({"task": task_id, "last_episode_transitions": oracle.transitions}) + "\n")
            r._write_outputs(out_root, results, args.seed)   # incremental, survives a kill
    finally:
        close_fn = getattr(adapter, "close", None)
        if callable(close_fn):
            close_fn()

    r._write_outputs(out_root, results, args.seed)
    logging.info("Wrote oracle-fed Task %s-%s sweep outputs to %s", args.task_start, args.task_end, out_root)


if __name__ == "__main__":
    main()

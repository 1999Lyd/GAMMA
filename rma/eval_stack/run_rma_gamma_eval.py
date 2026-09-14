"""RoboMemArena 1-26 sweep with GAMMA feeding the subtask-conditioned S1 policy:
no simulator hooks -- the subgoal comes from the agent server (SAM-3 + writer +
harness over the robot's own camera and state). Same loop/scoring/outputs as
their run_all_tasks1_26.py; plans are the fixed per-task subtask sequences."""
from __future__ import annotations
import json, logging, os, sys
from pathlib import Path
from typing import Any
BENCH = os.path.expandvars("${RMA_BENCH_ROOT}"); STACK = os.path.dirname(os.path.abspath(__file__))
for p in (f"{BENCH}/scripts", f"{BENCH}/libero_fork", STACK):
    if p not in sys.path: sys.path.insert(0, p)
import eval_common as ec  # noqa: E402
import eval_tasks2_26 as tasks26  # noqa: E402
import run_all_tasks1_26 as r  # noqa: E402
from policy_adapter import load_policy_adapter  # noqa: E402
import mme_rma_adapter  # noqa: E402
from rma_gamma_provider import GammaProvider  # noqa: E402
PLANS = {int(k): [{"instr": s["instr"], "kind": s.get("kind"), "median": s.get("median")} for s in v] for k, v in json.load(open(f"{STACK}/rma_oracle_plans.json")).items()}

def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = r.build_argparser().parse_args()
    prov = GammaProvider()
    adapter_kwargs = ec.parse_adapter_kwargs(args.adapter_kwargs)
    adapter = load_policy_adapter(args.adapter_spec, **adapter_kwargs)
    prov.bind(adapter); mme_rma_adapter.SUBGOAL_PROVIDER = prov.current; sys.modules[type(adapter).__module__].SUBGOAL_PROVIDER = prov.current
    out_root = Path(args.out_root); (out_root / "videos").mkdir(parents=True, exist_ok=True)
    os.environ["RMA_SUBGOAL_LOG"] = str(out_root / "subgoals.jsonl"); os.environ["RMA_GAMMA_LOG"] = str(out_root / "gamma_ticks.jsonl")
    tasks26._patch_env_resolution(); results: list[dict[str, Any]] = []
    try:
        for task_id in range(args.task_start, args.task_end + 1):
            _, task_key = ec._resolve_task_id(task_id); bddl_path = ec._resolve_bddl_path(task_id)
            prompt = ec.get_prompt(task_key, bddl_path.stem)
            prov.task_id = task_id; prov.prompt = prompt; prov.plan = PLANS[task_id]; prov.episode = -1
            logging.info("task=%s trials=%s prompt=%s [plan: %s]", task_id, args.num_trials_per_task, prompt, " | ".join(x["instr"] for x in PLANS[task_id]))
            res = r._run_task(task_id=task_id, adapter=adapter, num_trials_per_task=args.num_trials_per_task, resize_size=args.resize_size,
                              replan_steps=args.replan_steps, num_steps_wait=args.num_steps_wait, max_steps=args.max_steps, post_goal_steps=args.post_goal_steps,
                              fail_on_extra_pour=args.fail_on_extra_pour, extra_pour_monitor_steps=args.extra_pour_monitor_steps,
                              video_dir=out_root / "videos" / f"task{task_id}", seed=args.seed)
            results.append(res); r._write_outputs(out_root, results, args.seed)
    finally:
        close_fn = getattr(adapter, "close", None)
        if callable(close_fn): close_fn()
    r._write_outputs(out_root, results, args.seed); logging.info("Wrote GAMMA sweep outputs to %s", out_root)

if __name__ == "__main__":
    main()

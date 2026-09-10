"""Env-only smoke for the grounded-subgoal oracle (no policy, no server).
For each task: build the env the benchmark way (with the oracle hook), reset
on eval seed 50, run the dummy-action prefix, then report the resolved body
names, the live projection of each pick object vs the training median, and
save an overlay of projections on the processed 256x256 policy frame.
Usage: MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=5 .venv/bin/python smoke_oracle_env.py 1 2 6
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
os.environ.setdefault("LIBERO_CONFIG_PATH", os.path.expandvars("${GAMMA_ROOT}/rma/eval_stack/libero_config"))
BENCH = os.path.expandvars("${RMA_BENCH_ROOT}")
STACK = os.path.dirname(os.path.abspath(__file__))
for p in (f"{BENCH}/scripts", f"{BENCH}/libero_fork", STACK):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
import cv2  # noqa: E402
import eval_common as ec  # noqa: E402
import eval_tasks2_26 as tasks26  # noqa: E402
from policy_adapter import build_eval26_policy_input  # noqa: E402
import rma_oracle_subgoal as om  # noqa: E402

OUT = os.path.join(STACK, "outputs", "oracle_smoke")
os.makedirs(OUT, exist_ok=True)


OFFSETS = []  # (task, obj, iqr, live-median)


def run(task_id: int, seed: int = 50, n_steps: int = 12):
    oracle = om.ORACLE
    oracle.task_id = task_id
    bddl_path = ec._resolve_bddl_path(task_id)
    _, task_key = ec._resolve_task_id(task_id)
    prompt = ec.get_prompt(task_key, bddl_path.stem)
    Env = ec._get_env_class()
    env = Env(bddl_file_name=str(bddl_path), camera_heights=256, camera_widths=256,
              ignore_done=True, reward_shaping=True, control_freq=20, initialization_noise=None)
    try:
        np.random.seed(seed)
        try:
            env.seed(seed)
        except AttributeError:
            pass
        obs = env.reset()
        for _ in range(n_steps):
            obs, _, _, _ = env.step(ec.LIBERO_DUMMY_ACTION)
        adapter_obs, frame, _ = build_eval26_policy_input(raw_obs=obs, prompt=prompt, resize_size=256)
        img = np.ascontiguousarray(frame[..., ::-1]).copy()  # RGB->BGR for cv2
        print(f"task{task_id} prompt={prompt!r} frame={frame.shape} current={oracle.current()!r}")
        for step in oracle.plan:
            med = tuple(step["median"])
            if step["kind"] == "pick":
                pos = tasks26._current_body_pos(env, step["obj"])
                rc = om.project_rc(pos)
                d = float(np.hypot(rc[0] - med[0], rc[1] - med[1]))
                print(f"   pick {step['obj']:20s} live=<{rc[0]},{rc[1]}> median=<{med[0]},{med[1]}> off={d:.1f}px iqr={step['iqr']}")
                OFFSETS.append({"task": task_id, "obj": step["obj"], "iqr": step["iqr"], "delta": [rc[0] - med[0], rc[1] - med[1]]})
                cv2.circle(img, (rc[1], rc[0]), 4, (0, 0, 255), 1)       # red: live projection
                cv2.circle(img, (med[1], med[0]), 4, (0, 255, 0), 1)     # green: training median
                cv2.putText(img, step["obj"].split("_")[0], (rc[1] + 5, rc[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
            else:
                cv2.drawMarker(img, (med[1], med[0]), (255, 128, 0), cv2.MARKER_CROSS, 8, 1)  # blue: fixture median
                cv2.putText(img, step["instr"][:22], (med[1] + 4, med[0] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 128, 0), 1)
        big = cv2.resize(img, (768, 768), interpolation=cv2.INTER_NEAREST)
        path = os.path.join(OUT, f"task{task_id}_seed{seed}.png")
        cv2.imwrite(path, big)
        print(f"   overlay -> {path}")
    finally:
        env.close()


if __name__ == "__main__":
    om.install()
    tasks26._patch_env_resolution()
    ids = [int(a) for a in sys.argv[1:]] or [1]
    for t in ids:
        try:
            run(t)
        except Exception as e:  # keep sweeping; report at the end
            print(f"task{t} FAILED: {e!r}")
    import json
    json.dump(OFFSETS, open(os.path.join(OUT, "pick_offsets.json"), "w"), indent=1)
    fixed = [o for o in OFFSETS if max(o["iqr"]) <= 3]
    byobj = {}
    for o in fixed:
        byobj.setdefault(o["obj"], []).append(o["delta"])
    print("per-object live-median offset on fixed layouts (row, col): mean / n")
    for k, v in sorted(byobj.items()):
        a = np.array(v); print(f"   {k:22s} {a.mean(0).round(1)}  sd={a.std(0).round(1)} n={len(v)}")

"""Env-only smoke test for the RoboMemArena eval stack (no policy, no server).

Builds one task-2 env exactly the way run_all_tasks1_26.py does (their
eval_common._get_env_class + task2_26_reference_stage._patch_env_resolution),
resets with eval seed 50, steps a few dummy actions, and reports the raw and
post-processed frame shapes.

Usage:
    MUJOCO_GL=egl .venv/bin/python smoke_env_only.py
    MUJOCO_GL=osmesa .venv/bin/python smoke_env_only.py
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
os.environ.setdefault(
    "LIBERO_CONFIG_PATH", os.path.expandvars("${GAMMA_ROOT}/rma/eval_stack/libero_config")
)

BENCH = os.path.expandvars("${RMA_BENCH_ROOT}")
for p in (f"{BENCH}/scripts", f"{BENCH}/libero_fork"):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

import eval_common as ec  # noqa: E402
import eval_tasks2_26 as tasks26  # noqa: E402
from policy_adapter import build_eval26_policy_input  # noqa: E402


def main() -> int:
    task_id = int(os.environ.get("SMOKE_TASK_ID", "2"))
    seed = int(os.environ.get("SMOKE_SEED", "50"))
    n_steps = int(os.environ.get("SMOKE_STEPS", "5"))

    print(f"[smoke] MUJOCO_GL={os.environ['MUJOCO_GL']}")
    bddl_path = ec._resolve_bddl_path(task_id)
    _, task_key = ec._resolve_task_id(task_id)
    prompt = ec.get_prompt(task_key, bddl_path.stem)
    print(f"[smoke] bddl={bddl_path}")
    print(f"[smoke] prompt={prompt!r}")

    # Same call order as run_all_tasks1_26.main().
    tasks26._patch_env_resolution()
    OffScreenRenderEnv = ec._get_env_class()

    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_path),
        camera_heights=256,
        camera_widths=256,
        ignore_done=True,
        reward_shaping=True,
        control_freq=20,
        initialization_noise=None,
    )
    try:
        np.random.seed(seed)
        try:
            env.seed(seed)
        except AttributeError:
            pass
        obs = env.reset()
        print(f"[smoke] reset ok; obs keys with 'image': "
              f"{sorted(k for k in obs if 'image' in k)}")
        print(f"[smoke] raw agentview_image        {obs['agentview_image'].shape} "
              f"{obs['agentview_image'].dtype}")
        print(f"[smoke] raw robot0_eye_in_hand_image {obs['robot0_eye_in_hand_image'].shape} "
              f"{obs['robot0_eye_in_hand_image'].dtype}")

        for t in range(n_steps):
            obs, _, done, _ = env.step(ec.LIBERO_DUMMY_ACTION)
            adapter_obs, main_img, wrist_img = build_eval26_policy_input(obs, prompt, 256)
            if t == n_steps - 1:
                print(f"[smoke] step {t}: processed main {main_img.shape} {main_img.dtype} "
                      f"min={main_img.min()} max={main_img.max()}")
                print(f"[smoke] step {t}: processed wrist {wrist_img.shape} {wrist_img.dtype}")
                st = adapter_obs["observation/state"]
                print(f"[smoke] step {t}: state dim={st.shape} value={np.round(st, 4).tolist()}")
                print(f"[smoke] step {t}: gripper_qpos dim="
                      f"{np.asarray(obs['robot0_gripper_qpos']).shape}")
        assert main_img.shape == (256, 256, 3), main_img.shape
        assert main_img.dtype == np.uint8
        assert st.shape == (8,), st.shape
        assert main_img.max() > 0, "rendered frame is all black"
        print("[smoke] PASS")
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())

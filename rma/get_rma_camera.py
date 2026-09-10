#!/usr/bin/env python
"""Extract the RMA agentview world->pixel camera matrix and verify it by
projecting subtask-anchor ee positions onto extracted dataset frames.

Run in the rma_eval_stack venv with EGL on a GPU:
  MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=<idx> MUJOCO_EGL_DEVICE_ID=<idx> \
    .venv/bin/python get_rma_camera.py
Outputs: rma_agentview_camera.json (4x4 world->pixel matrix, 256x256) and
overlay pngs under camera_check/.
"""
from __future__ import annotations
import json, os, sys

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
os.environ.setdefault("LIBERO_CONFIG_PATH",
    os.path.expandvars("${GAMMA_ROOT}/rma/eval_stack/libero_config"))
BENCH = os.path.expandvars("${RMA_BENCH_ROOT}")
for p in (f"{BENCH}/scripts", f"{BENCH}/libero_fork"):
    sys.path.insert(0, p)

import numpy as np
import eval_common as ec
import eval_tasks2_26 as tasks26
from robosuite.utils import camera_utils as CU

HERE = os.path.dirname(os.path.abspath(__file__))
bddl_path = ec._resolve_bddl_path(1)
tasks26._patch_env_resolution()
Env = ec._get_env_class()
env = Env(bddl_file_name=str(bddl_path), camera_heights=256, camera_widths=256,
          ignore_done=True, reward_shaping=True, control_freq=20,
          initialization_noise=None)
env.reset()
sim = env.env.sim if hasattr(env, "env") else env.sim
M = CU.get_camera_transform_matrix(sim, "agentview", 256, 256)
json.dump({"matrix": M.tolist(), "camera": "agentview", "hw": [256, 256]},
          open(f"{HERE}/rma_agentview_camera.json", "w"))
print("world->pixel matrix saved")

# sanity: project current ee site and compare with rendered gripper location
eef = sim.data.site_xpos[sim.model.site_name2id("gripper0_grip_site")]
px = CU.project_points_from_world_to_camera(
    np.array([eef]), M, 256, 256)
print("live ee world:", eef, "-> pixel (row, col):", px)

# dataset-side verification: project anchor ee positions onto frames
from PIL import Image, ImageDraw
O = os.path.expandvars("${ROBOMME_ROOT}")
V = os.path.expandvars("${GAMMA_DATA}/data/rma_sft_v1")
import pickle
kf = json.load(open(f"{O}/data/rma_preprocessed_data/meta/keyframes.json"))
sg = json.load(open(f"{O}/data/rma_preprocessed_data/meta/segments.json"))
et = json.load(open(f"{O}/data/rma_preprocessed_data/meta/episode_lengths.json"))
etask = json.load(open(f"{O}/data/rma_preprocessed_data/meta/episode_task.json"))
os.makedirs(f"{HERE}/camera_check", exist_ok=True)
# build flat index offsets: pkls are dense 0..N-1 in episode order
offs, acc = {}, 0
for e in sorted(et, key=int):
    offs[e] = acc; acc += et[e]
done = 0
for e in sorted(kf, key=lambda x: int(x) if x.isdigit() else 10**9):
    if not e.isdigit() or done >= 4:
        continue
    tinfo = etask[e]
    man = json.load(open(f"{V}/manifests/{tinfo['task']}_seed{tinfo['seed']}.json")) \
        if os.path.exists(f"{V}/manifests/{tinfo['task']}_seed{tinfo['seed']}.json") else None
    grid = man["grid"] if man else None
    im = None; d = None; used = None
    for si, a in enumerate(kf[e]):
        p = pickle.load(open(f"{O}/data/rma_preprocessed_data/data/{offs[e]+a}.pkl", "rb"))
        ee = np.array(p["state"][:3], dtype=float)
        px = CU.project_points_from_world_to_camera(np.array([ee]), M, 256, 256)[0]
        if im is None and grid is not None:
            g = min(grid, key=lambda x: abs(x - a))
            im = Image.open(f"{V}/frames/{tinfo['task']}_seed{tinfo['seed']}/f{g}.jpg").resize((512, 512))
            d = ImageDraw.Draw(im); used = g
        if d is not None:
            r, c = float(px[0]), float(px[1])
            d.ellipse([c*2-5, r*2-5, c*2+5, r*2+5], outline=(255, 0, 0), width=2)
            d.text((c*2+6, r*2-6), sg[e]["instructions"][si][:14], fill=(255, 255, 0))
    if im is not None:
        im.save(f"{HERE}/camera_check/ep{e}_{tinfo['task']}_seed{tinfo['seed']}_f{used}.png")
        done += 1
print("overlays ->", f"{HERE}/camera_check/")
env.close()

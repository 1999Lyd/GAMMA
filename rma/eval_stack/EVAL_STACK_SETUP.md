# RoboMemArena closed-loop eval stack

Everything needed to run the official RoboMemArena 1-26 benchmark
(`${RMA_BENCH_ROOT}`) against our
RMA-trained `mme_vla_suite` checkpoints, served over the openpi-client websocket
protocol.

Status: **installed and smoke-tested. No evaluation has been run.**

```
rma_eval_stack/
  .venv/                      dedicated eval venv (py3.10) -- NOT the training venv
  libero_config/config.yaml   isolated LIBERO config (avoids ~/.libero collisions)
  mme_rma_adapter.py          our BasePolicyAdapter -> websocket policy server
  run_rma_eval.sh             end-to-end runner (serve_policy + 1-26 sweep)
  smoke_env_only.py           env-stack smoke test (no policy, no server)
  smoke_adapter_protocol.py   adapter wire-protocol smoke test (no GPU, mock server)
  outputs/                    per-run results (created on demand)
```

---

## 1. Install (what was actually done)

Their `evaluation_benchmark/README.md` does **not** ask for a `pip install -e` of the
LIBERO fork -- `libero_fork/` ships no `setup.py`/`pyproject.toml`, and the repo root
README says to make it importable via `PYTHONPATH`. Their `eval_common.py` also
`sys.path.insert`s `libero_fork` itself. So the fork is wired in by path, not by
install.

```bash
uv venv --python 3.10 ${GAMMA_ROOT}/rma/eval_stack/.venv

uv pip install --python .venv/bin/python \
    "numpy<2" "robosuite==1.4.1" "mujoco==2.3.7" "bddl==1.0.1" \
    opencv-python-headless "imageio[ffmpeg]" scipy tqdm pyyaml easydict \
    cloudpickle termcolor "gym==0.25.2" matplotlib h5py PyOpenGL future

# websocket client for our policy server (editable, points at the training repo's
# packages/ dir -- it does not modify the training venv)
uv pip install --python .venv/bin/python \
    -e ${ROBOMME_ROOT}/packages/openpi-client \
    msgpack-numpy
```

Then a `.pth` in the venv's site-packages puts their fork and their scripts dir on
`sys.path` permanently:

```
.venv/lib/python3.10/site-packages/rma_eval_stack.pth
  ${RMA_BENCH_ROOT}/libero_fork
  ${RMA_BENCH_ROOT}/scripts
```

and `libero_config/config.yaml` is written by hand so that importing `libero` never
hits its interactive `input()` prompt (it prompts when `~/.libero/config.yaml` is
missing). Always export:

```bash
export LIBERO_CONFIG_PATH=${GAMMA_ROOT}/rma/eval_stack/libero_config
```

### robosuite version

`robosuite 1.5.2` (installed in `openpi_robocasa/.venv`, source at
`<workspace>/robosuite_src`) **cannot** run this fork: 1.5.x removed
`robosuite/environments/manipulation/single_arm_env.py`, and
`libero_fork/libero/envs/bddl_base_domain.py` imports `SingleArmEnv` from it. Hence
the pinned `robosuite==1.4.1` in this venv.

### Rendering

No `libOSMesa` on this host (`ldconfig -p | grep -i osmesa` is empty, and we have no
root to install it), so software rendering is not available -- **EGL on a GPU is
mandatory** even for the env-only smoke test. MuJoCo selects the EGL device with
`MUJOCO_EGL_DEVICE_ID` (a numeric index, not a UUID), and robosuite 1.4.1 asserts
that the index string appears inside `CUDA_VISIBLE_DEVICES`; `run_rma_eval.sh`
resolves the client UUID to its `nvidia-smi` index and handles that assert.

---

## 2. Run an evaluation

```bash
${GAMMA_ROOT}/rma/eval_stack/run_rma_eval.sh \
    <cfg> <exp> <ckpt_id> <server_gpu_uuid> <client_gpu_uuid>

# concrete example
${GAMMA_ROOT}/rma/eval_stack/run_rma_eval.sh \
    rma_fs_modul rma_fs_modul_s7 20000 \
    ${GPU5} \
    ${GPU1}
```

`<cfg>` is one of `rma_fs_modul` / `rma_dual_fs` / `rma_dualgate_fs`; the checkpoint
is looked up at `robomme_policy_learning_official/runs/ckpts/<cfg>/<exp>/<ckpt_id>`
(override with `RMA_CKPT_DIR`).

The script:

1. starts `scripts/serve_policy.py` in the training repo (`uv run`, training venv)
   on `$MME_RMA_PORT` (default 8105) pinned to `<server_gpu_uuid>`;
2. polls `http://127.0.0.1:8105/healthz` until the server answers;
3. runs their `scripts/run_all_tasks1_26.py` in *this* venv, on `<client_gpu_uuid>`
   for EGL rendering only (no JAX client-side), through
   `--adapter-spec .../mme_rma_adapter.py:build_adapter`;
4. tears the server down on exit.

Logs: `logs/rma_eval_<exp>_<ckpt>.log` (harness) and
`logs/rma_serve_<exp>_<ckpt>.log` (server).
Results: `rma_eval_stack/outputs/<exp>_<ckpt>/{episodes.tsv,task_summary.tsv,summary.json,aggregate.json,videos/}`.

The equivalent bare command, for reference:

```bash
cd ${RMA_BENCH_ROOT}
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
CUDA_VISIBLE_DEVICES=<client_gpu> MUJOCO_EGL_DEVICE_ID=<client_idx> \
LIBERO_CONFIG_PATH=${GAMMA_ROOT}/rma/eval_stack/libero_config \
${GAMMA_ROOT}/rma/eval_stack/.venv/bin/python scripts/run_all_tasks1_26.py \
  --adapter-spec ${GAMMA_ROOT}/rma/eval_stack/mme_rma_adapter.py:build_adapter \
  --adapter-kwargs '{"host": "127.0.0.1", "port": 8105}' \
  --task-start 1 --task-end 26 \
  --num-trials-per-task 50 --seed 50 \
  --replan-steps 10 --max-steps 2500 --num-steps-wait 10 --resize-size 256 \
  --fail-on-extra-pour \
  --out-root ${GAMMA_ROOT}/rma/eval_stack/outputs/<exp>_<ckpt>
```

### Fixed protocol

| flag | value | note |
|---|---|---|
| `--seed` | 50 | eval layouts only |
| `--num-trials-per-task` | 50 | -> seeds 50..99 |
| `--replan-steps` | 10 | first 10 of our 20-step chunk are executed |
| `--max-steps` | 2500 | |
| `--num-steps-wait` | 10 | dummy `[0]*6+[-1]` actions before the policy runs |
| `--resize-size` | 256 | |
| extra-pour rejection | ON | harness default; 30-step post-stage monitor |

`run_rma_eval.sh` hard-codes these and refuses to start if the seed window would
reach 100.

### SEED RULE -- CONTAMINATION WARNING

**Eval seeds are 50-99 and nothing else. Never use seed >= 100.**

Verified directly against the training data: every episode in
`robomme_policy_learning_official/data/rma_preprocessed_data/meta/episode_task.json`
has `seed >= 100` (observed range 100..813, 273 distinct seeds). Seeds >= 100 are the
object layouts our RMA checkpoints were trained on -- evaluating there is train-set
leakage, not evaluation. Note that their own harness defaults are
`--num-trials-per-task 51` (README) and `--seed 100` for the single-task
`eval_tasks2_26.py` entry point, so **both defaults are unsafe for us**; always pass
`--seed 50 --num-trials-per-task 50` explicitly, or use `run_rma_eval.sh`.

---

## 3. Adapter

`mme_rma_adapter.py` implements their `BasePolicyAdapter`:

- `reset()` -- `client.reset()`, asserts `{"reset_finished": True}`, clears the local
  pending buffer. Called once per episode by their `run_eval_task`.
- `observe(obs, prompt, resize_size)` -- their optional per-step hook, called at
  **every** env step before the replan check. We use it to accumulate the
  agentview frame + 8-d state, which is what keeps our perceptual (frame-sampling)
  memory fed with the full per-step history rather than only replan boundaries.
- `infer_actions(obs, prompt, resize_size)` -- ships the accumulated buffer via
  `client.add_buffer({"images": (T,1,256,256,3) uint8, "state": (T,8) float32,
  "add_buffer": True, "exec_start_idx": 0})`, then `client.infer({observation/image,
  observation/wrist_image, observation/state, prompt})`, then clears the buffer.
  Returns `[20, 7]` float32.
- `close()` -- closes the websocket (their sweep calls it in a `finally`).

This mirrors `robomme_policy_learning_official/examples/robomme/eval.py`
(`get_action_chunk` + `epstate.clear_buffers()` right after each infer) and
`examples/robomme/utils.py::pack_buffer` exactly: the server-side
`MME_VLA_Policy.add_buffer` keeps a monotonic `step_idx` and accumulates, so the
client only ever ships the frames observed **since the last infer**, whose last
element is the frame the infer is conditioned on. `exec_start_idx` is always 0 --
it is non-zero only for RoboMME tasks with a video-demo prefix, and RoboMemArena has
none.

Images are taken from `obs["observation/image"]` / `obs["observation/wrist_image"]`,
which their `build_eval26_policy_input` has **already** flipped (`np.flipud`) and
resized. The adapter does not flip or resize again.

Prompts are lowercased: `build_rma_dataset.py:265` writes
`record["prompt"].strip().lower()` into the training pkls (verified: the task2 pkl
prompt is `'pick and place butter into the basket, ...'`), while the harness serves
the capitalized `TASK_PROMPTS` strings. The paligemma tokenizer lowercases again, so
this is belt-and-braces.

Connection: `MME_RMA_HOST` / `MME_RMA_PORT` (default `127.0.0.1:8105`), overridable
per run via `--adapter-kwargs '{"host": ..., "port": ...}'`.

Other `--adapter-kwargs`: `use_history` (default true), `lowercase_prompt` (true),
`action_horizon` (20), `strict_horizon` (true), `state_convention` (`"harness"`, see
below).

---

## 4. Smoke tests

```bash
# 1. env stack only -- no policy, no server. Needs EGL (see Deviations).
cd ${GAMMA_ROOT}/rma/eval_stack
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
CUDA_VISIBLE_DEVICES=${GPU5} MUJOCO_EGL_DEVICE_ID=5 \
.venv/bin/python smoke_env_only.py

# 2. adapter wire protocol -- CPU only, mock server, no GPU
CUDA_VISIBLE_DEVICES= .venv/bin/python smoke_adapter_protocol.py
```

Both pass. `smoke_env_only.py` on task 2 / seed 50 reports a raw
`agentview_image (480, 640, 3) uint8`, processed `(256, 256, 3) uint8`, and an 8-d
state; `smoke_adapter_protocol.py` verifies the `reset / add_buffer / infer` message
order, `(1,1,256,256,3)` then `(10,1,256,256,3)` buffer shapes across replan cycles,
`exec_start_idx == 0`, buffer-tail == infer frame, and the lowercased prompt.

A third, end-to-end check was also run once (their `run_all_tasks1_26.py`, task 2,
1 trial, `--max-steps 40`, against the mock server): the full loop -- env, adapter,
websocket, video encode, `episodes.tsv`/`aggregate.json` -- works.

---

## 5. Deviations and gotchas

**No `pip install -e` of `libero_fork`.** It has no packaging metadata; it is a
path-only import, per their own README. Wired via a `.pth` instead.

**robosuite 1.4.1, not the 1.5.2 already on this box.** See above.

**EGL is mandatory; osmesa is unavailable.** `libOSMesa` is not installed and we
cannot install it, so `MUJOCO_GL=osmesa` fails at import with
`AttributeError: 'NoneType' object has no attribute 'glGetError'` inside PyOpenGL.
Every run, including the env smoke test, needs a GPU for rendering.

**The harness renders at 480x640, not 256x256 -- and this is correct, do not
"fix" it.** `run_all_tasks1_26.py` calls `task2_26_reference_stage._patch_env_resolution()`,
which monkey-patches `OffScreenRenderEnv.__init__` to force
`camera_heights=480, camera_widths=640`, overriding the `camera_heights=256`
that `run_eval_task` passes. Frames are then flipped and squashed (non-uniformly) to
256x256. We verified this is what the training data looks like: comparing the RMA
training frame for task2/seed100 against both renders,

| render | MAE vs training frame | pearson |
|---|---|---|
| forced 480x640 -> 256x256 (what the harness does) | **2.63** | **0.9851** |
| native square 256x256 | 19.20 | 0.7934 |

So the aspect squash is baked into the released dataset and the patch is required
for train/eval parity. **Consequence:** any code path that builds the env *without*
calling `_patch_env_resolution()` silently produces out-of-distribution square
renders. `run_all_tasks1_26.py` and `eval_tasks2_26.py --task-id ...` both call it;
importing `run_eval_task` directly does not.

**Axis-angle state convention mismatch (the big one).** Their
`policy_adapter._quat2axisangle` canonicalises the quaternion to the positive-w
hemisphere (`if quat[3] < 0: quat = -quat`) before converting;
`robosuite.utils.transform_utils.quat2axisangle` -- which produced the `ee_states`
in the released RoboMemArena demonstrations we train on -- does not. Measured on
task 2 at the reset pose:

```
eef_quat                  [ 0.9996  0.0002 -0.0284 -0.0000]
harness  _quat2axisangle  [-3.1403 -0.0008  0.0892]
robosuite quat2axisangle  [ 3.1403  0.0008 -0.0892]   <- equals the RMA training state
```

The training distribution of state dim 3, over a 160-frame sample of our RMA pkls,
is `[+2.4173, +4.7154]` -- strictly positive and exceeding pi, which the harness's
canonical form (|theta| <= pi) cannot represent at all. Under q01/q99 normalisation
the harness value at reset maps to roughly -5.9 where all training values map into
[-1, 1]. Expect this to cost real success rate on every task.

This is *their* adapter contract, and their own reference adapter
(`openpi_minimal_runtime/robocerebra_adapter.py`) uses the same canonicalising
helper, so it presumably also applies to their published numbers -- but it means the
public dataset and the public eval adapter disagree, and any model trained on the
released `ee_states` is evaluated off-distribution.

The adapter therefore defaults to `state_convention="harness"` (their contract
verbatim -- the only protocol-legal setting, and the only one comparable to their
leaderboard). `state_convention="robosuite"` recomputes dims 3..5 from
`robot0_eef_quat` with the raw conversion to match our training data:

```bash
RMA_STATE_CONVENTION=robosuite ./run_rma_eval.sh <cfg> <exp> <ckpt> <srv_gpu> <cli_gpu>
```

Treat any such run as an off-protocol diagnostic and label it as such. If the gap
between the two is large, that is a finding about the benchmark, not about our
memory arms, and both numbers should be reported.

**Task-count and metric naming.** `CSR` in the output files is the average
*stage-completion* percentage (stored in the legacy `goal_success_rate` field), and
`TSR` is the strict all-required-stages success rate. Drawer tasks (4, 5, 11, 12, 13,
14, 17) exclude their final "close drawer" stage from the TSR denominator; microwave
tasks do not require the final close. Counting-pour tasks (6, 7, 8, 9, 10, 15, 16, 22)
additionally fail if a third pour is detected inside a 30-step monitor after pour 2.

**One env instance is reused across all 50 episodes of a task** (`run_eval_task`
constructs the env once and calls `env.seed(s); env.reset()` per episode). Our
adapter's per-episode `reset()` drops the server-side memory buffer, so no history
leaks between episodes.

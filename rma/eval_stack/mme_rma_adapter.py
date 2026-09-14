"""RoboMemArena eval adapter for our RMA-trained mme_vla_suite policies.

Implements the RoboMemArena `BasePolicyAdapter` contract
(`rma_eval_repo/evaluation_benchmark/scripts/policy_adapter.py`) on top of our
openpi-client websocket policy server
(`robomme_policy_learning_official/scripts/serve_policy.py`).

Protocol mirrored from `robomme_policy_learning_official/examples/robomme/eval.py`
(`EpisodeEvaluator.eval_each_episode` / `get_action_chunk`) and
`examples/robomme/utils.py::pack_buffer`:

  reset   -> client.reset(), poll until {"reset_finished": True}
  step    -> observe() appends (agentview frame, 8-d state) to a pending buffer
  infer   -> client.add_buffer({"images": (T,1,H,W,3) uint8,
                                "state":  (T,8) float32,
                                "add_buffer": True,
                                "exec_start_idx": 0}) and poll until
             {"add_buffer_finished": True}, then client.infer({...}) and
             clear the pending buffer.

The server-side `MME_VLA_Policy.add_buffer` accumulates the frames into its own
`mem_buffer` (SigLIP-encoded, keyed by a monotonic step index), so the client
only ever ships the frames observed *since the last infer* -- the last of which
is always the frame the infer is conditioned on.  That is exactly what the
RoboMME eval client does (it calls `epstate.clear_buffers()` right after every
infer).  `exec_start_idx` stays 0: it only differs from 0 for RoboMME tasks that
prepend a video demo, and RoboMemArena has none.

Notes on the observation contract:
  * `obs["observation/image"]` / `obs["observation/wrist_image"]` are produced by
    their `build_eval26_policy_input` -> `_process_image_match_eval26`, which
    ALREADY applies `np.flipud` and the resize to `resize_size`.  Do not flip or
    resize again.
  * `obs["observation/state"]` is already the 8-d
    `eef_pos(3) | quat2axisangle(eef_quat)(3) | gripper_qpos(2)` vector, which is
    the same *layout* as the RMA training state (`ee_states(6) | gripper_states(2)`)
    -- but NOT the same rotation *convention*, see below.

`state_convention` (train/eval mismatch, read this before reporting numbers)
--------------------------------------------------------------------------
Their `policy_adapter._quat2axisangle` canonicalises the quaternion to the
positive-w hemisphere (`if quat[3] < 0: quat = -quat`) before converting, so its
rotation triplet always has |theta| <= pi.  `robosuite.utils.transform_utils.
quat2axisangle` -- which produced `ee_states` in the released RoboMemArena
demonstrations we train on -- does NOT do that flip, so its theta ranges over
[0, 2*pi).  Measured on task2 at the reset pose:

    eef_quat                  [ 0.9996  0.0002 -0.0284 -0.0000]
    harness  _quat2axisangle  [-3.1403 -0.0008  0.0892]
    robosuite quat2axisangle  [ 3.1403  0.0008 -0.0892]   <- RMA training value

and over a 160-frame sample of our RMA training pkls, state dim 3 lives in
[+2.4173, +4.7154] (never negative, and exceeding pi), which the harness's
convention cannot even represent.  Feeding harness-convention state to a model
trained on the raw convention puts dims 3..5 far outside the q01/q99
normalisation range for most timesteps.

`state_convention="harness"` (DEFAULT) reproduces their adapter contract exactly
and is what any comparison against their published numbers must use.
`state_convention="robosuite"` recomputes dims 3..5 from `robot0_eef_quat` with
the raw robosuite conversion, i.e. matches our training data. Treat any run with
`"robosuite"` as an off-protocol diagnostic, not a benchmark result.
  * prompts are lowercased: `build_rma_dataset.py` writes
    `record["prompt"].strip().lower()` into the training pkls (and the paligemma
    tokenizer lowercases again), while the harness serves the capitalized
    `TASK_PROMPTS` strings.

Connection settings come from `MME_RMA_HOST` / `MME_RMA_PORT`
(default 127.0.0.1:8105) and can be overridden per-run via `--adapter-kwargs`.

Usage:
    python scripts/run_all_tasks1_26.py \
      --adapter-spec ${GAMMA_ROOT}/rma/eval_stack/mme_rma_adapter.py:build_adapter \
      ...
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import time
from typing import Any

import numpy as np

# Their scripts/ dir holds `policy_adapter`. When the harness loads us through
# `--adapter-spec /abs/path.py:build_adapter` the scripts dir is normally
# sys.path[0] already, but make the import work standalone too.
_BENCH_SCRIPTS = os.environ.get(
    "MME_RMA_BENCH_SCRIPTS",
    os.path.expandvars("${RMA_BENCH_ROOT}/scripts"),
)
if _BENCH_SCRIPTS not in sys.path:
    sys.path.insert(0, _BENCH_SCRIPTS)

from policy_adapter import BasePolicyAdapter  # noqa: E402

from openpi_client import websocket_client_policy as _websocket_client_policy  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_HOST = os.environ.get("MME_RMA_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.environ.get("MME_RMA_PORT", "8105"))

# Our RMA configs (rma_fs_modul / rma_dual_fs / rma_dualgate_fs) all train
# HistoryPi0Config(action_horizon=20).  The harness only executes the first
# `--replan-steps` (=10) rows of whatever we return.
DEFAULT_ACTION_HORIZON = 20
ACTION_DIM = 7

STATE_CONVENTIONS = ("harness", "robosuite")


def _robosuite_quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """`robosuite.utils.transform_utils.quat2axisangle`, inlined verbatim.

    Unlike the harness's `_quat2axisangle` this does NOT flip the quaternion to
    the positive-w hemisphere, so theta ranges over [0, 2*pi) -- which is the
    convention the released RoboMemArena `ee_states` (our training state) use.
    """
    q = np.asarray(quat, dtype=np.float64).copy()
    w = float(np.clip(q[3], -1.0, 1.0))
    den = math.sqrt(max(0.0, 1.0 - w * w))
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)
    return ((q[:3] * 2.0 * math.acos(w)) / den).astype(np.float32)


# Set by run_rma_oracle_eval.py: callable() -> current grounded subgoal string.
SUBGOAL_PROVIDER = None


class MMERMAAdapter(BasePolicyAdapter):
    """Websocket adapter for a served mme_vla_suite RMA checkpoint."""

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        *,
        use_history: bool = True,
        lowercase_prompt: bool = True,
        action_horizon: int = DEFAULT_ACTION_HORIZON,
        poll_interval: float = 0.05,
        strict_horizon: bool = True,
        state_convention: str = "harness",
        prompt_from_subgoal: bool = False,
    ) -> None:
        if state_convention not in STATE_CONVENTIONS:
            raise ValueError(f"state_convention must be one of {STATE_CONVENTIONS}, got {state_convention!r}")
        self.state_convention = state_convention
        if state_convention != "harness":
            logger.warning(
                "MMERMAAdapter running OFF-PROTOCOL with state_convention=%r; results are a "
                "diagnostic, not a benchmark number.",
                state_convention,
            )
        self.host = host or DEFAULT_HOST
        self.port = int(port if port is not None else DEFAULT_PORT)
        self.use_history = bool(use_history)
        self.lowercase_prompt = bool(lowercase_prompt)
        self.action_horizon = int(action_horizon)
        self.poll_interval = float(poll_interval)
        self.strict_horizon = bool(strict_horizon)
        # authors'-protocol S1 (rma_pi05_sgprompt): the oracle subtask text is
        # sent AS the prompt (their reference: prompt_for_vla = current subtask).
        self.prompt_from_subgoal = bool(prompt_from_subgoal)

        # frames + states observed since the last infer; the last entry is
        # always the frame the next infer is conditioned on.
        self._pending_images: list[np.ndarray] = []
        self._pending_states: list[np.ndarray] = []
        self._n_infers = 0
        self._n_episodes = 0

        logger.info("MMERMAAdapter connecting to ws://%s:%s", self.host, self.port)
        # WebsocketClientPolicy._wait_for_server() blocks/retries until the
        # server accepts, so a slow JAX startup on the serving side is fine.
        self._client = _websocket_client_policy.MMEVLAWebsocketClientPolicy(self.host, self.port)
        meta = self._client.get_server_metadata()
        logger.info("MMERMAAdapter connected; server metadata=%s", meta)

    # ------------------------------------------------------------------ #
    # BasePolicyAdapter
    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        """Per-episode reset: drop the server-side memory buffer and ours."""
        self._pending_images.clear()
        self._pending_states.clear()
        resp = self._client.reset()
        # The server answers reset synchronously with {"reset_finished": True};
        # anything else means the policy never dropped its memory buffer and the
        # episode would be contaminated by the previous one.
        if not resp.get("reset_finished", False):
            raise RuntimeError(f"Policy server did not acknowledge reset: {resp!r}")
        self._n_episodes += 1
        self._n_infers = 0

    def observe(self, obs: dict[str, Any], prompt: str, resize_size: int) -> None:
        """Called by the harness at EVERY env step, before the replan check.

        This is what keeps our perceptual (frame-sampling) memory fed with the
        full per-step history rather than only the replan-boundary frames.
        """
        if not self.use_history:
            return
        self._pending_images.append(np.ascontiguousarray(self._agentview(obs)))
        self._pending_states.append(self._state(obs))

    def infer_actions(self, obs: dict[str, Any], prompt: str, resize_size: int) -> np.ndarray:
        image = self._agentview(obs)
        wrist = self._wrist(obs)
        state = self._state(obs)

        if self.use_history:
            # Harnesses that skip `observe` (or a first call before any observe)
            # must still ship the current frame as the buffer tail.
            if not self._pending_images:
                self._pending_images.append(np.ascontiguousarray(image))
                self._pending_states.append(state)
            self._add_buffer()

        element = {
            "observation/image": image,
            "observation/wrist_image": wrist,
            "observation/state": state,
            "prompt": self._prompt(prompt),
        }
        # Grounded-subgoal feed (subgoal-conditioned S1 policies): a provider
        # installed by run_rma_oracle_eval.py returns the current
        # "<subtask> at <row, col>" string; absent provider = protocol run.
        # load_policy_adapter() imports this file as a SEPARATE module instance, so the
        # provider is bound on the adapter instance by run_rma_oracle_eval.py (the
        # module-level hook is kept as a fallback).
        _prov = getattr(self, "subgoal_provider", None) or SUBGOAL_PROVIDER
        subgoal = _prov() if _prov is not None else None
        if subgoal:
            element["grounded_subgoal"] = subgoal
            element["simple_subgoal"] = subgoal.split(" at <")[0]
            if self.prompt_from_subgoal:
                element["prompt"] = self._prompt(subgoal.split(" at <")[0])
            path = os.environ.get("RMA_SUBGOAL_LOG")
            if path:
                with open(path, "a") as fh:
                    fh.write(json.dumps({"ep": self._n_episodes, "infer": self._n_infers,
                                         "prompt": prompt, "subgoal": subgoal}) + "\n")
        resp = self._client.infer(element)
        self._pending_images.clear()
        self._pending_states.clear()
        self._n_infers += 1

        # Gate instrumentation: when the server runs --return-alphas and
        # RMA_ALPHA_LOG is set, append one jsonl record per infer. Inert
        # otherwise (protocol runs are byte-identical).
        alphas = resp.get("alphas")
        if alphas is not None:
            fh = getattr(self, "_alpha_fh", None)
            if fh is None:
                path = os.environ.get("RMA_ALPHA_LOG")
                fh = open(path, "a", buffering=1) if path else False
                self._alpha_fh = fh
            if fh:
                per_layer = alphas.get("per_layer")
                fh.write(json.dumps({
                    "ep": self._n_episodes,
                    "infer": self._n_infers,
                    "prompt": prompt,
                    "slots": alphas.get("slots"),
                    "skip": alphas.get("skip"),
                    "mu_V": alphas.get("mu_V"),
                    "mu_A": alphas.get("mu_A"),
                    "per_layer": np.asarray(per_layer).tolist() if per_layer is not None else None,
                }) + "\n")

        actions = np.asarray(resp["actions"], dtype=np.float32)
        if actions.ndim != 2:
            raise ValueError(f"Server returned actions with shape {actions.shape}; expected [horizon, dim].")
        if self.strict_horizon and actions.shape != (self.action_horizon, ACTION_DIM):
            raise ValueError(
                f"Server returned actions {actions.shape}; expected "
                f"({self.action_horizon}, {ACTION_DIM}). Set strict_horizon=false in "
                f"--adapter-kwargs if this is intentional."
            )
        return actions

    def close(self) -> None:
        ws = getattr(self._client, "_ws", None)
        if ws is not None:
            try:
                ws.close()
            except Exception:  # noqa: BLE001 - best effort teardown
                logger.warning("Failed to close policy websocket cleanly.", exc_info=True)

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _add_buffer(self) -> None:
        """`pack_buffer` from examples/robomme/utils.py, verbatim in shape."""
        images = np.stack(self._pending_images, axis=0).astype(np.uint8)[:, None]  # (T,1,H,W,3)
        states = np.stack(self._pending_states, axis=0).astype(np.float32)         # (T,8)
        payload = {
            "images": images,
            "state": states,
            "add_buffer": True,
            # RoboMemArena has no video-demo prefix -> the exec start is the
            # buffer head. The server only overwrites its own exec_start_idx
            # when this value is > 0.
            "exec_start_idx": 0,
        }
        resp = self._client.add_buffer(payload)
        if not resp.get("add_buffer_finished", False):
            raise RuntimeError(f"Policy server did not acknowledge add_buffer: {resp!r}")

    def _state(self, obs: dict[str, Any]) -> np.ndarray:
        """8-d state. Default is their `build_eval26_policy_input` output verbatim."""
        state = np.asarray(obs["observation/state"], dtype=np.float32)
        if state.shape != (8,):
            raise ValueError(f"observation/state must be 8-d, got {state.shape}.")
        if self.state_convention == "harness":
            return state
        # "robosuite": redo dims 3..5 with the un-canonicalised conversion that
        # produced the RMA training `ee_states`. Position and gripper dims are
        # identical under both conventions.
        raw = obs.get("_raw_obs", obs)
        quat = raw.get("robot0_eef_quat", raw.get("eef_quat"))
        if quat is None:
            raise KeyError("state_convention='robosuite' needs 'robot0_eef_quat' in the raw obs.")
        state = state.copy()
        state[3:6] = _robosuite_quat2axisangle(np.asarray(quat))
        return state

    def _prompt(self, prompt: str) -> str:
        text = str(prompt)
        return text.lower() if self.lowercase_prompt else text

    @staticmethod
    def _agentview(obs: dict[str, Any]) -> np.ndarray:
        # Already flipud'd + resized to resize_size by _process_image_match_eval26.
        img = np.asarray(obs["observation/image"])
        return MMERMAAdapter._as_rgb_u8(img, "observation/image")

    @staticmethod
    def _wrist(obs: dict[str, Any]) -> np.ndarray:
        img = np.asarray(obs["observation/wrist_image"])
        return MMERMAAdapter._as_rgb_u8(img, "observation/wrist_image")

    @staticmethod
    def _as_rgb_u8(img: np.ndarray, name: str) -> np.ndarray:
        if img.dtype != np.uint8:
            raise TypeError(f"{name} must be uint8, got {img.dtype}.")
        if img.ndim != 3 or img.shape[-1] != 3:
            raise ValueError(f"{name} must be HxWx3 RGB, got {img.shape}.")
        return img


def build_adapter(**kwargs: Any) -> BasePolicyAdapter:
    # --adapter-kwargs is JSON, so booleans/ints arrive already typed; tolerate
    # strings anyway for shell convenience.
    def _as_bool(v: Any, default: bool) -> bool:
        if v is None:
            return default
        if isinstance(v, str):
            return v.strip().lower() in ("1", "true", "yes", "y", "on")
        return bool(v)

    return MMERMAAdapter(
        host=kwargs.get("host"),
        port=kwargs.get("port"),
        use_history=_as_bool(kwargs.get("use_history"), True),
        lowercase_prompt=_as_bool(kwargs.get("lowercase_prompt"), True),
        action_horizon=int(kwargs.get("action_horizon", DEFAULT_ACTION_HORIZON)),
        strict_horizon=_as_bool(kwargs.get("strict_horizon"), True),
        state_convention=str(kwargs.get("state_convention", "harness")),
        prompt_from_subgoal=_as_bool(kwargs.get("prompt_from_subgoal"), False),
    )

"""Two-agent (memory writer + planner) subgoal predictor.

Drives the WAM agent service: the demo prefix is sent once at episode start
so agent-1 can build the memory bank, then each 16-step control chunk is sent
as five frames, agent-1 appends a memory line and agent-2 emits the subgoal
that pi0.5 consumes in place of the oracle string.
"""
import base64, io, json, os, urllib.request
from typing import Optional, Tuple

import numpy as np
from PIL import Image

from subgoal_predictor import SubgoalPredictorBase

HOST = os.environ.get("WAM_HOST", "127.0.0.1")
PORT = int(os.environ.get("WAM_PORT", "8899"))
STRIDE, WIN = 4, 16


def _b64(img: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(img.astype(np.uint8)).convert("RGB").save(
        buf, format="JPEG", quality=87)
    return base64.b64encode(buf.getvalue()).decode()


def _post(path: str, payload: dict, timeout: int = 900) -> dict:
    req = urllib.request.Request(
        f"http://{HOST}:{PORT}{path}", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


class WamSubgoalPredictor(SubgoalPredictorBase):
    def setup_api(self) -> None:
        print(f"[wam] using agent service at {HOST}:{PORT}")

    def start_episode(self, epstate, env_runner) -> None:
        super().start_episode(epstate, env_runner)
        self.buf = []
        demo = [_b64(f) for f in epstate.image_buffer[:-1]]
        cur = _b64(epstate.image_buffer[-1]) if epstate.image_buffer else None
        self.task_name = str(getattr(env_runner, "env_id", ""))
        try:
            r = _post("/reset", {"ep": self.episode_id,
                                 "instruction": self.task_goal,
                                 "demo_frames": demo, "cur_frame": cur,
                                 "task": getattr(env_runner, "env_id", "")})
        except Exception as e:
            # 2026-09-09 (VPO incident): a reset failure swallowed by the
            # framework's episode-level except ran 29 ORACLE episodes into
            # the off-arm. SystemExit is not an Exception -- it kills the
            # client outright instead of degrading to the oracle.
            raise SystemExit(
                f"[wam] /reset failed for ep{self.episode_id}: {e} -- "
                "agent server dead, terminating client") from e
        print(f"[wam] ep{self.episode_id}: demo {r.get('n_demo')} frames -> "
              f"{len(r.get('bank', []))} bank lines ({r.get('secs')}s)")

    def step(self, epstate) -> None:
        self.buf.append(epstate.image_buffer[-1])

    def get_subgoal(self, count: int, current_subgoal: Optional[str],
                    last_subgoal: Optional[str]) -> Tuple[Optional[str], bool]:
        frames = self.buf[-(WIN + 1):] if self.buf else []
        if not frames:
            return last_subgoal, False
        idx = list(range(0, len(frames), STRIDE))
        if idx[-1] != len(frames) - 1:
            idx.append(len(frames) - 1)
        sel = [_b64(frames[i]) for i in idx[-5:]]
        try:
            oracle = self.env_runner.grounded_subgoal_oracle
        except Exception:
            oracle = ""
        try:
            r = _post("/tick", {"frames": sel, "step": int(count),
                                "oracle": oracle})
        except Exception as e:
            # tolerate transient failures; a DEAD server must ABORT the run
            # (2026-09-09: a dead agent server silently degraded episodes,
            # and a wrapper-less relaunch scored pure-oracle episodes into
            # the off-arm -- never degrade silently again).
            self._fails = getattr(self, "_fails", 0) + 1
            print(f"[wam] tick failed at step {count} "
                  f"({self._fails} consecutive): {e}")
            if self._fails >= 3:
                print("[wam] agent server unreachable -- ABORTING")
                return None, True
            return last_subgoal, False
        self._fails = 0
        self.buf.clear()
        sub = (r.get("subgoal") or "").strip()
        if sub.lower().startswith("hold"):
            sub = "static"
        return sub, False

    def end_episode(self, epstate, success_flag: str) -> None:
        try:
            _post("/episode_end", {"success": success_flag}, timeout=60)
        except Exception:
            pass

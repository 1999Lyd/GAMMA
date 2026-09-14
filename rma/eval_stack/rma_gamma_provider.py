"""GAMMA subgoal provider for the RoboMemArena eval stack (non-privileged).
Hooks the adapter's per-step observe() to keep a rolling window of processed
agent-view frames + 8-d states, and on every infer (replan tick) posts the
window (frames at t-10, t-5, t + the new states) to rma_agent_server.py, which
returns the plan-pinned subgoal after the propose-verify harness. Installed like
the oracle provider (adapter.subgoal_provider = provider.current)."""
from __future__ import annotations
import base64, io, json, os, urllib.request
import numpy as np
from PIL import Image

def _b64(img):
    buf = io.BytesIO(); Image.fromarray(np.asarray(img).astype(np.uint8)).convert("RGB").save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()

class GammaProvider:
    def __init__(self, host=None, port=None):
        self.host = host or os.environ.get("RMA_AGENT_HOST", "127.0.0.1"); self.port = int(port or os.environ.get("RMA_AGENT_PORT", "8140"))
        self.task_id = None; self.prompt = ""; self.plan = []; self.episode = -1
        self.frames = []; self.states = []; self.n_sent = 0; self.last_sub = None; self.calls = 0
        # RMA_PLAN_LABELS=<json {task_id: [label,...]}>: the text sent to the POLICY is the
        # label of the plan step the agent is on (e.g. the authors' primitive_order labels
        # when serving their published VLA); the agent itself keeps our plan instr strings.
        lp = os.environ.get("RMA_PLAN_LABELS", "")
        self.labels = {int(k): v for k, v in json.load(open(lp)).items()} if lp else None

    def _label(self, j, sub):
        if self.labels is None or self.task_id is None: return sub
        labs = self.labels[int(self.task_id)]
        return labs[min(max(int(j), 0), len(labs) - 1)]

    def _post(self, path, payload, timeout=600):
        req = urllib.request.Request(f"http://{self.host}:{self.port}{path}", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r: return json.loads(r.read().decode())

    def bind(self, adapter):
        orig_obs, orig_reset = adapter.observe, adapter.reset
        def observe(obs, prompt, resize_size):
            orig_obs(obs, prompt, resize_size)
            self.frames.append(np.array(adapter._pending_images[-1])); self.states.append(np.asarray(adapter._pending_states[-1], float).tolist())
        def reset():
            orig_reset(); self.on_reset()
        adapter.observe = observe; adapter.reset = reset; adapter.subgoal_provider = self.current
        return self

    def on_reset(self):
        self.episode += 1; self.frames.clear(); self.states.clear(); self.n_sent = 0; self.last_sub = None; self.calls = 0
        self._post("/reset", {"instruction": self.prompt, "plan": self.plan, "task_id": self.task_id, "ep": self.episode})

    def current(self) -> str:
        if not self.frames: return self.last_sub or (self._label(0, self.plan[0]['instr'] if isinstance(self.plan[0], dict) else self.plan[0]) if self.plan else self.prompt)
        n = len(self.frames); idx = sorted({max(0, n - 11), max(0, n - 6), n - 1})
        while len(idx) < 3: idx = [idx[0]] + idx
        new_states = self.states[self.n_sent:]; self.n_sent = len(self.states)
        r = self._post("/tick", {"frames": [_b64(self.frames[i]) for i in idx], "offsets": [0, 5, 10], "frame_states": [self.states[i] for i in idx],
                                 "states": new_states, "step": n})
        self.calls += 1; self.last_sub = self._label(r["j"], r["subgoal"])
        path = os.environ.get("RMA_GAMMA_LOG")
        if path:
            with open(path, "a") as fh: fh.write(json.dumps({"ep": self.episode, "call": self.calls, "step": n, "subgoal": r["subgoal"], "j": r["j"], "verdicts": r["verdicts"], "secs": r["secs"]}) + "\n")
        return self.last_sub

    def end(self, success):
        try: self._post("/end", {"success": success})
        except Exception: pass

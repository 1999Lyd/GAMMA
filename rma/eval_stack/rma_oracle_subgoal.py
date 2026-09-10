"""Privileged grounded-subgoal oracle for the RoboMemArena eval stack.

Feeds the subgoal-conditioned S1 policy (config rma_ground_sg) the same
"<subtask> at <row, col>" stream it was trained on, from simulator state --
the RMA analogue of RoboMME's GroundSG+Oracle ceiling row.

* Plan per task: rma_oracle_plans.json (ordered subtasks derived from the
  training segments; task4/5 use the drawer variant the benchmark scores).
* Transitions: "pick up X" completes when X has risen 3 cm above its
  initial height; every other subtask consumes the benchmark's own scoring
  stage predicate for that step, in order (eval_tasks2_26._task_specs), so
  the oracle advances exactly when the benchmark registers the stage; the
  trailing "close the microwave door" (no scoring stage) uses the reference
  _microwave_closed predicate.
* Coordinates: pick subtasks project the object's live body position
  through the fixed agentview matrix (rma_agentview_camera.json, the same
  projection used to build the training targets); all other subtasks use
  the training-set median coordinate of that (task, subtask) -- receptacles
  and fixtures do not move across layouts (IQR <= a few px).

Hook: install() wraps eval_common._get_env_class so every env the benchmark
creates reports reset/step to the oracle; the adapter reads current() at
each replan. Scoring code is untouched.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PLANS = {int(k): v for k, v in json.load(open(os.path.join(HERE, "rma_oracle_plans.json"))).items()}
_CAM = json.load(open(os.path.expandvars("${GAMMA_ROOT}/rma/rma_agentview_camera.json")))
M = np.asarray(_CAM["matrix"], dtype=np.float64)
HW = tuple(_CAM["hw"])
LIFT_DELTA = 0.03
FIXED_LAYOUT_IQR = 3.0   # px; training IQR at or below this = object never moves across seeds
# live-projection minus training-median (row, col) on fixed-layout tasks, seed 50
GRASP_OFFSET = {"butter_1": (1, 0), "chocolate_pudding_1": (3, -1), "cookies_1": (0, -2),
                "cream_cheese_1": (-1, -2), "milk_1": (-2, -2), "popcorn_1": (6, 0),
                "tomato_sauce_1": (3, 1), "wine_bottle_1": (6, 0), "orange_juice_1": (0, 0)}
_OBJ = [("chocolate pudding", "chocolate_pudding_1"), ("cream cheese", "cream_cheese_1"),
        ("tomato sauce", "tomato_sauce_1"), ("tomato_sauce", "tomato_sauce_1"),
        ("wine bottle", "wine_bottle_1"), ("orange_juice", "orange_juice_1"), ("cookies", "cookies_1"),
        ("butter", "butter_1"), ("popcorn", "popcorn_1"), ("pudding", "chocolate_pudding_1"),
        ("chocolate", "chocolate_pudding_1"), ("cream", "cream_cheese_1"), ("milk", "milk_1")]


def _obj_of(ins: str) -> str:
    for k, v in _OBJ:
        if k in ins:
            return v
    raise KeyError(ins)


def project_rc(p) -> tuple[int, int]:
    """world xyz -> (row, col) on the training frame grid (patch_grounded_subgoals.py)."""
    world = np.array([p[0], p[1], p[2], 1.0])
    cam = M @ world
    pix = cam[:2] / cam[2]
    return (int(np.clip(round(pix[1]), 0, HW[0] - 1)), int(np.clip(round(pix[0]), 0, HW[1] - 1)))


class SubgoalOracle:
    def __init__(self, num_steps_wait: int = 10):
        self.num_steps_wait = num_steps_wait
        self.task_id: int | None = None
        self._tasks26 = None
        self.reset_counters()

    # ---- benchmark module (imported lazily: needs the eval venv path) ----
    @property
    def t26(self):
        if self._tasks26 is None:
            import eval_tasks2_26 as tasks26  # noqa
            self._tasks26 = tasks26
        return self._tasks26

    def reset_counters(self):
        self.steps = 0
        self.state = None
        self.idx = 0
        self.seg_start = 0
        self.checks = []
        self.plan = []
        self.transitions = []
        self.episode = -1

    # ---- env hooks ----
    def on_reset(self, env, obs):
        assert self.task_id is not None, "oracle.task_id must be set before env.reset()"
        self.reset_counters()
        self.episode += 1
        self.plan = [dict(s) for s in PLANS[self.task_id]]
        self.checks = [self._check_for(step, specs) for specs in [list(self.t26._task_specs(self.task_id))]
                       for step in self.plan]
        # every scoring stage must have been consumed by exactly one plan step
        left = [s.name for s in self._specs_left]
        assert not left, f"task{self.task_id}: unconsumed scoring stages {left}"
        for step in self.plan:
            if step["kind"] == "pick" and self.t26._current_body_pos(env, step["obj"]) is None:
                raise RuntimeError(f"task{self.task_id}: body {step['obj']} not found in env")
        self._env = env

    _specs_left: list = []

    def _check_for(self, step, specs):
        """Bind one plan step to its completion predicate: the benchmark's own
        scoring stage when the next unconsumed stage is of the same type
        (Lift <-> pick of that object, Pour, Open/Close, Place/Put), else a
        reference predicate for the unscored steps (place after pours, the
        task-22 microwave steps, the trailing 'close the microwave door')."""
        t = self.t26
        self._specs_left = specs
        ins, kind, obj = step["instr"], step["kind"], step["obj"]
        nxt = specs[0].name[3:] if specs else ""      # strip the "NN_" prefix
        if kind == "pick":
            if nxt.startswith("Lift"):
                step["stage"] = specs.pop(0).name
            return self._lift_check(obj)
        if kind == "microwave_closed":
            return t._microwave_closed()
        if ins.startswith("pour"):
            assert nxt.startswith("Pour"), f"{ins!r} vs stage {nxt!r}"
            step["stage"] = specs[0].name
            return specs.pop(0).check_fn
        head = ins.split()[0]
        if (head == "open" and nxt.startswith("Open")) or (head == "close" and nxt.startswith("Close")) \
                or (head == "place" and (nxt.startswith("Place") or nxt.startswith("Put"))):
            step["stage"] = specs[0].name
            return specs.pop(0).check_fn
        # unscored steps: reference predicates with the benchmark's thresholds
        if ins.startswith("open the microwave"):
            return t._microwave_open(0.30)
        o = _obj_of(ins)
        if "bowl drainer" in ins:
            return t._in_container_body(o, "bowl_drainer_1", 0.15, -0.05, 0.20)
        if "frypan" in ins:
            return t._in_container_body(o, "frypan_1", 0.12, -0.05, 0.15)
        if "on table" in ins:
            return t._table_return(o, 0.35 if o == "wine_bottle_1" else 0.40)
        if "aside" in ins:
            return t._near_fixed_position(o, np.array([0.0, -0.2, 0.50], dtype=np.float32), 0.20, 0.20)
        if "in the microwave" in ins:
            return t._in_microwave(o)
        raise ValueError(f"task{self.task_id}: no predicate for {ins!r} (next stage {nxt!r})")

    def on_step(self, env, obs):
        self.steps += 1
        if self.steps < self.num_steps_wait:
            return
        if self.state is None:
            # mirrors eval_tasks2_26: state is built once the dummy-action prefix is over
            self.state = self.t26._build_initial_state(env)
            self.seg_start = self.state["step_idx"]
            return
        self.t26._update_state(obs, self.state)
        if self.idx < len(self.plan) and self.checks[self.idx](env, self.state, self.seg_start):
            done = self.plan[self.idx]["instr"]
            self.idx += 1
            self.seg_start = self.state["step_idx"]
            nxt = self.plan[self.idx]["instr"] if self.idx < len(self.plan) else "(plan complete)"
            self.transitions.append((self.steps, done))
            logging.info(f"  [oracle t={self.steps}] completed '{done}' -> '{nxt}'")

    # ---- predicates ----
    def _lift_check(self, obj):
        def check(env, state, stage_start):
            pos = self.t26._current_body_pos(env, obj)
            init = state["initial_body_pos"].get(obj) if state else None
            if pos is None or init is None:
                init = self.t26._initial_body_pos(state, obj) if state else None
                if pos is None or init is None:
                    return False
            return float(pos[2] - init[2]) > LIFT_DELTA
        return check

    # ---- the subgoal string ----
    def current(self) -> str:
        if not self.plan:
            return ""
        step = self.plan[min(self.idx, len(self.plan) - 1)]
        r, c = step["median"]
        if step["kind"] == "pick" and max(step["iqr"]) > FIXED_LAYOUT_IQR:
            # layout varies across seeds: project the live body position, plus the
            # object's grasp offset (training anchor = gripper at grasp, not body centre;
            # measured on the fixed-layout tasks, smoke_oracle_env.py, 2026-09-10)
            pos = self.t26._current_body_pos(self._env, step["obj"])
            if pos is not None:
                pr, pc = project_rc(pos)
                dr, dc = GRASP_OFFSET.get(step["obj"], (0, 0))
                r, c = int(np.clip(pr - dr, 0, HW[0] - 1)), int(np.clip(pc - dc, 0, HW[1] - 1))
        return f"{step['instr']} at <{r}, {c}>"


ORACLE = SubgoalOracle()


def install(oracle: SubgoalOracle = ORACLE):
    """Wrap the benchmark's env class so reset/step report to the oracle.
    Call BEFORE eval_tasks2_26._patch_env_resolution()."""
    import eval_common as ec  # noqa

    base = ec._get_env_class()

    class OracleEnv(base):
        def reset(self, *a, **k):
            obs = super().reset(*a, **k)
            oracle.on_reset(self, obs)
            return obs

        def step(self, action):
            out = super().step(action)
            oracle.on_step(self, out[0])
            return out

    OracleEnv.__name__ = base.__name__
    ec._get_env_class = lambda: OracleEnv
    return OracleEnv

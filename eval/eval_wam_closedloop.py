"""Run RoboMME closed-loop with the two-agent subgoal system in place of the
oracle.  Wraps their eval.py and swaps only the subgoal predictor."""
import sys, tyro
import eval as ev
import subgoal_predictor as sp
from wam_subgoal_predictor import WamSubgoalPredictor

_orig = sp.build_subgoal_predictor


def build(args, save_dir):
    if getattr(args, "use_oracle", False):
        print("[wam] injecting WamSubgoalPredictor in place of the oracle")
        return WamSubgoalPredictor(args, save_dir)
    return _orig(args, save_dir)


sp.build_subgoal_predictor = build
ev.build_subgoal_predictor = build
tyro.cli(ev.evaluate)

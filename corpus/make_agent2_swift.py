#!/usr/bin/env python
"""Convert agent-2 records (bank -> oracle subgoal) into ms-swift SFT format.

The assistant target is the ORACLE STRING VERBATIM (grounded_subgoal_online),
because that is exactly what pi0.5 was trained to consume: agent-2's job is
to emit the same text the oracle handed the policy at that tick.

Env: SRC (agent2_*.jsonl dir), TAG (output suffix), BANKSRC=gt|pred
"""
import json, os, sys

SRC = os.environ.get("SRC", os.path.expandvars("${GAMMA_DATA}/data/wam_sft_v9"))
BANK = os.environ.get("BANKSRC", "gt")
TAG = os.environ.get("TAG", BANK)
# Bank-only agent-2 is the default (mirrors wam_agent_server.py serve default);
# WAM_A2_IMG=1 restores the legacy frame-in-context records (v17-era).
A2_IMG = os.environ.get("WAM_A2_IMG", "0") == "1"

SYS_IMG = ("You are the robot's planner (agent-2). You read the task, the "
       "robot's memory bank of grounded events, and the current camera frame "
       "with its detections, and you output the robot's current subgoal for "
       "this control chunk. Use the current frame to judge whether the "
       "ongoing subgoal is complete. Answer with the subgoal sentence only, "
       "lowercase, no trailing period, copying object coordinates from the "
       "memory bank or the current detections in the form <x, y>. During the "
       "demo phase answer exactly: hold and wait for the demo to complete.")
# verbatim SYS2_NOIMG from wam_agent_server.py (train/serve contract)
SYS_NOIMG = ("You are the robot's planner (agent-2). You read the task and the "
        "robot's memory bank of grounded events, and you output the robot's "
        "current subgoal for this control chunk. Judge progress from the "
        "bank's completed lines. Answer with the subgoal sentence only, "
        "lowercase, no trailing period, copying object coordinates from the "
        "memory bank in the form <x, y>. During the "
        "demo phase answer exactly: hold and wait for the demo to complete.")
SYS = SYS_IMG if A2_IMG else SYS_NOIMG


def prompt_of(r):
    bank = "\n".join("  " + l for l in r["bank"]) if r["bank"] else "  (empty)"
    phase_q = (f"It is now the "
               f"{'DEMO phase' if r['phase'] == 'demo' else 'EXECUTION phase'}.\n"
               "What is the robot's current subgoal?")
    if A2_IMG:  # legacy frame-in-context records
        return (f"Task: {r['instruction']}\n"
                f"Memory bank:\n{bank}\n"
                f"Current frame and detections:\n<image> {r['frame_det']}\n"
                + phase_q)
    return f"Task: {r['instruction']}\nMemory bank:\n{bank}\n" + phase_q


for split in ("train", "val"):
    fn = (f"{SRC}/agent2_{split}.jsonl" if BANK == "gt"
          else f"{SRC}/agent2_pred_{split}.jsonl")
    if not os.path.exists(fn):
        print(f"skip {fn} (missing)")
        continue
    out = f"{SRC}/agent2_swift_{TAG}_{split}.jsonl"
    n = 0
    with open(out, "w") as w:
        for l in open(fn):
            r = json.loads(l)
            rec = {"messages": [
                {"role": "system", "content": SYS},
                {"role": "user", "content": prompt_of(r)},
                {"role": "assistant", "content": r["target"]}]}
            if A2_IMG:
                rec["images"] = [os.path.join(SRC, q) for q in r["frames"]]
            w.write(json.dumps(rec) + "\n")
            n += 1
    print(f"{out}: {n}")

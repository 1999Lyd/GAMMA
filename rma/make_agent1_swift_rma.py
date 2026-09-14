#!/usr/bin/env python
"""RMA writer records -> ms-swift SFT format. The prompt is byte-identical to the
RMA agent server's (rma_agent_server.py): task, phase, the planner's expected
next subgoal, the last 8 bank lines, then one "<image> +{off}: {dets} {state}"
line per window frame (3 frames over the 10-step tick).  Env: SRC (default rma_sft_v1)."""
import json, os
SRC = os.environ.get("SRC", os.path.expandvars("${GAMMA_DATA}/data/rma_sft_v1"))
SYS = ("You are the robot's memory writer (agent-1). Each tick you watch the "
       "last 10 control steps (3 sampled frames with per-frame object detections "
       "and the robot's own state readout: end-effector position ee<row, col>, "
       "gripper open/closed with its opening, height z and wrist tilt), read the "
       "memory bank and the planner's expected next subgoal, and write ONE line "
       "faithfully describing what happened in this window, in the format "
       "'[event] ...  [sam] evidence'. Write one line per event; if two things "
       "happened, write two lines. If nothing new happened, write NONE.")
def user_of(r):
    bk = "\n".join(r["bank"][-8:]) if r["bank"] else "(empty)"
    per = "\n".join(f"<image> {d}" for d in r["frame_dets"])
    return (f"Task: {r['instruction']}\nPhase: {r['phase']}\n"
            f"Expected next subgoal from the planner: {r['expected_next']}\n"
            f"Memory bank so far:\n{bk}\nCurrent window frames and detections:\n{per}\n"
            "Write the memory line for this window (or NONE).")
if __name__ == "__main__":
    for split in ("train", "val"):
        src, out = f"{SRC}/agent1_{split}.jsonl", f"{SRC}/agent1_swift_{split}.jsonl"
        n = none = 0
        with open(out, "w") as w:
            for l in open(src):
                r = json.loads(l)
                w.write(json.dumps({"messages": [{"role": "system", "content": SYS}, {"role": "user", "content": user_of(r)},
                                                 {"role": "assistant", "content": r["target"]}],
                                    "images": [os.path.join(SRC, p) for p in r["frames"]]}) + "\n")
                n += 1; none += r["target"].strip() == "NONE"
        print(f"{out}: {n} records, NONE {none} ({none/max(n,1):.3f})")

#!/usr/bin/env python
"""Convert agent-1 records into ms-swift SFT format (multimodal).

The prompt is byte-identical to what rollout_agent1.py and wam_agent_server.py
build at inference time -- task, phase, the planner's expected next subgoal,
the last 8 bank lines, then one "<image> t{t}: {dets}" line per window frame --
so the writer never sees a different string at serve time than it trained on.

Env: SRC (dataset dir, default wam_sft_v12).
"""
import json, os

SRC = os.environ.get("SRC",
                     os.path.expandvars("${GAMMA_DATA}/data/wam_sft_v12"))

SYS = ("You are the robot's memory writer (agent-1). Each tick you watch the "
       "last 16 control steps (sampled frames with per-frame object detections), "
       "read the memory bank and the planner's expected next subgoal, and "
       "write ONE line faithfully describing what happened in this window "
       "with grounded object coordinates, in the format '[event] ...  [sam] "
       "name<x, y>'. Write one line per event; if two things happened, write "
       "two lines. If nothing new happened, write NONE.")


def user_of(r):
    bk = "\n".join(r["bank"][-8:]) if r["bank"] else "(empty)"
    per = "\n".join(f"<image> {d}" for d in r["frame_dets"])
    return (f"Task: {r['instruction']}\n"
            f"Phase: {r['phase']}\n"
            f"Expected next subgoal from the planner: {r['expected_next']}\n"
            f"Memory bank so far:\n{bk}\n"
            f"Current window frames and detections:\n{per}\n"
            "Write the memory line for this window (or NONE).")


for split in ("train", "val"):
    src, out = f"{SRC}/agent1_{split}.jsonl", f"{SRC}/agent1_swift_{split}.jsonl"
    n = none = 0
    with open(out, "w") as w:
        for l in open(src):
            r = json.loads(l)
            w.write(json.dumps({
                "messages": [{"role": "system", "content": SYS},
                             {"role": "user", "content": user_of(r)},
                             {"role": "assistant", "content": r["target"]}],
                "images": [os.path.join(SRC, p) for p in r["frames"]],
            }) + "\n")
            n += 1
            none += r["target"].strip() == "NONE"
    print(f"{out}: {n} records, NONE {none} ({none/max(n,1):.3f})")

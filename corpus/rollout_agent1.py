#!/usr/bin/env python
"""Phase-B: autoregressive agent-1 rollout -> predicted banks for agent-2.

At every tick agent-1 reads its OWN accumulated bank (not the GT bank) plus
the window frames/detections and the planner's expected-next, and writes the
next memory line. Episodes are batched against each other (all active
episodes advance one tick per generate call), so throughput is bounded by
batch size, not by the sequential dependency inside an episode.

Outputs $OUT/agent2_pred_{split}.jsonl: same schema as agent2_{split}.jsonl
but `bank` is agent-1's predicted bank; target is unchanged (the online
oracle subgoal). Also writes agent1_rollout_{split}.jsonl (per-tick GT vs
predicted line) for writer diagnostics.

Env: CKPT, SPLIT=val|train, SHARD/NSHARD (episode sharding), BATCH.
"""
import collections, json, os, re, sys, time
import numpy as np

CKPT = os.environ["CKPT"]
SPLIT = os.environ.get("SPLIT", "val")
SHARD = int(os.environ.get("SHARD", "0"))
NSHARD = int(os.environ.get("NSHARD", "1"))
BATCH = int(os.environ.get("BATCH", "12"))
OUT = os.environ.get("DATA_DIR", os.path.expandvars("${GAMMA_DATA}/data/wam_sft_v9"))
FRAMES = os.environ.get("FRAMES_DIR", os.path.expandvars("${GAMMA_DATA}/data/wam_sft_v8"))
t0 = time.time()

os.environ.setdefault("HF_HOME", os.path.expandvars("${HF_HOME}"))
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel

MID = os.environ.get("WAM_BASE", "Qwen/Qwen3-VL-4B-Instruct")
proc = AutoProcessor.from_pretrained(MID)
proc.tokenizer.padding_side = "left"
model = AutoModelForImageTextToText.from_pretrained(
    MID, dtype=torch.bfloat16, device_map="cuda:0")
model = PeftModel.from_pretrained(model, CKPT)
model.eval()
print(f"[rollout] loaded ({(time.time()-t0)/60:.1f} min)", flush=True)

SYS = ("You are the robot's memory writer (agent-1). Each tick you watch the "
       "last 16 control steps (sampled frames with per-frame object detections), "
       "read the memory bank and the planner's expected next subgoal, and "
       "write ONE line faithfully describing what happened in this window "
       "with grounded object coordinates, in the format '[event] ...  [sam] "
       "name<x, y>'. If nothing new happened, write NONE.")

# ---- load records (canonical offset-0 grid only)
a1 = collections.defaultdict(list)
for l in open(f"{OUT}/agent1_{SPLIT}.jsonl"):
    r = json.loads(l)
    if r.get("offset", 0) != 0:
        continue
    if r["ep"] % NSHARD != SHARD:
        continue
    a1[r["ep"]].append(r)
for e in a1:
    a1[e].sort(key=lambda r: (r["span"][1], r["span"][0]))
a2gt = {}
for l in open(f"{OUT}/agent2_{SPLIT}.jsonl"):
    r = json.loads(l)
    if r.get("offset", 0) != 0 or r["ep"] % NSHARD != SHARD:
        continue
    a2gt[(r["ep"], r["tq"])] = r
eps = sorted(a1)
print(f"[rollout] {len(eps)} episodes, "
      f"{sum(len(v) for v in a1.values())} ticks (shard {SHARD}/{NSHARD})",
      flush=True)


def build(rec, bank):
    per = [f"<image> {d}" for d in rec["frame_dets"]]
    bk = "\n".join(bank[-8:]) if bank else "(empty)"
    return (f"Task: {rec['instruction']}\n"
            f"Phase: {rec['phase']}\n"
            f"Expected next subgoal from the planner: {rec['expected_next']}\n"
            f"Memory bank so far:\n{bk}\n"
            f"Current window frames and detections:\n" + "\n".join(per) +
            "\nWrite the memory line for this window (or NONE).")


def to_msgs(rec, bank):
    txt = build(rec, bank)
    parts = txt.split("<image>")
    imgs = [Image.open(os.path.join(FRAMES, p)).convert("RGB")
            for p in rec["frames"]]
    content = [{"type": "text", "text": parts[0]}]
    for i, p in enumerate(parts[1:]):
        if i < len(imgs):
            content.append({"type": "image", "image": imgs[i]})
        content.append({"type": "text", "text": p})
    return [{"role": "system", "content": [{"type": "text", "text": SYS}]},
            {"role": "user", "content": content}]


@torch.no_grad()
def gen_batch(batch_msgs):
    inp = proc.apply_chat_template(
        batch_msgs, add_generation_prompt=True, enable_thinking=False, tokenize=True,
        return_dict=True, return_tensors="pt", padding=True).to("cuda:0")
    out = model.generate(**inp, max_new_tokens=280, do_sample=False,
                         pad_token_id=proc.tokenizer.pad_token_id)
    n = inp["input_ids"].shape[1]
    return [proc.decode(o[n:], skip_special_tokens=True).strip() for o in out]


banks = {e: [] for e in eps}
cursor = {e: 0 for e in eps}
a2_out = open(f"{OUT}/agent2_pred_{SPLIT}{'' if NSHARD == 1 else f'.s{SHARD}'}.jsonl", "w")
a1_out = open(f"{OUT}/agent1_rollout_{SPLIT}{'' if NSHARD == 1 else f'.s{SHARD}'}.jsonl", "w")
done = 0
ntick = sum(len(v) for v in a1.values())
active = [e for e in eps if cursor[e] < len(a1[e])]
while active:
    for i in range(0, len(active), BATCH):
        chunk = active[i:i + BATCH]
        recs = [a1[e][cursor[e]] for e in chunk]
        msgs = [to_msgs(r, banks[e]) for r, e in zip(recs, chunk)]
        try:
            outs = gen_batch(msgs)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            outs = []
            for m in msgs:
                outs.extend(gen_batch([m]))
        for e, r, o in zip(chunk, recs, outs):
            tq = r["span"][1]
            a1_out.write(json.dumps(dict(
                ep=e, tq=tq, family=r["family"], gt=r["target"], pred=o)) + "\n")
            if o.strip().upper() != "NONE" and o.strip():
                for ln in o.split("\n"):
                    ln = ln.strip()
                    if ln and ln.upper() != "NONE":
                        banks[e].append(ln)
            q = a2gt.get((e, tq))
            if q is not None:
                a2_out.write(json.dumps(dict(
                    instruction=q["instruction"], family=q["family"], ep=e,
                    tq=tq, phase=q["phase"], bank=list(banks[e]),
                    target=q["target"], offset=0,
                    frames=q.get("frames", []),
                    frame_det=q.get("frame_det", ""))) + "\n")
            cursor[e] += 1
            done += 1
        if done % (BATCH * 20) < BATCH:
            el = (time.time() - t0) / 60
            print(f"[rollout] {done}/{ntick} ticks ({el:.1f} min, "
                  f"eta {el/max(done,1)*(ntick-done):.0f} min)", flush=True)
    active = [e for e in eps if cursor[e] < len(a1[e])]
a2_out.close()
a1_out.close()
print(f"[rollout] DONE {done} ticks; {(time.time()-t0)/60:.1f} min", flush=True)

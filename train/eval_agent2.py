#!/usr/bin/env python
"""Open-loop agent-2 evaluation: does it emit the subgoal text the oracle
handed pi0.5?  Scored against BOTH the snapped target it was trained on and
the untouched oracle string (`target_oracle`).
"""
import collections, json, os, re, sys, time
import numpy as np

CKPT = os.environ["CKPT"]
VALFILE = os.environ["VALFILE"]
OUTJSON = os.environ.get("OUTJSON", os.path.expandvars("${GAMMA_DATA}/traces/agent2_openloop.json"))
NPF = int(os.environ.get("NPF", "0"))     # 0 = all
BATCH = int(os.environ.get("BATCH", "16"))
COORD = re.compile(r"<\s*(\d+)\s*,\s*(\d+)\s*>")
t0 = time.time()

os.environ.setdefault("HF_HOME", os.path.expandvars("${HF_HOME}"))
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel
MID = os.environ.get("WAM_BASE", "Qwen/Qwen3-VL-4B-Instruct")
DATA_DIR = os.environ.get("DATA_DIR", os.path.dirname(VALFILE))
proc = AutoProcessor.from_pretrained(MID)
proc.tokenizer.padding_side = "left"
model = AutoModelForImageTextToText.from_pretrained(MID, dtype=torch.bfloat16, device_map="cuda:0")
model = PeftModel.from_pretrained(model, CKPT)
model.eval()
print(f"[a2eval] loaded ({(time.time()-t0)/60:.1f} min)", flush=True)

SYS = ("You are the robot's planner (agent-2). You read the task, the "
       "robot's memory bank of grounded events, and the current camera frame "
       "with its detections, and you output the robot's current subgoal for "
       "this control chunk. Use the current frame to judge whether the "
       "ongoing subgoal is complete. Answer with the subgoal sentence only, "
       "lowercase, no trailing period, copying object coordinates from the "
       "memory bank or the current detections in the form <x, y>. During the "
       "demo phase answer exactly: hold and wait for the demo to complete.")

from PIL import Image

def prompt_of(r):
    bank = "\n".join("  " + l for l in r["bank"]) if r["bank"] else "  (empty)"
    return (f"Task: {r['instruction']}\nMemory bank:\n{bank}\n"
            f"Current frame and detections:\n<image> {r.get('frame_det','')}\n"
            f"It is now the "
            f"{'DEMO phase' if r['phase']=='demo' else 'EXECUTION phase'}.\n"
            "What is the robot's current subgoal?")

def msgs_of(r):
    txt = prompt_of(r)
    pre, post = txt.split("<image>", 1)
    img = Image.open(os.path.join(DATA_DIR, r["frames"][0])).convert("RGB")
    return [{"role": "system", "content": [{"type": "text", "text": SYS}]},
            {"role": "user", "content": [
                {"type": "text", "text": pre},
                {"type": "image", "image": img},
                {"type": "text", "text": post}]}]

@torch.no_grad()
def gen(recs):
    msgs = [msgs_of(r) for r in recs]
    inp = proc.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, tokenize=True,
                                   return_dict=True, return_tensors="pt", padding=True).to("cuda:0")
    out = model.generate(**inp, max_new_tokens=48, do_sample=False,
                         pad_token_id=proc.tokenizer.pad_token_id)
    n = inp["input_ids"].shape[1]
    return [proc.decode(o[n:], skip_special_tokens=True).strip() for o in out]

rows = [json.loads(l) for l in open(VALFILE)]
if NPF:
    c = collections.Counter(); sel = []
    for r in rows:
        if c[r["family"]] < NPF:
            sel.append(r); c[r["family"]] += 1
    rows = sel
print(f"[a2eval] {len(rows)} val ticks", flush=True)

def norm(s): return re.sub(r"\s+", " ", s.lower().strip().rstrip("."))
def nocoord(s): return re.sub(r"\s*at\s*<[^>]*>", "", norm(s)).strip()

res = []
for i in range(0, len(rows), BATCH):
    ch = rows[i:i+BATCH]
    outs = gen(ch)
    for r, o in zip(ch, outs):
        d = dict(fam=r["family"], ep=r["ep"], tq=r["tq"], out=o,
                 tgt=r["target"], oracle=r.get("target_oracle", r["target"]))
        d["exact_trained"] = norm(o) == norm(d["tgt"])
        d["exact_oracle"] = norm(o) == norm(d["oracle"])
        d["actobj"] = nocoord(o) == nocoord(d["oracle"])
        g = COORD.findall(d["oracle"]); p = COORD.findall(o)
        if g and p:
            gx, gy = int(g[0][0]), int(g[0][1])
            d["derr"] = float(min(np.hypot(int(a)-gx, int(b)-gy) for a, b in p))
        elif g and not p:
            d["derr"] = 1e9
        res.append(d)
    if (i // BATCH) % 20 == 0:
        print(f"[a2eval] {i+len(ch)}/{len(rows)} ({(time.time()-t0)/60:.1f} min)", flush=True)

def rate(rs, k): return float(np.mean([x[k] for x in rs])) if rs else float("nan")
withc = [x for x in res if "derr" in x]
print(f"\nOPEN-LOOP AGENT-2 vs ORACLE  (n={len(res)})")
print(f"  exact match to oracle string : {rate(res,'exact_oracle'):.3f}")
print(f"  exact match to trained target: {rate(res,'exact_trained'):.3f}")
print(f"  action+object (coords ignored): {rate(res,'actobj'):.3f}")
if withc:
    d = np.array([x["derr"] for x in withc])
    print(f"  coord <=8px  : {np.mean(d<=8):.3f}   <=30px: {np.mean(d<=30):.3f}  (n={len(withc)})")
    ok = np.array([x["actobj"] for x in withc])
    print(f"  action+object AND coord<=8px : {np.mean(ok & (d<=8)):.3f}")
print(f"\n{'task':10s} {'n':>5s} {'exact':>7s} {'act+obj':>8s} {'<=8px':>7s}")
for f in sorted({x['fam'] for x in res}):
    rs = [x for x in res if x['fam'] == f]
    wc = [x for x in rs if 'derr' in x]
    dd = np.array([x['derr'] for x in wc]) if wc else np.array([])
    print(f"{f:10s} {len(rs):5d} {rate(rs,'exact_oracle'):7.3f} {rate(rs,'actobj'):8.3f} "
          f"{(np.mean(dd<=8) if len(dd) else float('nan')):7.3f}")
json.dump(res, open(OUTJSON, "w"), indent=1)
print(f"[a2eval] wrote {OUTJSON}; {(time.time()-t0)/60:.1f} min")

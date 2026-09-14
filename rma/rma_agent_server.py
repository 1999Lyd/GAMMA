#!/usr/bin/env python
"""GAMMA agent server for RoboMemArena (non-privileged).
Per tick (10 control steps) the client posts 3 window frames + the new
proprioceptive states; the server runs SAM-3 (r6 concepts, task vocabulary),
the writer (Qwen3.5-9B LoRA, prompt byte-identical to make_agent1_swift_rma),
and the propose-verify harness:
  progress claims ('[event] completed: <plan step>') are ADMITTED only when the
  state predicate for the CURRENT plan step (rma_progress_predicates v5, scanned
  from the last verified completion) has fired; otherwise DEFERRED (re-checked
  every tick); a claim for any other step is REJECTED (the plan is pinned);
  scene/observation lines are admitted (consecutive duplicates rejected).
  Limiting case: if the predicate fired >= AUTO_GRACE ticks ago and no claim was
  admitted, the harness emits the completion itself ('(auto)').
The reasoner is plan-pinned: subgoal = plan[j] (Algorithm 1, recorded sequence).
Env: WAM_BASE, A1_CKPT (LoRA dir; WAM_A1_OFF=1 -> no writer, predicates only),
     PORT (8140), WAM_TRACE, WAM_HARNESS_OFF=1 (admit writer claims unverified,
     no auto-claims), AUTO_GRACE (2), PROMPTS (r6 json), DET_THR (0.15)
"""
import os, sys, json, time, base64, io, re
import numpy as np, torch
from PIL import Image
from http.server import BaseHTTPRequestHandler, HTTPServer
from scipy.spatial.transform import Rotation as Rot
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import rma_progress_predicates as pp
from make_agent1_swift_rma import SYS, user_of
os.environ.setdefault("HF_HOME", os.path.expandvars("${HF_HOME}"))
DEV = "cuda:0"; PORT = int(os.environ.get("PORT", "8140")); TRACE = os.environ.get("WAM_TRACE")
HARNESS_OFF = os.environ.get("WAM_HARNESS_OFF", "0") == "1"; A1_OFF = os.environ.get("WAM_A1_OFF", "0") == "1"
AUTO_GRACE = int(os.environ.get("AUTO_GRACE", "2"));
# 2026-09-14 WAM_SETTLE_TICKS=<n> (opt-in): a VERIFIED writer completion claim is
# admitted to the bank at once, but the executor's subgoal pointer j advances only
# once the predicate has held for n ticks -- the same settle rule the auto-claim
# path already applies (AUTO_GRACE). Without it the writer hands off 60-160 steps
# earlier than harness-only mode (claim at the release instant, arm mid-retract).
SETTLE_TICKS = int(os.environ.get("WAM_SETTLE_TICKS", "0") or 0); SAM_OFF = os.environ.get("WAM_SAM_OFF", "0") == "1"; VERIFY_VISION = os.environ.get("WAM_VERIFY_VISION", "1") == "1"; W_TOL = 0.007
WIDTHS = json.load(open(f"{HERE}/grasp_widths.json")) if os.path.exists(f"{HERE}/grasp_widths.json") else {}
RELIABLE = {"tomato_sauce", "wine", "orange_juice"}   # tune_prompts_rma: mean best score >= 0.4, low false positives
OBJ_CONCEPT = {"cookies": "cookies", "tomato sauce": "tomato_sauce", "sauce": "tomato_sauce", "butter": "butter", "popcorn": "popcorn", "cream": "cream", "chocolate": "chocolate", "pudding": "chocolate", "milk": "milk", "wine": "wine", "orange": "orange_juice"}
def obj_of(ins):
    i = ins.lower()
    for o in ("cookies", "tomato sauce", "butter", "popcorn", "cream", "chocolate", "pudding", "milk", "wine", "orange", "sauce"):
        if o in i: return o
    return None
def band_for(ins):
    o = obj_of(ins); k = pp.kind_of(ins)
    if o is None or k not in ("pick", "place"): return None
    ent = WIDTHS.get(("carry:" + o) if k == "place" else o) or WIDTHS.get(o)
    if not ent: return None
    return (max(0.006, ent["p10"] - 0.004), min(0.0395, ent["p90"] + 0.004))
DET_THR = float(os.environ.get("DET_THR", "0.15")); TICK = 10
cfg = json.load(open(os.environ.get("PROMPTS", f"{HERE}/prompts_rma_r6.json")))
CONCEPTS = [tuple(c) for c in cfg["concepts"]]; ALWAYS = [tuple(a) for a in cfg["always"]]; THR = float(cfg.get("thr", 0.05)); TOPK = 3
CAM = json.load(open(f"{HERE}/rma_agentview_camera.json")); M = np.asarray(CAM["matrix"], float); HW = tuple(CAM["hw"])
PRETTY = {'tomato_sauce': 'tomato sauce', 'orange_juice': 'orange juice', 'drawer_open': 'open drawer', 'drawer_content': 'drawer content', 'microwave_open': 'open microwave'}
pp.SEQUENTIAL = False

# ---- helpers (verbatim from build_rma_corpus.py: train/serve parity) ----
def project(p):
    w = np.array([p[0], p[1], p[2], 1.0]); c = M @ w; pix = c[:2] / c[2]
    return int(np.clip(round(pix[1]), 0, HW[0] - 1)), int(np.clip(round(pix[0]), 0, HW[1] - 1))
def tilt(aa): R = Rot.from_rotvec(aa).as_matrix(); return float(np.degrees(np.arccos(np.clip(-R[2, 2], -1, 1))))
def det_string(dets):
    best = {}
    for name, r, c, sc, *_ in dets:
        if sc >= DET_THR and (name not in best or sc > best[name][2]): best[name] = (r, c, sc)
    return ' '.join(f'{n}<{r}, {c}>' for n, (r, c, _) in sorted(best.items(), key=lambda kv: -kv[1][2]))
def state_string(s):
    r, c = project(s[:3]); q = s[6]
    return f"ee<{r}, {c}> grip:{'open' if q > 0.037 else 'closed'}({q:.3f}) z:{s[2]:.2f} tilt:{tilt(s[3:6]):.0f}"
def b64_to_pil(s): return Image.open(io.BytesIO(base64.b64decode(s))).convert("RGB")
def norm(t): return re.sub(r"\s+", " ", t.strip().lower().rstrip("."))

# ---- models ----
s3proc = s3 = None
if not (SAM_OFF and A1_OFF and not VERIFY_VISION):
    from transformers import Sam3Model, Sam3Processor
    s3proc = Sam3Processor.from_pretrained("facebook/sam3"); s3 = Sam3Model.from_pretrained("facebook/sam3", dtype=torch.bfloat16).to(DEV).eval()
agent = proc = None
if not A1_OFF:
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from peft import PeftModel
    base = os.environ.get("WAM_BASE", "Qwen/Qwen3.5-9B"); ck = os.environ["A1_CKPT"]
    proc = AutoProcessor.from_pretrained(base)
    agent = AutoModelForImageTextToText.from_pretrained(base, dtype=torch.bfloat16).to(DEV)
    agent = PeftModel.from_pretrained(agent, ck).eval()
print(f"[rma-agent] ready: sam={'OFF' if s3 is None else 'on'} verify_vision={VERIFY_VISION and s3 is not None} widths={len(WIDTHS)} writer={'OFF' if A1_OFF else os.environ.get('A1_CKPT')} harness={'OFF' if HARNESS_OFF else 'on'} port {PORT}", flush=True)

@torch.no_grad()
def detect(pils, cands):
    res = [[] for _ in pils]
    inp0 = s3proc(images=pils, text=[cands[0][0]] * len(pils), return_tensors="pt").to(DEV)
    ve = s3.get_vision_features(pixel_values=inp0["pixel_values"]); sizes = inp0["original_sizes"].tolist()
    for phrase, name in cands:
        tin = s3proc(images=pils, text=[phrase] * len(pils), return_tensors="pt").to(DEV)
        out = s3(vision_embeds=ve, input_ids=tin["input_ids"], attention_mask=tin["attention_mask"])
        for i, r in enumerate(s3proc.post_process_instance_segmentation(out, threshold=THR, mask_threshold=0.5, target_sizes=sizes)):
            order = sorted(range(len(r["scores"])), key=lambda j: -float(r["scores"][j]))[:TOPK]
            for j in order:
                x0, y0, x1, y1 = [float(v) for v in r["boxes"][j].tolist()]
                res[i].append([name, int(round((y0 + y1) / 2)), int(round((x0 + x1) / 2)), round(float(r["scores"][j]), 3)])
    return res

def cands_for(instr, plan):
    bag = set(w for w in re.split(r"[^a-z]+", (instr + " " + " ".join(plan)).lower()) if w)
    return [(p, n) for p, n, trig in CONCEPTS if any(t in bag for t in trig)] + [(p, n) for p, n in ALWAYS]

@torch.no_grad()
def a1_write(rec, pils):
    txt = user_of(rec); parts = txt.split("<image>")
    content = [{"type": "text", "text": parts[0]}]
    for i, p in enumerate(parts[1:]):
        if i < len(pils): content.append({"type": "image", "image": pils[i]})
        content.append({"type": "text", "text": p})
    msgs = [{"role": "system", "content": [{"type": "text", "text": SYS}]}, {"role": "user", "content": content}]
    inp = proc.apply_chat_template([msgs], add_generation_prompt=True, enable_thinking=False, tokenize=True, return_dict=True, return_tensors="pt", padding=True).to(DEV)
    out = agent.generate(**inp, max_new_tokens=200, do_sample=False, pad_token_id=proc.tokenizer.pad_token_id)
    return proc.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()

STATE = {}
def log(rec):
    if TRACE:
        with open(TRACE, "a") as fh: fh.write(json.dumps(rec) + "\n")

def fired_for(j):
    """Step at which the predicate for plan[j] fires, scanning from the last verified completion."""
    st = np.asarray(STATE["states"], float); n = len(st)
    if n < 12 or j >= len(STATE["plan"]): return None
    a = (STATE["comps"][-1] + 1) if STATE["comps"] else 0
    a = max(a, STATE.get("min_start", {}).get(j, 0))
    if a >= n - 1: return None
    p = pp.predict(st, [(a, n - a, STATE["plan"][j])], [STATE["targets"][j] if j < len(STATE.get("targets", [])) else None],
                   [STATE["widths"][j] if j < len(STATE.get("widths", [])) else None])[0][4]
    return p

REFUTE_TICKS = int(os.environ.get("REFUTE_TICKS", "3"))
def refute_or_phantom(j, p, pils):
    """True if the claim for plan[j] at step p is (still) refuted by vision. The
    first REFUTE_TICKS-1 refutations only DEFER it (the object may leave its rest
    spot a tick later, e.g. a grasped can carried off to pour); after REFUTE_TICKS
    consecutive refutations the event is treated as a phantom and the scan moves on."""
    if not vision_refutes(j, pils):
        STATE["refute_cnt"].pop((j, p), None); return False
    c = STATE["refute_cnt"].get((j, p), 0) + 1; STATE["refute_cnt"][(j, p)] = c; STATE["refuted"].append((j, p, c))
    if c >= REFUTE_TICKS: STATE["min_start"][j] = p + 1
    return True

def vision_refutes(j, pils):
    """Pick/place claim for plan[j] is refuted if the object is still detected at
    its resting position (recorded at tick 1). Only for objects SAM-3 saw at rest."""
    if s3 is None or not VERIFY_VISION: return False
    ins = STATE["plan"][j]; k = pp.kind_of(ins); o = obj_of(ins)
    if k not in ("pick", "place") or o is None: return False
    con = OBJ_CONCEPT.get(o); rest = STATE["rest"].get(con)
    if not con or not rest or con not in RELIABLE: return False
    # a rest position coinciding with a fixture detection means the phrase locked onto the fixture
    for fx in ("basket", "frypan", "mug", "drainer", "plate", "cabinet", "microwave", "drawer_open", "microwave_open"):
        fr = STATE["rest"].get(fx)
        if fr and ((fr[0] - rest[0]) ** 2 + (fr[1] - rest[1]) ** 2) ** 0.5 <= 20: return False
    phrase = next((p for p, n_, _ in CONCEPTS if n_ == con), None)
    if phrase is None: return False
    d = detect([pils[-1]], [(phrase, con)])[0]
    still = any(sc >= 0.3 and ((r - rest[0]) ** 2 + (c - rest[1]) ** 2) ** 0.5 <= 12 for _, r, c, sc in d)
    return still

def reset(req):
    raw_plan = req.get("plan", [])
    plan = [norm(x["instr"] if isinstance(x, dict) else x) for x in raw_plan]
    targets = [(tuple(x.get("median")) if isinstance(x, dict) and x.get("median") and x.get("kind") != "pick" else None) for x in raw_plan]
    STATE.clear(); STATE.update(targets=targets, widths=[band_for(x) for x in plan], min_start={}, rest={}, refuted=[], refute_cnt={})
    STATE.update(instr=req.get("instruction", "").strip().lower(), plan=plan,
                                task=req.get("task_id"), ep=req.get("ep"), j=0, states=[], comps=[], bank=[], tick=0, pending=[], admitted_ticks=[], settle=None)
    STATE["cands"] = cands_for(STATE["instr"], STATE["plan"])
    log({"kind": "reset", "task": STATE["task"], "ep": STATE["ep"], "plan": STATE["plan"], "concepts": [n for _, n in STATE["cands"]]})
    return {"ok": True, "n_concepts": len(STATE["cands"])}

def tick(req):
    t0 = time.time(); STATE["tick"] += 1; k = STATE["tick"]
    for s in req.get("states", []): STATE["states"].append([float(v) for v in s])
    step = int(req.get("step", len(STATE["states"])))
    pils = [b64_to_pil(s) for s in req["frames"]]; offs = req.get("offsets", [0, 5, 10])[:len(pils)]
    fstates = req.get("frame_states") or [STATE["states"][-1]] * len(pils)
    dets = [[] for _ in pils] if (SAM_OFF and A1_OFF) else detect(pils, STATE["cands"])
    if k == 1 and s3 is not None and VERIFY_VISION:
        d0 = dets[0] if any(dets) else detect([pils[0]], STATE["cands"])[0]
        for name, r, c, sc in d0:
            if name != "arm" and sc >= 0.25 and (name not in STATE["rest"] or sc > STATE["rest"][name][2]): STATE["rest"][name] = (r, c, sc)
    frame_dets = [f"{o:+d}: {det_string(d)} {state_string(np.asarray(s, float))}".strip() for o, d, s in zip(offs, dets, fstates)]
    plan, j = STATE["plan"], STATE["j"]; expected = plan[min(j, len(plan) - 1)] if plan else ""
    rec = {"instruction": STATE["instr"], "phase": "execution", "expected_next": expected, "bank": list(STATE["bank"][-8:]) if k > 1 else [], "frame_dets": frame_dets}
    raw = "" if A1_OFF else a1_write(rec, pils)
    lines = [l.strip() for l in raw.split("\n") if l.strip() and l.strip().upper() != "NONE"]
    verdicts = []; admitted_progress = False
    def _advance(p):
        nonlocal admitted_progress
        STATE["comps"].append(int(p)); STATE["j"] += 1; admitted_progress = True; STATE["settle"] = None
    def admit_progress(line, p):
        STATE["bank"].append(line)
        if SETTLE_TICKS > 0 and (len(STATE["states"]) - 1 - int(p)) < SETTLE_TICKS * TICK:
            STATE["settle"] = (int(p), STATE["j"]); verdicts.append(("settling", line)); return
        _advance(p)
    # a claim admitted earlier is still settling: hand off once the predicate has held long enough
    if STATE.get("settle") is not None:
        sp, sj = STATE["settle"]
        if sj != STATE["j"]: STATE["settle"] = None
        elif (len(STATE["states"]) - 1 - sp) >= SETTLE_TICKS * TICK:
            _advance(sp); verdicts.append(("settled", plan[min(sj, len(plan) - 1)]))
    # pending progress claims first (DEFER -> re-check)
    still = []
    for line in STATE["pending"]:
        p = fired_for(STATE["j"])
        if p is not None and refute_or_phantom(STATE["j"], p, pils):
            still.append(line); verdicts.append(("refute-pending", line)); continue
        if p is not None and norm(line.split("completed:", 1)[1].split("[sam]")[0]) == plan[min(STATE["j"], len(plan) - 1)]:
            admit_progress(line, p); verdicts.append(("admit-pending", line))
        else: still.append(line)
    STATE["pending"] = still
    for line in lines:
        if "completed:" in line:
            claim = norm(line.split("completed:", 1)[1].split("[sam]")[0]); jj = STATE["j"]
            if jj >= len(plan) or claim != plan[jj]:
                verdicts.append(("reject", line)); continue                  # not the current plan step
            if STATE.get("settle") is not None and STATE["settle"][1] == jj:
                verdicts.append(("settling-dup", line)); continue          # already verified, waiting to settle
            if HARNESS_OFF: admit_progress(line, step); verdicts.append(("admit-unverified", line)); continue
            p = fired_for(jj)
            if p is not None and refute_or_phantom(jj, p, pils):
                STATE["pending"].append(line); verdicts.append(("refute-defer", line)); continue
            if p is not None: admit_progress(line, p); verdicts.append(("admit", line))
            else: STATE["pending"].append(line); verdicts.append(("defer", line))
        else:
            if STATE["bank"] and norm(STATE["bank"][-1]) == norm(line): verdicts.append(("reject-dup", line)); continue
            STATE["bank"].append(line); verdicts.append(("admit", line))
    if not HARNESS_OFF and not admitted_progress and STATE["j"] < len(plan) and STATE.get("settle") is None:
        p = fired_for(STATE["j"])
        if p is not None and refute_or_phantom(STATE["j"], p, pils):
            verdicts.append(("refute-auto", plan[STATE["j"]])); p = None
        if p is not None and (len(STATE["states"]) - 1 - p) >= AUTO_GRACE * TICK:
            line = f"[event] completed: {plan[STATE['j']]}  [sam] {state_string(np.asarray(STATE['states'][p], float))} (auto)"
            admit_progress(line, p); verdicts.append(("auto", line)); STATE["pending"] = []
    sub = plan[min(STATE["j"], len(plan) - 1)] if plan else STATE["instr"]
    log({"kind": "tick", "task": STATE["task"], "ep": STATE["ep"], "tick": k, "step": step, "frame_dets": frame_dets, "writer": raw,
         "verdicts": verdicts, "j": STATE["j"], "subgoal": sub, "bank_len": len(STATE["bank"]), "rest": STATE.get("rest"), "secs": round(time.time() - t0, 2)})
    return {"subgoal": sub, "j": STATE["j"], "bank": STATE["bank"], "verdicts": verdicts, "secs": round(time.time() - t0, 2)}

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, obj, code=200):
        b = json.dumps(obj).encode(); self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self): self._send({"ok": True, "tick": STATE.get("tick", 0), "j": STATE.get("j", 0)})
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0)); req = json.loads(self.rfile.read(n) or b"{}")
        try:
            if self.path == "/reset": self._send(reset(req))
            elif self.path == "/tick": self._send(tick(req))
            elif self.path == "/end":
                log({"kind": "end", "task": STATE.get("task"), "ep": STATE.get("ep"), "success": req.get("success"), "j": STATE.get("j"), "bank": STATE.get("bank", [])}); self._send({"ok": True})
            else: self._send({"error": "unknown path"}, 404)
        except Exception as e:
            import traceback; traceback.print_exc(); self._send({"error": str(e)}, 500)

if __name__ == "__main__":
    srv = HTTPServer(("0.0.0.0", PORT), H)      # binds first: a port collision raises here, before any "listening" line
    print(f"[rma-agent] listening on port {PORT} pid {os.getpid()}", flush=True)
    srv.serve_forever()

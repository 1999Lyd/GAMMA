#!/usr/bin/env python
"""WAM two-agent subgoal service.

Hosts the detector + agent-1 (memory writer) + agent-2 (planner) behind a
tiny HTTP API so the RoboMME eval client (older transformers, no peft) can
drive them.  One bank per episode, exactly as in training:

  POST /reset {ep, instruction, demo_frames[b64]}  -> build bank from the demo
  POST /tick  {frames[b64], step}                  -> agent-1 writes, agent-2 plans
  GET  /health

Env: A1_CKPT, A2_CKPT, PORT, DEVICE.
"""
import base64, io, json, os, re, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch
# 2026-09-09: the cuDNN SDPA backend crashed two agent servers mid-lane
# (RuntimeError: mha_graph.execute ... is_good() false); flash/mem-efficient
# SDPA remain enabled.
torch.backends.cuda.enable_cudnn_sdp(False)
from PIL import Image

A1_CKPT = os.environ["A1_CKPT"]
A2_CKPT = os.environ["A2_CKPT"]
PORT = int(os.environ.get("PORT", "8899"))
SNAP, FSTEP = 16, 4
# v24: BinFill deposit dwell -- windows the picked cube's colour must be absent
# from its pick origin before a deposit is auto-claimed (measured carry: median
# 4, p90 6 ticks; the writer's put line arrives median 3 ticks after the
# oracle's switch and is missing in 24% of carries at 0.8B)
BF_PUT_DWELL = 7
import container_binding as CB
# v14 (user): one window geometry everywhere -- training now generates the
# demo phase in the same 16-step/5-frame windows served here (v13 trained
# demos at 8-step/3-frame against this 16-step replay: every closed-loop
# demo tick was off-distribution; vrp identity line 31/34 open vs 0 closed).
COORD = re.compile(r"<\s*(\d+)\s*,\s*(\d+)\s*>")

os.environ.setdefault("HF_HOME", os.path.expandvars("${HF_HOME}"))
from transformers import (AutoModelForImageTextToText, AutoProcessor,
                          AutoModelForZeroShotObjectDetection)
from peft import PeftModel

t0 = time.time()
DEV = "cuda:0"

# ---- detector (identical config to the offline cache) -----------------------
# ---- detector: SAM 3 + HSV highlight channel, IDENTICAL to the cache
# generator (sam3_precompute.py) that produced agent-1's training inputs.
# The previous serve-time detector was Grounding-DINO while training moved to
# SAM 3 -- a train/serve mismatch that collapsed the highlight channel and
# degraded every task in closed loop (SR 0.481 -> 0.439 despite better
# open-loop agreement).
from transformers import Sam3Model, Sam3Processor
s3proc = Sam3Processor.from_pretrained("facebook/sam3")
s3model = Sam3Model.from_pretrained("facebook/sam3",
                                    dtype=torch.bfloat16).to(DEV).eval()
CON = [("a small red block on the table", "red_cube"),
       ("a small green block on the table", "green_cube"),
       ("a small blue block on the table", "blue_cube"),
       ("white container box", "container"),
       ("a grey push button on the table", "button"),
       ("a flat square target marker on the table", "target"),
       ("a white open box", "bin"),
       ("a long cylindrical peg on the table", "peg"),
       ("a thin stick standing on the table", "stick"),
       ("robot arm", "arm")]
THR = 0.35


def white_blobs(rgb, min_area=40):
    from scipy import ndimage
    a = rgb.astype(np.float32) / 255.0
    mx, mn = a.max(-1), a.min(-1)
    v = mx * 255.0
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0) * 255.0
    m = (v > 185) & (sat < 50)
    lab, n = ndimage.label(m)
    out = []
    for i in range(1, n + 1):
        ys, xs = np.where(lab == i)
        if len(ys) < min_area:
            continue
        wx, wy = int(round(ys.mean())), int(round(xs.mean()))
        if wx > 45 and 14 < wy < 242:
            out.append((wx, wy))
    return out


@torch.no_grad()
def detect(pils):
    res = [[] for _ in pils]
    for phrase, name in CON:
        inp = s3proc(images=pils, text=[phrase]*len(pils),
                     return_tensors="pt").to(DEV)
        out = s3model(**inp)
        pp = s3proc.post_process_instance_segmentation(
            out, threshold=THR, mask_threshold=0.5,
            target_sizes=inp.get("original_sizes").tolist())
        for i, r in enumerate(pp):
            for box, sc in zip(r["boxes"], r["scores"]):
                x0, y0, x1, y1 = box.tolist()
                x, y = int(round((y0+y1)/2)), int(round((x0+x1)/2))
                if not (5 < x < 251 and 5 < y < 251):
                    continue
                res[i].append((name, x, y, float(sc)))
    clean = []
    for i, dets in enumerate(res):
        dets.sort(key=lambda d: -d[3])
        keep = []
        for n, x, y, sc in dets:
            if any((x-a)**2 + (y-b)**2 <= 10**2 for _, a, b in keep):
                continue
            keep.append((n, x, y))
        # highlight is NEVER deduped against object detections -- identical to
        # the repaired cache (the decal surrounds a cube; dedup cut ph
        # display-window recall to 0.20).  Train and serve must match exactly.
        # ph-only channel: on the other 15 tasks white containers/bins/
        # targets read as highlights (measured non-discriminative), so the
        # cache carries highlights only for ph and the live detector matches:
        # active only when the instruction speaks of a highlighted cube.
        if "highlight" in STATE.get("instr", "").lower():
            for hx, hy in white_blobs(np.array(pils[i])):
                keep.append(("highlight", hx, hy))
        clean.append(keep)
    # v14 (user): rs arc-tip channel -- the demo draws a white arc; the tip
    # (newest white pixels between consecutive stride-4 frames) is appended
    # as an 'arc_tip' detection at window-INTERIOR frames, exactly as in
    # training (f%16 in 4,8,12 -> within-window diffs, frames i=1..3).
    instr_l = STATE.get("instr", "").lower()
    if "navigate around" in instr_l or "retrace" in instr_l:
        def _wm(im_):
            a = np.asarray(im_, dtype=np.int16)
            mx = a.max(axis=2); mn = a.min(axis=2)
            m = (mx > 190) & ((mx - mn) < 28)
            m[:40, :] = False
            return m
        masks = [_wm(pp_) for pp_ in pils]
        for i in range(1, min(len(clean), 4)):
            new = masks[i] & ~masks[i-1]
            ys, xs = np.nonzero(new)
            if len(ys) >= 3:
                clean[i].append(("arc_tip", int(ys.mean()), int(xs.mean())))
    return clean


# ---- agents -----------------------------------------------------------------
MID = os.environ.get("WAM_BASE", "Qwen/Qwen3-VL-4B-Instruct")
proc = AutoProcessor.from_pretrained(MID)
proc.tokenizer.padding_side = "left"
_base = AutoModelForImageTextToText.from_pretrained(MID, dtype=torch.bfloat16,
                                                    device_map=DEV)
agent = PeftModel.from_pretrained(_base, A1_CKPT, adapter_name="a1")
agent.load_adapter(A2_CKPT, adapter_name="a2")
agent.eval()
print(f"[wam] models ready ({(time.time()-t0)/60:.1f} min)", flush=True)

SYS1 = ("You are the robot's memory writer (agent-1). Each tick you watch the "
        "last 16 control steps (sampled frames with per-frame object detections), "
        "read the memory bank and the planner's expected next subgoal, and "
        "write ONE line faithfully describing what happened in this window "
        "with grounded object coordinates, in the format '[event] ...  [sam] "
        "name<x, y>'. Write one line per event; if two things happened, "
        "write two lines. If nothing new happened, write NONE.")
# a2 input mode: default is bank-only (no image); WAM_A2_IMG=1 restores the
# legacy frame-in-context prompt for checkpoints trained with the frame.
A2_IMG = os.environ.get("WAM_A2_IMG", "0") == "1"
# WAM_HARNESS_OFF=1: disable ALL serve-time propose-verify machinery for the
# harness on/off ablation -- press-verify gate, deterministic rs/pl demo
# records + stroke advance/pin, stopcube trigger, grounding CORRECT, dedup
# REJECT. Agent claims then take effect exactly as proposed.
HARNESS_OFF = os.environ.get("WAM_HARNESS_OFF", "0") == "1"
# WAM_V19_GATES=1: full-coverage harness round (0.8B finding: claim types
# reliable at 9B become schedule-driven at small scale) -- pick-completion
# DEFER, place/drop-arrival DEFER, and the demo-sequence coordinate pin.
# Opt-in so v18-config arms stay comparable.
V19 = os.environ.get("WAM_V19_GATES", "0") == "1"


def _v19_video():
    """v19 extra gates are scoped to tasks where the 0.8B round MEASURED the
    failure classes (user rule: never touch tasks the VLM already handles).
    Video family: premature picks + wrong-step citations."""
    il = STATE.get("instr", "").lower()
    # v22: PickXtimes joins (measured: 4/9 failures = writer silent on
    # "place ... onto the target", 64-70-tick stalls; place evidence above
    # is clean on this task). BinFill deliberately NOT included: a bin
    # deposit is unobservable to the detector (cube hidden, arm centroid
    # far), so gating it would deadlock.
    return "watch the video" in il or "repeating this action" in il


def _v19_button():
    return "button" in STATE.get("instr", "").lower() and \
        "stop the cube" not in STATE.get("instr", "").lower()
if HARNESS_OFF:
    os.environ["WAM_NO_CORRECT"] = "1"
SYS2_IMG = ("You are the robot's planner (agent-2). You read the task, the "
        "robot's memory bank of grounded events, and the current camera frame "
        "with its detections, and you output the robot's current subgoal for "
        "this control chunk. Use the current frame to judge whether the "
        "ongoing subgoal is complete. Answer with the subgoal sentence only, "
        "lowercase, no trailing period, copying object coordinates from the "
        "memory bank or the current detections in the form <x, y>. During the "
        "demo phase answer exactly: hold and wait for the demo to complete.")
SYS2_NOIMG = ("You are the robot's planner (agent-2). You read the task and the "
        "robot's memory bank of grounded events, and you output the robot's "
        "current subgoal for this control chunk. Judge progress from the "
        "bank's completed lines. Answer with the subgoal sentence only, "
        "lowercase, no trailing period, copying object coordinates from the "
        "memory bank in the form <x, y>. During the "
        "demo phase answer exactly: hold and wait for the demo to complete.")
SYS2 = SYS2_IMG if A2_IMG else SYS2_NOIMG


@torch.no_grad()
def run(adapter, msgs, max_new):
    agent.set_adapter(adapter)
    inp = proc.apply_chat_template([msgs], add_generation_prompt=True, enable_thinking=False,
                                   tokenize=True, return_dict=True,
                                   return_tensors="pt", padding=True).to(DEV)
    out = agent.generate(**inp, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=proc.tokenizer.pad_token_id)
    n = inp["input_ids"].shape[1]
    return proc.decode(out[0][n:], skip_special_tokens=True).strip()


def a1_write(instr, phase, expected, bank, pils, dets, step):
    stp = 2 if len(pils) == 9 else FSTEP
    per = [f"<image> +{i*stp}: " +
           ("(sam skipped)" if d is None else
            (" ".join(f"{n}<{x}, {y}>" for n, x, y in d) or "(none)"))
           for i, d in enumerate(dets)]
    bk = "\n".join(bank[-8:]) if bank else "(empty)"
    txt = (f"Task: {instr}\nPhase: {phase}\n"
           f"Expected next subgoal from the planner: {expected}\n"
           f"Memory bank so far:\n{bk}\n"
           f"Current window frames and detections:\n" + "\n".join(per) +
           "\nWrite the memory line for this window (or NONE).")
    parts = txt.split("<image>")
    content = [{"type": "text", "text": parts[0]}]
    for i, p in enumerate(parts[1:]):
        if i < len(pils):
            content.append({"type": "image", "image": pils[i]})
        content.append({"type": "text", "text": p})
    return run("a1", [{"role": "system", "content": [{"type": "text", "text": SYS1}]},
                      {"role": "user", "content": content}], 280)


def a2_plan(instr, phase, bank, step, cur_pil=None, cur_det="(none)"):
    bk = "\n".join("  " + l for l in bank) if bank else "  (empty)"
    phase_q = (f"It is now the "
               f"{'DEMO phase' if phase == 'demo' else 'EXECUTION phase'}.\n"
               "What is the robot's current subgoal?")
    if A2_IMG:  # legacy frame-in-context mode
        pre = (f"Task: {instr}\nMemory bank:\n{bk}\n"
               f"Current frame and detections:\n")
        post = f" now: {cur_det}\n" + phase_q
        content = [{"type": "text", "text": pre}]
        if cur_pil is not None:
            content.append({"type": "image", "image": cur_pil})
        content.append({"type": "text", "text": post})
    else:       # default: bank-only reasoner, no pixels
        content = [{"type": "text",
                    "text": f"Task: {instr}\nMemory bank:\n{bk}\n" + phase_q}]
    return run("a2", [{"role": "system", "content": [{"type": "text", "text": SYS2}]},
                      {"role": "user", "content": content}], 48)


def b64_to_pil(s):
    return Image.open(io.BytesIO(base64.b64decode(s))).convert("RGB")


STATE = {"bank": [], "instr": "", "expected": "(none yet)", "ep": None,
         "task": "", "tick": 0}


def _lift_cls(text):
    """Object class of a pick claim ('pick up the container ... that hides
    the green cube' is a CONTAINER pick -- check container detections, not
    cubes; v20: the class-blind check made container picks self-verify
    against trivially-true cube absence)."""
    t = (text or "").lower()
    for k in ("container", "peg", "stick", "cube"):
        if k in t:
            return k
    return "cube"


def _cube_lifted(dets, cc, near=14.0, need=2, cls="cube"):
    """Pick-completion evidence (v19, WAM_PICK_GATE=1): the object seen at
    cc is gone from its rest spot (grasped/occluded by the gripper) for >=
    `need` consecutive window frames. Mirrors _press_occluded. Motivated by
    the 0.8B round: the small writer completes picks on its learned
    schedule (10/27 VPB failures = premature pick->place); at 9B this
    evidence is present whenever claimed, so the gate is a no-op there.
    ABSENCE-ONLY: adequate for gating writer claims (a premature claim
    cites where the object still IS), NOT for emitting auto-claims -- use
    _lift_transition there."""
    if not dets:
        return False
    run = 0
    for d in dets:
        if d is None:
            run = 0
            continue
        present = any(cls in n and (x - cc[0]) ** 2 + (y - cc[1]) ** 2
                      <= near ** 2 for n, x, y in d)
        run = 0 if present else run + 1
        if run >= need:
            return True
    return False


def _lift_transition(prev, dets, cc, near=14.0, need=2, cls="cube"):
    """v20 auto-claim pick evidence: PRESENCE then ABSENCE. The claimed
    object must have stood at cc (previous or current window) before
    vanishing for >= `need` trailing frames. Plain absence cannot fire an
    auto-claim: VPB ep1 auto-claimed a pick at a2's stale demo coordinate
    where no cube ever stood (live cube 17px away), desyncing the plan for
    the full 82-tick horizon; VU auto-claimed container picks against
    cube-class absence in 17/30 episodes."""
    frames = [d for d in list(prev or []) + list(dets or [])
              if d is not None]
    if not frames:
        return False
    was = [any(cls in n and (x - cc[0]) ** 2 + (y - cc[1]) ** 2 <= near ** 2
               for n, x, y in d) for d in frames]
    if not any(was):
        return False
    last = max(i for i, w in enumerate(was) if w)
    return len(was) - 1 - last >= need


def _placed_at(dets, tc):
    """Place/drop-completion evidence at target tc. Visible placements
    (targets, plates): a MOVABLE detection within radius. CONTAINMENT
    targets (bin/container/drainer): the deposited object is hidden inside,
    so movable-arrival is unobservable by construction -- evidence is the
    arm's deposit visit (arm detection within radius)."""
    if not dets:
        return False
    contain = any(
        any(any(k in n for k in ("bin", "container", "drainer"))
            and (x - tc[0]) ** 2 + (y - tc[1]) ** 2 <= 20 ** 2
            for n, x, y in (d or []))
        for d in dets if d)
    if contain:
        return any(
            any(n == "arm" and (x - tc[0]) ** 2 + (y - tc[1]) ** 2 <= 22 ** 2
                for n, x, y in (d or []))
            for d in dets if d)
    # v21c RELEASE EVIDENCE (VRP ep7: "completed: put it down at <58,97>"
    # admitted while the cube was still in the gripper at <58,98> -- the held
    # cube itself satisfied "movable within radius"). A placement is
    # witnessed only when a movable object rests at the target WITHOUT the
    # arm on it: no arm detection within 24px of the target in that frame.
    # v22 WINDOW-DWELL: measured on PickX (42 true placements / 409 hover
    # ticks): "movable within 16px in the LAST frame" fires on 13% of hover
    # ticks; "in EVERY frame of the window" fires on 0% of them (recall
    # 0.69 at the switch tick, and the object keeps resting there until the
    # plan advances, so the claim lands a tick later at worst). The arm
    # detection is the wrist centroid (never within 40px of a deposit
    # point), so arm-distance carries no evidence here.
    frames = [d for d in dets if d]
    if not frames:
        return False
    if not STATE.get("place_strict", True):
        # v23 repeat tasks (PickX): v21c evidence -- movable at the target
        # in the last frame without the arm on it
        d = frames[-1]
        return any(any(k in n for k in ("cube", "peg", "container"))
                   and (x - tc[0]) ** 2 + (y - tc[1]) ** 2 <= 16 ** 2
                   for n, x, y in d) and not any(
            n == "arm" and (x - tc[0]) ** 2 + (y - tc[1]) ** 2 <= 24 ** 2
            for n, x, y in d)
    return all(any(any(k in n for k in ("cube", "peg", "container"))
                   and (x - tc[0]) ** 2 + (y - tc[1]) ** 2 <= 16 ** 2
                   for n, x, y in d) for d in frames)


def _press_occluded(dets, bc, near=12.0, arm_r=45.0, need=2):
    """True when the window shows the button at bc occluded for >= `need`
    consecutive frames with the arm on/next to it -- the visual signature
    of a REGISTERED press (validated on the 30-episode bus audit; a hover
    that fails to register never occludes the button det)."""
    if not dets:
        return False
    run = 0
    for d in dets:
        if d is None:
            run = 0
            continue
        vis = False
        ad = 1e9
        for n, r_, c_ in d:
            dist = ((r_ - bc[0])**2 + (c_ - bc[1])**2) ** 0.5
            if dist <= near and ("button" in n or n == "container"):
                vis = True
            if n == "arm":
                ad = min(ad, dist)
        if not vis and ad <= arm_r:
            run += 1
            if run >= need:
                return True
        else:
            run = 0
    return False


def bank_append(step, ln):
    # Dedup CONSECUTIVE repeats only (writer stutter).  Distant identical
    # lines are REAL recurrences -- pickx/binfill/vrp legitimately complete
    # the same action at the same coordinates several times per episode, and
    # a global dedup silently erased the 2nd+ completions, structurally
    # breaking the counting tasks (v12 closed loop ran with that bug).
    body = ln.strip()
    if not HARNESS_OFF and STATE["bank"] and STATE["bank"][-1] == body:
        return
    STATE["bank"].append(body)
# live diagnostic log: one JSON record per tick and per episode reset, so the
# bank the policy actually acted on can be replayed after the fact
TRACE = os.environ.get("WAM_TRACE",
                       os.path.expandvars("${GAMMA_DATA}/traces/wam_live_bank.jsonl"))
os.makedirs(os.path.dirname(TRACE), exist_ok=True)


def log(rec):
    rec["t"] = round(time.time() - t0, 1)
    with open(TRACE, "a") as f:
        f.write(json.dumps(rec) + "\n")


def build_from_demo(instr, frames, cur=None):
    """Tick agent-1 over the demo prefix exactly as in training: an initial
    scene line at frame 0, then one window per 16 demo steps."""
    STATE["bank"] = []
    # tasks with no demo still get the initial-scene line, from the first
    # execution frame -- the bank must never start empty (as in training)
    seed = frames[0] if frames else cur
    if seed is None:
        return
    # v18 (user: "do eval demo side"): DETERMINISTIC RS DEMO RECORD.  The
    # eval demo prefix is K back-to-back 50-frame circling strokes; the red
    # marker stands at each stroke's start pad and the stick's fast-fading
    # ink wisp samples the arc.  rs_dir_algo recovers every stroke's
    # direction and side from those two signals (33/33 = 1.00 on the
    # regenerated GT demos for all 10 test episodes; the v15c-v17 tip-chain
    # override never exceeded chance because it segmented by temporal gaps
    # -- strokes are CONTIGUOUS, NO-RECORD transit frames are filtered out
    # by the env -- and measured sweeps around one global center while each
    # stroke pivots around its own cylinder).
    if "navigate around" in (instr or "").lower():
        import numpy as _np
        import rs_dir_algo as _RD
        try:
            STATE["rs_strokes"] = _RD.stroke_dirs(
                [_np.asarray(f) for f in frames])
        except Exception as _e:
            STATE["rs_strokes"] = []
            STATE["rs_stroke_err"] = repr(_e)
        STATE["cw_log"] = [([s["med"], s["n"], s["lat"], s["dir"]]
                            if s else None)
                           for s in STATE.get("rs_strokes", [])]
    # v18 (user: "do eval demo side"): DETERMINISTIC PL DEMO RECORD.  The
    # PatternLock demo lights the route's dots in sequence (start dot at
    # frame 0; the next dot lights ~7 frames before the current stroke
    # ends).  The dot chain gives every stroke's endpoints INCLUDING the
    # fore/aft component no arm/ink channel could measure (10/10 GT
    # episodes exact, and consecutive same-word strokes -- invisible in the
    # oracle's run-length view -- are recovered separately).
    if "retrace" in (instr or "").lower():
        import numpy as _np
        import rs_dir_algo as _RD
        try:
            _st, _ap = _RD.pl_route([_np.asarray(f) for f in frames])
        except Exception as _e:
            _st, _ap = [], []
            STATE["pl_stroke_err"] = repr(_e)
        STATE["pl_strokes"], STATE["pl_appear"] = _st, _ap
        STATE["cw_log"] = [([s["word"], s["A"], s["B"]] if s else None)
                           for s in _st]
    d0 = detect([seed])[0]
    init = ("[event] initial scene: "
            + (", ".join(f"{n.replace('_',' ')} at <{x}, {y}>" for n, x, y in d0)
               or "empty table")
            + "  [sam] "
            + (" ".join(f"{n}<{x}, {y}>" for n, x, y in d0) or "(none)"))
    STATE["bank"].append(init)
    # v23 (user 2026-09-10: "gate it by checking which pad is occluded by the
    # cube"): VideoPlaceButton DEMO READER. The 0.8B writer omits the press
    # line in 64/69 serve episodes and mislabels drops, so the reasoner's
    # pad is right in 2/69. The cube's detected track gives the pads it
    # rests on (a pad's own detection vanishes under the cube); the demo is
    # always pick-drop-press-pick-drop-pick-drop(table), so "right before /
    # after the press" = 1st / 2nd pad landing (corpus: 82% recovery of the
    # oracle pad from detections alone). The pad pins the reasoner's
    # "place ... correct target" citation at /tick (CORRECT).
    vpb_task = "right after the button" in (instr or "").lower() or \
        "right before the button" in (instr or "").lower()
    _vpb_track, _vpb_targets = {}, []
    _vpb_col = (re.search(r"(red|green|blue) cube", (instr or "").lower())
                or [None, "cube"])[1]
    for end in range(SNAP, len(frames), SNAP):
        idx = list(range(end-SNAP, end+1, FSTEP))
        pils = [frames[min(i, len(frames)-1)] for i in idx]
        dets = detect(pils)
        if vpb_task and not HARNESS_OFF:
            for _i, _d in zip(idx, dets):
                _cs = [(x, y) for n, x, y in (_d or []) if n == f"{_vpb_col}_cube"]
                _vpb_track[_i] = _cs[0] if _cs else None
                _vpb_targets += [(x, y) for n, x, y in (_d or []) if n == "target"]
            STATE["_vpb_demo"] = (instr, _vpb_track, _vpb_targets)
        rs_task = "navigate around" in (instr or "").lower()
        pl_task = "retrace" in (instr or "").lower()
        if rs_task or pl_task:
            for d in dets:
                for n, x, y in d:
                    if n == "arc_tip":
                        STATE["arc_hist"].append((x, y))
                    elif n == "stick":
                        STATE["stick_pts"].append((x, y))
        line = a1_write(instr, "demo", "hold and wait for the demo to complete",
                        STATE["bank"], pils, dets, end)
        if line.strip().upper() != "NONE" and line.strip():
            for ln in line.split("\n"):
                if ln.strip() and ln.strip().upper() != "NONE":
                    # v18 (user): the VLM's pl demo lines are DROPPED -- the
                    # deterministic dot-route record below replaces them
                    # (supersedes the v15 lateral-only override: the dot
                    # chain also gives fore/aft, which was VLM-guessed).
                    if not HARNESS_OFF and pl_task and "demo showed" in ln:
                        STATE["pl_vlm_dropped"] = \
                            STATE.get("pl_vlm_dropped", 0) + 1
                        continue
                    # v18 (user): the VLM's rs circling lines are DROPPED --
                    # the deterministic stroke record below replaces them
                    # (writer words at serve were stereotyped guesses, bank
                    # cw/ccw fidelity 4/17 in the v17 trace audit).
                    if not HARNESS_OFF and rs_task and "circling" in ln.lower():
                        STATE["rs_vlm_dropped"] = \
                            STATE.get("rs_vlm_dropped", 0) + 1
                        continue
                    bank_append(end, ln)
        # v18: deterministic numbered-plain stroke lines, emitted in the
        # exact v17 training format at the window containing each stroke's
        # end (stroke k spans demo frames [50k, 50k+50)).
        if not HARNESS_OFF and rs_task and STATE.get("rs_strokes"):
            _last_end = ((len(frames) - 1) // SNAP) * SNAP
            for _k, _s in enumerate(STATE["rs_strokes"]):
                if _s is None:
                    continue
                _b = min(50 * (_k + 1), _last_end)
                if end - SNAP < _b <= end:
                    bank_append(end, f"[event] demo showed: stroke {_k+1}: "
                                f"move to the nearest {_s['lat']} target by "
                                f"circling around the stick {_s['dir']}"
                                f"  [sam] (none)")
                    STATE["cw_overridden"] = \
                        STATE.get("cw_overridden", 0) + 1
        # v18: deterministic pl stroke lines -- stroke k ends ~7 frames
        # after its destination dot lights up (appear[k+1]); the last
        # stroke's line lands in the final window.
        if not HARNESS_OFF and pl_task and STATE.get("pl_strokes"):
            _ap = STATE.get("pl_appear", [])
            _last_end = ((len(frames) - 1) // SNAP) * SNAP
            for _k, _s in enumerate(STATE["pl_strokes"]):
                if _s is None:
                    continue
                _b = min(_ap[_k + 1] + 8 if _k + 1 < len(_ap) else _last_end,
                         _last_end)
                if end - SNAP < _b <= end:
                    bank_append(end, f"[event] demo showed: stroke {_k+1}: "
                                f"move {_s['word']}  [sam] (none)")
                    STATE["lat_overridden"] = \
                        STATE.get("lat_overridden", 0) + 1


def _vpb_finish(instr, track, targets):
    """v23: compute the VPB pad from the demo cube track (see vpb_reader)."""
    try:
        import vpb_reader as _VR
        pad, drops = _VR.vpb_target(sorted(track.items()), targets, instr)
        STATE["vpb_pad"] = pad
        STATE["vpb_drops"] = [(d["frame"], d["pos"], d["pad"]) for d in drops]
    except Exception as _e:
        STATE["vpb_pad"] = None
        STATE["vpb_err"] = repr(_e)


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        self._send({"ok": True, "bank": len(STATE["bank"])})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        if self.path == "/reset":
            STATE["instr"] = req.get("instruction", "")
            STATE["ep"] = req.get("ep")
            STATE["expected"] = "(none yet)"
            frames = [b64_to_pil(s) for s in req.get("demo_frames", [])]
            cur = req.get("cur_frame")
            cur = b64_to_pil(cur) if cur else None
            t = time.time()
            STATE["arc_hist"] = []; STATE["stick_pts"] = []; STATE["arc_mark"] = 0
            STATE.pop("sc", None)
            for _k in ("rs_strokes", "pl_strokes", "pl_appear", "cw_log",
                       "rs_stroke_err", "pl_stroke_err", "rs_adv",
                       "rs_stall", "press_pending", "prev_dets",
                       "pl_adv", "pl_stall",
                       # v21b: v19/v20 per-episode state was never cleared --
                       # pending claims, the auto-claim/dedup memory and the
                       # swing side leaked into the next episode (SwingX ep4:
                       # all 10 writer lines rejected against ep3's side).
                       "pick_pending", "place_pending", "auto_claimed",
                       "last_move_side", "seq_pinned", "stale_fixed",
                       "cstate_fixed", "binfill_pinned",
                       "binfill_press_deferred",
                       # v23: VPB demo reader + colour-aware citation guard
                       "vpb_pad", "vpb_drops", "vpb_pinned", "cite_fixed",
                       "cls_hist", "place_strict",
                       "plan_pinned", "bf_origin", "bf_absent_run", "bf_put_auto"):
                STATE.pop(_k, None)
            STATE.pop("_vpb_demo", None)
            build_from_demo(STATE["instr"], frames, cur)
            if STATE.get("_vpb_demo"):
                _vpb_finish(*STATE.pop("_vpb_demo"))
            # v23: strict window-dwell placement evidence deadlocks repeat
            # tasks (PickX: 31% of true placements never admitted -> stuck);
            # keep it only where the place is the final step.
            STATE["place_strict"] = "repeat" not in STATE["instr"].lower()
            STATE["task"] = req.get("task", "")
            STATE["tick"] = 0
            log({"kind": "reset", "ep": STATE["ep"], "task": STATE["task"],
                 "cw_log": STATE.get("cw_log", []),
                 "cw_overridden": STATE.get("cw_overridden", 0),
                 "lat_overridden": STATE.get("lat_overridden", 0),
                 "instruction": STATE["instr"], "n_demo": len(frames),
                 "vpb_pad": STATE.get("vpb_pad"), "vpb_drops": STATE.get("vpb_drops"),
                 "bank_after_demo": list(STATE["bank"])})
            self._send({"bank": STATE["bank"], "n_demo": len(frames),
                        "secs": round(time.time()-t, 1)})
        elif self.path == "/tick":
            step = int(req.get("step", 0))
            pils = [b64_to_pil(s) for s in req["frames"]]
            dets = detect(pils)
            line = a1_write(STATE["instr"], "execution", STATE["expected"],
                            STATE["bank"], pils, dets, step)
            if line.strip().upper() != "NONE" and line.strip():
                for ln in line.split("\n"):
                    if ln.strip() and ln.strip().upper() != "NONE":
                        # v18c (user): PRESS-VERIFY GATE (bu/bus).  The
                        # writer's press completion is a learned CLOCK, not
                        # a percept (a registered press changes zero
                        # pixels): in the 30-episode bus audit it fired at
                        # step ~96 in 27/27 failures while the press had
                        # not registered, and the hold then walked the arm
                        # off the button (bus 3/30).  A real press puts the
                        # gripper ON the button: its detection vanishes for
                        # >=2 consecutive frames with the arm nearby --
                        # present in every registered standard-window press
                        # in the audit, absent in every unregistered one.
                        # Unverified completions are HELD (a2 keeps
                        # demanding the press; the arm keeps trying) and
                        # appended when the occlusion press is seen.
                        # v19: a completion already supplied by the
                        # evidence extractor (auto-claim below) supersedes
                        # the writer's version -- REJECT the duplicate so
                        # counting tasks never double-count.
                        _mdup = re.search(
                            r"completed: (press|pick|drop|place|put|insert)"
                            r".*?<\s*(\d+)\s*,\s*(\d+)\s*>", ln)
                        if V19 and not HARNESS_OFF and _mdup:
                            _dv = _mdup.group(1)
                            _dx, _dy = int(_mdup.group(2)), int(_mdup.group(3))
                            if any(_dv == v and abs(_dx-x)+abs(_dy-y) <= 14
                                   for v, x, y in STATE.get("auto_claimed", [])):
                                STATE["auto_dup_rejected"] = \
                                    STATE.get("auto_dup_rejected", 0) + 1
                                continue
                        _il = STATE.get("instr", "").lower()
                        # v20 SWING ALTERNATION at the writer gate: in a
                        # back-and-forth task two consecutive same-side
                        # move records are impossible; a writer move line
                        # for the side already on record (from the writer
                        # OR the auto-claim) is a duplicate -> REJECT.
                        # Admitted writer lines set the side too, so the
                        # auto-claim never doubles a writer record.
                        _msw = re.search(r"completed: move to the top of the "
                                         r"(left|right)-side target", ln)
                        if (not HARNESS_OFF and V19 and _msw
                                and "back-and-forth" in _il):
                            if STATE.get("last_move_side") == _msw.group(1):
                                STATE["swing_dup_rejected"] = \
                                    STATE.get("swing_dup_rejected", 0) + 1
                                continue
                            STATE["last_move_side"] = _msw.group(1)
                        _mpr = re.search(r"completed: press .*?button.*?"
                                         r"<\s*(\d+)\s*,\s*(\d+)\s*>", ln)
                        if (not HARNESS_OFF and _mpr and "button" in _il
                                and "stop the cube" not in _il):
                            _bc = (int(_mpr.group(1)), int(_mpr.group(2)))
                            if _press_occluded(dets, _bc) or \
                               _press_occluded(STATE.get("prev_dets"), _bc):
                                STATE["press_ok"] = \
                                    STATE.get("press_ok", 0) + 1
                            else:
                                STATE.setdefault("press_pending", [])
                                STATE["press_pending"].append((_bc, ln, STATE.get("tick", 0)))
                                STATE["press_held"] = \
                                    STATE.get("press_held", 0) + 1
                                continue
                        # v19 pick-completion gate: same DEFER contract, new
                        # claim type -- see _cube_lifted.
                        _mpk = re.search(r"completed: pick[^<]*"
                                         r"<\s*(\d+)\s*,\s*(\d+)\s*>", ln)
                        if not HARNESS_OFF and V19 and _v19_video() and _mpk:
                            _cc = (int(_mpk.group(1)), int(_mpk.group(2)))
                            _lc = _lift_cls(ln)
                            if _cube_lifted(dets, _cc, cls=_lc) or \
                               _cube_lifted(STATE.get("prev_dets"), _cc,
                                            cls=_lc):
                                STATE["pick_ok"] = \
                                    STATE.get("pick_ok", 0) + 1
                            else:
                                STATE.setdefault("pick_pending", [])
                                STATE["pick_pending"].append((_cc, ln, STATE.get("tick", 0)))
                                STATE["pick_held"] = \
                                    STATE.get("pick_held", 0) + 1
                                continue
                        # v19 place/drop-completion gate: arrival evidence --
                        # some non-arm detection sits within radius of the
                        # claimed target in the window; else DEFER.
                        _mpl = re.search(r"completed: (?:drop|place|put|insert)"
                                         r".*<\s*(\d+)\s*,\s*(\d+)\s*>", ln)
                        if not HARNESS_OFF and V19 and _v19_video() and _mpl:
                            _tc = (int(_mpl.group(1)), int(_mpl.group(2)))
                            _arr = _placed_at(dets, _tc) or \
                                _placed_at(STATE.get("prev_dets"), _tc)
                            if not _arr:
                                STATE.setdefault("place_pending", [])
                                STATE["place_pending"].append((_tc, ln, STATE.get("tick", 0)))
                                STATE["place_held"] = \
                                    STATE.get("place_held", 0) + 1
                                continue
                        bank_append(step, ln)
            # release held press completions once their occlusion press
            # is finally observed. v20 second release path (paired-trace
            # diagnosis: PH 7 harness-lost eps, all press-pin deadlocks):
            # some buttons never register occlusion under this detector, so
            # a sustained arm visit at the claimed button -- arm detection
            # within radius in BOTH the current and previous window, claim
            # at least one tick old -- is accepted as the press evidence.
            # Evidence-anchored (the arm demonstrably executed the press),
            # not a timeout.
            def _arm_at(ds, bc, r=22.0):
                return bool(ds) and any(
                    any(n == "arm" and (x-bc[0])**2 + (y-bc[1])**2 <= r*r
                        for n, x, y in (d or []))
                    for d in ds if d)
            if STATE.get("press_pending"):
                _still = []
                for _bc, _ln, _t0 in STATE["press_pending"]:
                    _armvisit = (STATE.get("tick", 0) > _t0
                                 and _arm_at(dets, _bc)
                                 and _arm_at(STATE.get("prev_dets"), _bc))
                    if _press_occluded(dets, _bc) or _armvisit:
                        bank_append(step, _ln)
                        STATE["press_released"] = \
                            STATE.get("press_released", 0) + 1
                        if _armvisit:
                            STATE["press_armvisit_released"] = \
                                STATE.get("press_armvisit_released", 0) + 1
                    else:
                        _still.append((_bc, _ln, _t0))
                STATE["press_pending"] = _still
            # release held pick completions once the cube's lift is observed
            if STATE.get("pick_pending"):
                _still = []
                for _cc, _ln, _t0 in STATE["pick_pending"]:
                    if _cube_lifted(dets, _cc, cls=_lift_cls(_ln)):
                        bank_append(step, _ln)
                        STATE["pick_released"] = \
                            STATE.get("pick_released", 0) + 1
                    else:
                        _still.append((_cc, _ln, _t0))
                STATE["pick_pending"] = _still
            # release held place completions on arrival evidence
            if STATE.get("place_pending"):
                _still = []
                for _tc, _ln, _t0 in STATE["place_pending"]:
                    _arr = _placed_at(dets, _tc)
                    if _arr:
                        bank_append(step, _ln)
                        STATE["place_released"] = \
                            STATE.get("place_released", 0) + 1
                    else:
                        _still.append((_tc, _ln, _t0))
                STATE["place_pending"] = _still
            # v19 AUTO-CLAIMS: evidence-driven completion supply for the
            # active subgoal (covers writer OMISSIONS without ever admitting
            # an unverified claim -- the same predicates that gate the
            # writer's claims here EMIT the claim when the evidence is
            # observed; "proposal and verification coincide").
            if V19 and not HARNESS_OFF:
                _act = (STATE.get("expected") or "").lower()
                _mac = re.search(r"\b(press|pick|drop|place|put|insert)\b"
                                 r".*?<\s*(\d+)\s*,\s*(\d+)\s*>", _act)
                if _mac:
                    _av = _mac.group(1)
                    _ac = (int(_mac.group(2)), int(_mac.group(3)))
                    _seen = any(
                        _av == v and abs(_ac[0]-x)+abs(_ac[1]-y) <= 14
                        for v, x, y in STATE.get("auto_claimed", []))
                    _have = any(
                        f"completed: {_av}" in ln and any(
                            abs(_ac[0]-int(a))+abs(_ac[1]-int(b)) <= 14
                            for a, b in COORD.findall(ln))
                        for ln in STATE["bank"])
                    _ev = False
                    _scope = (_v19_button() if _av == "press"
                              else _v19_video())
                    if not _seen and not _have and _scope:
                        if _av == "press" and "stop the cube" not in \
                                STATE.get("instr", "").lower():
                            _ev = _press_occluded(dets, _ac) or \
                                _press_occluded(STATE.get("prev_dets"), _ac)
                        elif _av == "pick":
                            # v20: auto-claims need the presence->absence
                            # TRANSITION -- absence alone self-verified
                            # picks at coordinates nothing ever occupied.
                            _ev = _lift_transition(
                                STATE.get("prev_dets"), dets, _ac,
                                cls=_lift_cls(_act))
                        elif _av in ("drop", "place", "put", "insert"):
                            # strict movable-arrival only: containment
                            # placements get NO auto-claim (arm-visit alone
                            # is too weak without a proposer)
                            _ev = any(
                                any(any(k in n for k in
                                        ("cube", "peg", "container"))
                                    and (x-_ac[0])**2 +
                                    (y-_ac[1])**2 <= 16**2
                                    for n, x, y in (d or []))
                                for d in dets if d)
                    if _ev:
                        bank_append(step, f"[event] completed: "
                                    f"{STATE.get('expected')}  [sam] (auto)")
                        STATE.setdefault("auto_claimed", [])
                        STATE["auto_claimed"].append(
                            (_av, _ac[0], _ac[1]))
                        STATE["auto_claims"] = \
                            STATE.get("auto_claims", 0) + 1
                # v20 SWING MOVE AUTO-CLAIM (paired-trace diagnosis, SwingX
                # 6 harness-lost eps): the 0.8B writer records only the
                # first left/right pair, the bank freezes, and the reasoner
                # guesses swing parity at chance. Evidence supplies the
                # missing records: a carried cube arriving at the active
                # move target writes the completion, with the repetition
                # ordinal counted from the bank so counting stays exact.
                # Alternation guard: the same target is never auto-claimed
                # twice in a row (a hover cannot double-count).
                _mmv = re.search(r"\bmove\b.*?(left|right)-side target"
                                 r".*?<\s*(\d+)\s*,\s*(\d+)\s*>", _act)
                # v21: OFF by default (WAM_SWING_AUTO=1 re-enables). Measured
                # on 1,402 SwingX ticks: a full-window dwell within 30px of
                # the target occurs on 25% of true arrivals vs 21% of
                # non-arrival ticks -- 2D position cannot read this
                # completion (height/dwell live in 3D), so the auto-claim
                # only races the env (v20 SwingX 5/30 vs off 17/30).
                if (_mmv and os.environ.get("WAM_SWING_AUTO") == "1"
                        and "back-and-forth"
                        in STATE.get("instr", "").lower()):
                    _side = _mmv.group(1)
                    _mc2 = (int(_mmv.group(2)), int(_mmv.group(3)))
                    # one record per swing, whichever source wrote it:
                    # last_move_side is set by auto-claims AND admitted
                    # writer move lines (writer duplicates of the same
                    # side are rejected at the claim gate above).
                    # Radius 30px: measured cube-to-target distance at 111
                    # real arrivals is 20-28px (cube held ABOVE the target;
                    # 16px caught 2%, 30px catches 100%).
                    if STATE.get("last_move_side") != _side:
                        _ev2 = any(
                            any("cube" in n and "container" not in n
                                and (x-_mc2[0])**2 +
                                (y-_mc2[1])**2 <= 30**2
                                for n, x, y in (d or []))
                            for d in dets if d)
                        if _ev2:
                            _k = 1 + sum(1 for ln in STATE["bank"]
                                         if "completed: move" in ln
                                         and f"{_side}-side" in ln)
                            _ord = {1: "first", 2: "second", 3: "third",
                                    4: "fourth", 5: "fifth", 6: "sixth"}
                            bank_append(step, "[event] completed: move to "
                                        f"the top of the {_side}-side "
                                        f"target at <{_mc2[0]}, {_mc2[1]}> "
                                        f"for the {_ord.get(_k, str(_k)+'th')} "
                                        f"time  [sam] (auto)")
                            STATE["last_move_side"] = _side
                            STATE["auto_move_claims"] = \
                                STATE.get("auto_move_claims", 0) + 1
            STATE["prev_dets"] = dets
            bank_before = list(STATE["bank"])
            cur_det = " ".join(f"{n}<{x}, {y}>" for n, x, y in dets[-1]) or "(none)"
            sub = a2_plan(STATE["instr"], "execution", STATE["bank"], step,
                          cur_pil=pils[-1], cur_det=cur_det)
            # v15b (user): DETERMINISTIC STOPCUBE PRESS TRIGGER.  The
            # instruction names the reach count N ("...for the fourth time");
            # the oracle presses 1.4-1.8 ticks (~22-29 steps) before the Nth
            # cube arrival (measured 91/94).  The system tracks arrivals from
            # detections, predicts the Nth from the observed period, and
            # fires the press ~28 steps ahead.  Agent-2's own press is
            # suppressed until the trigger fires; the latch below holds it.
            _instr_l = STATE.get("instr", "").lower()
            if (not HARNESS_OFF and "stop the cube" in _instr_l
                    and "reaches the target" in _instr_l):
                sc = STATE.setdefault("sc", {"arr": [], "prev_d": None,
                                             "fired": False})
                _ords = {"first": 1, "second": 2, "third": 3, "fourth": 4,
                         "fifth": 5, "sixth": 6}
                _mN = re.search(r"for the (\w+) time", _instr_l)
                _N = _ords.get(_mN.group(1) if _mN else "", 4)
                for _i, _d in enumerate(dets):
                    if _d is None:
                        continue
                    _cs = [(x, y) for n, x, y in _d if "cube" in n]
                    _ts = [(x, y) for n, x, y in _d if n == "target"]
                    if not _cs or not _ts:
                        continue
                    _dist = min(((c[0]-t[0])**2 + (c[1]-t[1])**2) ** 0.5
                                for c in _cs for t in _ts)
                    _fs = step - 16 + _i * (16 // max(len(dets)-1, 1))
                    if _dist < 20 and (sc["prev_d"] is None
                                       or sc["prev_d"] >= 20):
                        if not sc["arr"] or _fs - sc["arr"][-1] > 24:
                            sc["arr"].append(_fs)
                    sc["prev_d"] = _dist
                if not sc["fired"]:
                    _gaps = [b-a for a, b in zip(sc["arr"], sc["arr"][1:])]
                    _P = sorted(_gaps)[len(_gaps)//2] if _gaps else 80
                    if len(sc["arr"]) >= 1:
                        _pred = sc["arr"][-1] + _P * (_N - len(sc["arr"]))
                        if len(sc["arr"]) >= _N - 1 and _pred - step <= 28:
                            _t0 = [(x, y) for n, x, y in (dets[-1] or [])
                                   if n == "target"]
                            _tc = _t0[0] if _t0 else None
                            sub = ("press the button to stop the cube on the "
                                   + (f"target at <{_tc[0]}, {_tc[1]}>"
                                      if _tc else "target"))
                            sc["fired"] = True
                            STATE["sc_fired_at"] = step
                    if not sc["fired"] and "press the button" in sub.lower():
                        sub = "remain static"
                        STATE["sc_suppressed"] = \
                            STATE.get("sc_suppressed", 0) + 1
            # v15 (user): STOPCUBE PRESS LATCH -- once the press subgoal is
            # issued there is nothing after it in this task, so it never
            # reverts to "remain static" (kills the press-static-press
            # stutter that aborted the physical press in v14).
            if "press the button to stop" in STATE.get("expected", "").lower() \
                    and "press the button to stop" not in sub.lower():
                sub = STATE["expected"]
                STATE["press_latched"] = STATE.get("press_latched", 0) + 1
            # v15 (user): RS HOLD -- the recorded demo is a LIST; agent-2 has
            # never sequenced it reliably (correct-index 0.20 v13 / 0.17 v14,
            # invents cross-stroke combos).  The exec subgoal is pinned to
            # the k-th recorded stroke, k = stroke completions in the bank;
            # agent-1's completion lines are the only advance signal.
            if (not HARNESS_OFF
                    and "navigate around" in STATE.get("instr", "").lower()):
                demo = []
                for ln in STATE["bank"]:
                    if re.search(r"demo showed: (?:stroke \d+: )?move", ln):
                        t = ln.split("demo showed: ", 1)[1]
                        t = re.sub(r"^stroke \d+: ", "", t)
                        t = t.split("  [sam]")[0]
                        t = re.sub(r"\s*\(arm from [^)]*\)", "", t)
                        t = re.sub(r"\s*\(arc tip from [^)]*\)", "", t)
                        demo.append(t.strip())
                ncomp = sum(1 for ln in STATE["bank"]
                            if "completed: move" in ln)
                # v18b: DETERMINISTIC STROKE ADVANCE.  The writer's exec
                # completion line lags the oracle switch by up to 80 steps
                # or never comes (v18 run: ep2 lag-80, ep6 absent -- both
                # failed with a perfect record and correct hold).  The held
                # stroke's destination pad B is KNOWN from the demo record,
                # so tip/arm arrival advances the hold without waiting for
                # the writer.
                _strokes = STATE.get("rs_strokes") or []
                _adv = STATE.setdefault("rs_adv", 0)
                if _adv < len(_strokes) and _strokes[_adv]:
                    _bx, _by = _strokes[_adv]["B"]     # (col, row)
                    _hit = any((r_-_by)**2 + (c_-_bx)**2 <= 14**2
                               for d in dets if d
                               for n, r_, c_ in d if n == "arc_tip")
                    if not _hit:
                        # exec frames always show the stick's red tip band
                        # (~30px, pure red); the ink channel can vanish at
                        # the end-of-stroke dwell exactly when it matters
                        import numpy as _np
                        from scipy import ndimage as _ndi
                        for _p in pils:
                            _a = _np.asarray(_p)
                            _m = ((_a[..., 0] > 150) & (_a[..., 1] < 70)
                                  & (_a[..., 2] < 70))
                            _lab, _n = _ndi.label(
                                _ndi.binary_dilation(_m, iterations=2))
                            for _c in range(1, _n + 1):
                                _ys, _xs = _np.nonzero((_lab == _c) & _m)
                                if 8 <= len(_ys) <= 220 and \
                                   (_ys.mean()-_by)**2 + \
                                   (_xs.mean()-_bx)**2 <= 12**2:
                                    _hit = True
                                    break
                            if _hit:
                                break
                    if _hit:
                        STATE["rs_adv"] = _adv + 1
                        STATE["rs_advanced"] = \
                            STATE.get("rs_advanced", 0) + 1
                        STATE["rs_stall"] = 0
                # the deterministic counter is the SOLE authority (the
                # writer's completions ran EARLY in ep7 and never in ep6);
                # after a long stall with the writer ahead, accept ONE
                # writer completion so a missed arrival cannot pin forever
                STATE["rs_stall"] = STATE.get("rs_stall", 0) + 1
                if STATE["rs_stall"] > 6 and ncomp > STATE.get("rs_adv", 0):
                    STATE["rs_adv"] = STATE.get("rs_adv", 0) + 1
                    STATE["rs_stall"] = 0
                    STATE["rs_stall_adv"] = STATE.get("rs_stall_adv", 0) + 1
                ncomp = STATE.get("rs_adv", 0)
                if demo:
                    # clamp: the route IS the whole task -- after the last
                    # stroke either the env ends or the arm must still
                    # finish it; free-running a2 emitted invented combos
                    held = demo[min(ncomp, len(demo) - 1)]
                    if held.lower() != sub.strip().lower():
                        STATE["rs_held"] = STATE.get("rs_held", 0) + 1
                    sub = held
            # v18 (user): PL HOLD -- mirror of the rs hold.  The exec
            # subgoal is pinned to the k-th deterministic demo stroke,
            # k = stroke completions in the bank (the v17 trace showed a2
            # stuck on stroke 1's word while the oracle advanced).
            if (not HARNESS_OFF
                    and "retrace" in STATE.get("instr", "").lower()):
                demo = []
                for ln in STATE["bank"]:
                    m = re.search(r"demo showed: (?:stroke \d+: )?"
                                  r"(move [a-z][a-z-]*)", ln)
                    if m:
                        demo.append(m.group(1))
                ncomp = sum(1 for ln in STATE["bank"]
                            if "completed: move" in ln)
                # v18b: DETERMINISTIC PL ADVANCE (sole authority) -- the
                # writer completed every tick in the pl-10 run (ep0: 3
                # completions in 4 ticks, a2 ran off the route).  Arrival =
                # a strict-red blob at the held stroke's destination dot B:
                # both the stick's red tip band and the dot lighting up on
                # a successful press manifest as red pixels at B.
                _pst = STATE.get("pl_strokes") or []
                _adv = STATE.setdefault("pl_adv", 0)
                if _adv < len(_pst) and _pst[_adv]:
                    _br, _bc = _pst[_adv]["B"]        # (row, col)
                    import numpy as _np
                    from scipy import ndimage as _ndi
                    _hit = False
                    for _p in pils:
                        _a = _np.asarray(_p)
                        _m = ((_a[..., 0] > 150) & (_a[..., 1] < 70)
                              & (_a[..., 2] < 70))
                        _lab, _n = _ndi.label(
                            _ndi.binary_dilation(_m, iterations=2))
                        for _c in range(1, _n + 1):
                            _ys, _xs = _np.nonzero((_lab == _c) & _m)
                            if 6 <= len(_ys) <= 220 and \
                               (_ys.mean()-_br)**2 + \
                               (_xs.mean()-_bc)**2 <= 12**2:
                                _hit = True
                                break
                        if _hit:
                            break
                    if _hit:
                        STATE["pl_adv"] = _adv + 1
                        STATE["pl_advanced"] = \
                            STATE.get("pl_advanced", 0) + 1
                        STATE["pl_stall"] = 0
                STATE["pl_stall"] = STATE.get("pl_stall", 0) + 1
                if STATE["pl_stall"] > 6 and ncomp > STATE.get("pl_adv", 0):
                    STATE["pl_adv"] = STATE.get("pl_adv", 0) + 1
                    STATE["pl_stall"] = 0
                    STATE["pl_stall_adv"] = STATE.get("pl_stall_adv", 0) + 1
                ncomp = STATE.get("pl_adv", 0)
                if demo:
                    held = demo[min(ncomp, len(demo) - 1)]
                    if held.lower() != sub.strip().lower():
                        STATE["pl_held"] = STATE.get("pl_held", 0) + 1
                    sub = held
            # v19 SEQUENCE PIN: on video pick/drop sequence tasks the 0.8B
            # reasoner cites the WRONG DEMO STEP's coordinate with the right
            # verb (VPB trace ep1 t1: step-2 pick coord at j=0). Algorithm-1
            # "pin to the j-th recorded step", generalised from rs/pl:
            # expected coordinate = demo step j of the subgoal's verb class,
            # j = admitted completions of that class; CORRECT a citation that
            # sits on a DIFFERENT demo step. Runs before the stale-demo
            # guard, which then re-snaps demo coords to current detections.
            if (V19 and not HARNESS_OFF
                    and not os.environ.get("WAM_NO_CORRECT")):
                _dpk = [(int(a), int(b)) for ln in STATE["bank"]
                        if "demo showed" in ln and "pick" in ln
                        for a, b in re.findall(
                            r"<\s*(\d+)\s*,\s*(\d+)\s*>", ln)]
                _ddr = [(int(a), int(b)) for ln in STATE["bank"]
                        if "demo showed" in ln
                        and re.search(r"drop|place|put", ln)
                        for a, b in re.findall(
                            r"<\s*(\d+)\s*,\s*(\d+)\s*>", ln)]
                if len(_dpk) >= 2 and _ddr:
                    _jp = sum(1 for ln in STATE["bank"]
                              if "completed: pick" in ln)
                    _jd = sum(1 for ln in STATE["bank"]
                              if re.search(r"completed: (?:drop|place|put)",
                                           ln))
                    _sl = sub.lower()
                    _exp = None
                    if "pick" in _sl and _jp < len(_dpk):
                        _exp = _dpk[_jp]
                    elif re.search(r"drop|place|put", _sl) \
                            and _jd < len(_ddr):
                        _exp = _ddr[_jd]
                    _m0 = re.search(r"<\s*(\d+)\s*,\s*(\d+)\s*>", sub)
                    if _exp is not None and _m0:
                        _cx, _cy = int(_m0.group(1)), int(_m0.group(2))
                        _steps = [s for s in _dpk + _ddr if s != _exp]
                        _other = any(abs(_cx - u) + abs(_cy - v) <= 10
                                     for u, v in _steps)
                        # v21 VALID-CITATION GUARD (as for the container
                        # authority): a citation that IS a live detection of
                        # the claimed class stands; the pin repairs only
                        # citations matching nothing on the table.
                        _pcls = _lift_cls(_sl)
                        _live = any(_pcls in n and abs(_cx-x)+abs(_cy-y) <= 14
                                    for d in dets if d for n, x, y in d)
                        if abs(_cx - _exp[0]) + abs(_cy - _exp[1]) > 20 \
                                and _other and not _live:
                            sub = (sub[:_m0.start()]
                                   + f"<{_exp[0]}, {_exp[1]}>"
                                   + sub[_m0.end():])
                            STATE["seq_pinned"] = \
                                STATE.get("seq_pinned", 0) + 1
            # v15b (user): STALE-DEMO-COORDINATE GUARD.  In demo->exec reset
            # tasks, a2 sometimes copies demo VIDEO coordinates verbatim
            # (vpb 9->3: picks at literal demo-line coords on empty table).
            # An output coord that matches a demo-line coord (<=6px) and no
            # current detection (>20px from all) is a stale copy: re-snap it
            # to the nearest class-compatible current-frame detection.
            if any("demo showed" in ln for ln in STATE["bank"]) \
                    and not os.environ.get("WAM_NO_CORRECT"):
                _cur = dets[-1] or []
                # v21 PERSISTENCE GUARD: "no live detection" must hold in
                # EVERY frame of the current and previous window, not the
                # last frame alone -- a one-frame occlusion gap at tick 1
                # made the re-snap move CORRECT citations onto another cube
                # (VRP ep2/ep6: off-arm cited the true cube, on-arm rebound
                # it 80px away; v20 VRP 0/12 vs off 8/30).
                _allf = [d for d in list(STATE.get("prev_dets") or [])
                         + list(dets or []) if d]
                _dcs = [(int(a), int(b)) for ln in STATE["bank"]
                        if "demo showed" in ln
                        for a, b in re.findall(r"<\s*(\d+)\s*,\s*(\d+)\s*>", ln)]
                def _fix(m):
                    x, y = int(m.group(1)), int(m.group(2))
                    near_cur = min(((x-u)**2+(y-v)**2 for d in _allf
                                    for _, u, v in d), default=1e9)
                    near_demo = min(((x-u)**2+(y-v)**2 for u, v in _dcs),
                                    default=1e9)
                    if near_demo <= 36 and near_cur > 400 and _cur:
                        pre = sub[:m.start()].rsplit(" the ", 1)[-1]
                        cls = next((w for w in ("cube", "target", "button",
                                                "container", "bin", "peg")
                                    if w in pre), None)
                        cand = [(u, v) for n, u, v in _cur
                                if cls is None or cls in n]
                        if cand:
                            u, v = min(cand, key=lambda c: (c[0]-x)**2
                                       + (c[1]-y)**2)
                            STATE["stale_fixed"] = \
                                STATE.get("stale_fixed", 0) + 1
                            return f"<{u}, {v}>"
                    return m.group(0)
                sub = re.sub(r"<\s*(\d+)\s*,\s*(\d+)\s*>", _fix, sub)
            # v23 COLOUR-AWARE VALID CITATION (VPB/VPO, "place the <colour>
            # cube on the target ..."): the reasoner cites a pad or a cube of
            # another colour for "pick up the cube" (VPO: 13/30 first-tick
            # citations >14px off while the detector had the task cube in
            # 30/30). CORRECT to the nearest task-colour cube when no such
            # cube was EVER detected within 14px of the citation in this
            # execution phase and the class is stably detected elsewhere
            # (>=3 of the window frames). Trace simulation: VPO 203 fires,
            # 109 land on the oracle, 8 on already-correct ticks; VPB 909 /
            # 265 / 1; scoped to these two tasks (PickX would misfire).
            _il3 = STATE.get("instr", "").lower()
            _mcol = re.search(r"place the (red|green|blue) cube on the", _il3)
            if _mcol and V19 and not HARNESS_OFF \
                    and not os.environ.get("WAM_NO_CORRECT"):
                STATE.setdefault("cls_hist", []).extend([d for d in dets if d])
                _cls = f"{_mcol.group(1)}_cube"
                _mp3 = re.search(r"pick up the cube at <\s*(\d+)\s*,\s*(\d+)\s*>", sub)
                if _mp3:
                    _c = (int(_mp3.group(1)), int(_mp3.group(2)))
                    _seen = any(n == _cls and abs(x-_c[0]) + abs(y-_c[1]) <= 14
                                for d in STATE["cls_hist"] for n, x, y in d)
                    _cands = [(x, y) for d in dets if d for n, x, y in d if n == _cls]
                    if not _seen and len(_cands) >= 3:
                        _nb = min(_cands, key=lambda q: abs(q[0]-_c[0]) + abs(q[1]-_c[1]))
                        sub = sub[:_mp3.start()] + f"pick up the cube at <{_nb[0]}, {_nb[1]}>" + sub[_mp3.end():]
                        STATE["cite_fixed"] = STATE.get("cite_fixed", 0) + 1
                # v23 VPB PAD PIN: the demo reader's pad owns the place target
                _mpl = re.search(r"place the cube onto the correct target at <\s*(\d+)\s*,\s*(\d+)\s*>", sub)
                if _mpl and STATE.get("vpb_pad"):
                    _pd = STATE["vpb_pad"]
                    if abs(int(_mpl.group(1))-_pd[0]) + abs(int(_mpl.group(2))-_pd[1]) > 6:
                        sub = sub[:_mpl.start()] + f"place the cube onto the correct target at <{_pd[0]}, {_pd[1]}>" + sub[_mpl.end():]
                        STATE["vpb_pinned"] = STATE.get("vpb_pinned", 0) + 1
            # v24 PLAN-STEP PIN ON ADMITTED COMPLETIONS (user 2026-09-10: "subgoal
            # no change after pick-up complete identified and written", BinFill
            # and VPO). Anchor = the bank's EVIDENCE-GATED completions (the
            # verified progress index of Alg. 1) plus detections; never the
            # writer's schedule or a count of unverifiable lines.
            _il4 = STATE.get("instr", "").lower()
            _last_c = next((ln for ln in reversed(STATE["bank"]) if "completed:" in ln), "")
            _held = "completed: pick up" in _last_c      # cube in hand, no deposit/place admitted yet
            if V19 and not HARNESS_OFF and not os.environ.get("WAM_NO_CORRECT"):
                # (a) VPB/VPO: pick admitted but a2 re-issues the pick -> place at the reader's pad
                if _mcol and _held and STATE.get("vpb_pad") \
                        and sub.strip().lower().startswith("pick up the cube"):
                    _pd = STATE["vpb_pad"]
                    sub = f"place the cube onto the correct target at <{_pd[0]}, {_pd[1]}>"
                    STATE["plan_pinned"] = STATE.get("plan_pinned", 0) + 1
                # (b) BinFill: pick admitted -> the step is "put it into the bin" until the
                # deposit is admitted: by the writer's line, or auto-claimed once the picked
                # cube's colour has been absent from its origin for BF_PUT_DWELL windows.
                if "into the bin" in _il4:
                    _binc = next(((x, y) for d in reversed([d for d in dets if d])
                                  for n, x, y in d if n == "bin"), None)
                    if _binc is None:
                        _mb = re.search(r"bin at <\s*(\d+)\s*,\s*(\d+)\s*>", STATE["bank"][0] if STATE["bank"] else "")
                        _binc = (int(_mb.group(1)), int(_mb.group(2))) if _mb else None
                    _mo = re.search(r"completed: pick up the \w+ (red|green|blue) cube at <\s*(\d+)\s*,\s*(\d+)\s*>", _last_c)
                    if _held and _mo and _binc:
                        _colr, _org = _mo.group(1), (int(_mo.group(2)), int(_mo.group(3)))
                        _absent = all(not any(n == f"{_colr}_cube" and abs(x-_org[0]) + abs(y-_org[1]) <= 14
                                              for n, x, y in d) for d in dets if d)
                        if STATE.get("bf_origin") != (_colr, _org):
                            STATE["bf_origin"] = (_colr, _org)
                            STATE["bf_absent_run"] = 0
                        STATE["bf_absent_run"] = STATE.get("bf_absent_run", 0) + 1 if _absent else 0
                        if STATE["bf_absent_run"] >= BF_PUT_DWELL:
                            bank_append(step, f"[event] completed: put it into the bin at <{_binc[0]}, {_binc[1]}>"
                                        f"  [sam] bin<{_binc[0]}, {_binc[1]}>")
                            STATE["bf_put_auto"] = STATE.get("bf_put_auto", 0) + 1
                        elif not sub.strip().lower().startswith("put it into the bin"):
                            sub = f"put it into the bin at <{_binc[0]}, {_binc[1]}>"
                            STATE["plan_pinned"] = STATE.get("plan_pinned", 0) + 1
            # v17-serve (user): CONTAINER-COORDINATE AUTHORITY -- a2 misfolds
            # long swap chains (vus 10->9->6 on identical banks).  When it
            # asks to pick the container hiding cube C, the coordinate comes
            # from the deterministic fold of the writer's own chain.
            _mc = re.search(r"pick up the container at <\s*(\d+)\s*,\s*(\d+)\s*> "
                            r"that hides the (\w+) cube", sub)
            if _mc and any("container moved" in ln for ln in STATE["bank"]) \
                    and not os.environ.get("WAM_NO_CORRECT"):
                _st = CB.fold_state(STATE["bank"])
                _pos = _st.get(_mc.group(3))
                if _pos:
                    _ax, _ay = int(_mc.group(1)), int(_mc.group(2))
                    # v20 VALID-CITATION GUARD (paired-trace diagnosis, VU 6
                    # + VUS 5 harness-lost eps): the fold is only as good as
                    # the writer's chain, and at 0.8B a wrong move line made
                    # the authority overwrite CORRECT reasoner citations
                    # with the fold of a hallucinated chain. The authority
                    # now repairs only INVALID citations -- a coordinate
                    # matching no live container detection; a citation that
                    # IS a live container stands. (CORRECT was already null
                    # at 9B, so nothing is lost at scale.)
                    _live = any(
                        "container" in n and abs(_ax-x)+abs(_ay-y) <= 14
                        for d in dets if d for n, x, y in d)
                    if abs(_ax-_pos[0]) + abs(_ay-_pos[1]) > 24 \
                            and not _live:
                        sub = sub.replace(f"<{_ax}, {_ay}>",
                                          f"<{_pos[0]}, {_pos[1]}>", 1)
                        STATE["cstate_fixed"] = \
                            STATE.get("cstate_fixed", 0) + 1
            # v22 BINFILL STEP DISCIPLINE (user: "no need to track the cube's
            # position, just pick the right colour at the right time"). The
            # plan is a count template read off the instruction; admitted
            # (evidence-gated) pick completions give the true step. Corrects
            # a2's ORDINAL and colour to the next unfilled quota, cites any
            # live cube of that colour (position is irrelevant to the task),
            # and DEFERs a press proposed before the deposit quota is met.
            # 8/18 v21c failures were exactly this counting drift.
            _il2 = STATE.get("instr", "").lower()
            # 2026-09-10: DEFAULT OFF (WAM_BINFILL_DISCIPLINE=1 to enable). Measured
            # on 0.8B v22 laneW: 4/19 vs 12/30 without it -- the press deferral
            # waits for "completed: put it into the bin" lines the writer does not
            # produce (deposits are unobservable), and duplicate pick completions
            # drive the ordinal past the quota. Both anchors are model-written
            # state, not perception: the scale law says such corrections invert.
            if (V19 and not HARNESS_OFF and "into the bin" in _il2
                    and os.environ.get("WAM_BINFILL_DISCIPLINE") == "1"
                    and not os.environ.get("WAM_NO_CORRECT")):
                _num = {"a": 1, "one": 1, "two": 2, "three": 3, "four": 4,
                        "five": 5, "six": 6}
                _ordw = {1: "first", 2: "second", 3: "third", 4: "fourth",
                         5: "fifth", 6: "sixth"}
                _quota = {}
                for _w, _col in re.findall(r"(\w+) (red|green|blue) cube", _il2):
                    if _w in _num:
                        _quota[_col] = _quota.get(_col, 0) + _num[_w]
                if _quota:
                    _picks = {c_: 0 for c_ in _quota}
                    _puts = 0
                    for ln in STATE["bank"]:
                        _mp = re.search(r"completed: pick up the \w+ (red|green|blue) cube", ln)
                        if _mp and _mp.group(1) in _picks:
                            _picks[_mp.group(1)] += 1
                        if "completed: put it into the bin" in ln:
                            _puts += 1
                    _total = sum(_quota.values())
                    _live = [(n, x, y) for d in dets if d for n, x, y in d]
                    _mpk2 = re.search(r"pick up the (\w+) (red|green|blue) cube"
                                      r"(?: at <\s*(\d+)\s*,\s*(\d+)\s*>)?", sub)
                    if _mpk2:
                        _col = _mpk2.group(2)
                        if _picks.get(_col, 0) >= _quota.get(_col, 0):
                            _rem = [c_ for c_ in _quota if _picks[c_] < _quota[c_]]
                            _col = _rem[0] if _rem else _col
                        _k = _picks.get(_col, 0) + 1
                        _cx = (int(_mpk2.group(3)), int(_mpk2.group(4))) \
                            if _mpk2.group(3) else None
                        _cands = [(x, y) for n, x, y in _live if n == f"{_col}_cube"]
                        _ok = _cx is not None and any(
                            abs(_cx[0]-x)+abs(_cx[1]-y) <= 14 for x, y in _cands)
                        if _cands and not _ok:
                            _cx = min(_cands, key=lambda q: (q[0]-_cx[0])**2 + (q[1]-_cx[1])**2) \
                                if _cx else _cands[0]
                        _new = f"pick up the {_ordw.get(_k, str(_k)+'th')} {_col} cube"
                        if _cx:
                            _new += f" at <{_cx[0]}, {_cx[1]}>"
                        if _new != sub:
                            sub = _new
                            STATE["binfill_pinned"] = STATE.get("binfill_pinned", 0) + 1
                    elif "press the button" in sub.lower() and _puts < _total:
                        sub = STATE.get("expected") or sub
                        STATE["binfill_press_deferred"] = \
                            STATE.get("binfill_press_deferred", 0) + 1
            STATE["tick"] += 1
            log({"kind": "tick", "ep": STATE["ep"], "task": STATE["task"],
                 "step": step, "tick": STATE["tick"],
                 "dets": [[f"t{i}"] + ([f"{n}<{x}, {y}>" for n, x, y in d]
                                        if d is not None else ["(sam skipped)"])
                          for i, d in enumerate(dets)],
                 "expected_in": STATE["expected"],
                 "oracle": req.get("oracle", ""),
                 "agent1_out": line, "bank": bank_before, "agent2_out": sub})
            STATE["expected"] = sub
            self._send({"subgoal": sub, "wrote": line, "bank": len(STATE["bank"])})
        elif self.path == "/episode_end":
            log({"kind": "end", "ep": STATE["ep"], "task": STATE["task"],
                 "success": req.get("success", ""), "bank": list(STATE["bank"])})
            self._send({"ok": True})
        else:
            self._send({"error": "unknown"}, 404)


print(f"[wam] serving on {PORT}", flush=True)
ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()

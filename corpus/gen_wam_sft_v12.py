#!/usr/bin/env python
"""WAM v8 unified SFT generator — all 16 RoboMME tasks, both agents.

System (see wam_system_spec.md): both agents tick every 25 steps, agent-1
(writer) then agent-2 (foresight). Tick 0 writes the initial scene.

agent1 record: instruction + past bank(-8) + 5 window frames + PER-FRAME sam
  detections + expected-next-subgoal (agent-2's previous-tick GT)
  -> one [event]...[sam]... line for the window, or NONE.
agent2 record: instruction + full current bank -> expected next subgoal
  (annotated string with coords) | "hold and wait for the demo to complete"
  (demo phase) | "hold and observe the scene" (GT coord not yet in bank)
  | "all tasks completed; remain static".

Augmented annotation (verified): initial-scene line (detector, frame 0);
covering + container-move lines for vu/vus/bu/bus (offline tracking, oracle
QC); highlight-appearance lines for ph (white-blob detection, oracle-matched).
Val = episode_index % 10 == 0.
"""
import collections, glob, io, json, os, re, sys, time
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sam_registry as SR
import annot_pipeline as AP
import container_binding as CB

D = os.path.expandvars("${GAMMA_DATA}/data/robomme_lerobot")
OUT = os.environ.get("OUT_DIR",
                     os.path.expandvars("${GAMMA_DATA}/data/wam_sft_v8"))
# tick = 16 control steps: matches the pi0.5 action-chunk replan cadence,
# i.e. the same frequency at which RoboMME's oracle grounded-subgoal
# baseline refreshes (one oracle read per exec_horizon=16 chunk)
SNAP, FSTEP = 16, 4
# v14 (user): ONE window geometry everywhere -- 16 steps, 5 frames, demo and
# exec alike, matching the serve loop exactly.  ("Denser" was a frame-rate
# idea, not a tick-rate one, and is dropped.)
SNAP_DEMO = int(os.environ.get('SNAP_DEMO', '16'))
OVER_SWAP = int(os.environ.get('OVER_SWAP', '8'))
OVER_MULTI = int(os.environ.get('OVER_MULTI', '3'))
OVER_CROSS = int(os.environ.get('OVER_CROSS', '3'))
OVER_COMPLETE = int(os.environ.get('OVER_COMPLETE', '4'))
# v12 NOTE on button presses: a press registers inside the env with NO
# visual consequence (measured: button patch changes 6-10 RGB on
# registration, inside arm-shadow noise; the vus/bus container swap runs on
# a fixed 114/164/214 schedule and carries no press information).  The
# completion label is already the right rule -- stamped on the tick the
# online oracle switches -- and the writer reproduces it exactly in open
# loop (37/40 val press completions on the exact tick, 0 spurious).  It can
# do so because the GT registration step is near-constant (80 for the first
# press in 9/10 bus episodes, 176 for the second), i.e. it learns a clock,
# not a percept -- which is why it fires anyway in closed loop when the
# press did not land.  A retry augmentation was tried here and reverted: it
# required overwriting expected_next with a subgoal the planner never
# emitted at that tick, i.e. training on a fabricated input.  No press fix
# is applied in v12; the failure mode stands documented instead.
PRESS_FAMS = ("bu", "bus", "ph")
COORD = re.compile(r"<\s*(\d+)\s*,\s*(\d+)\s*>")
SKIP = ("static", "remain static", "all tasks completed", "no record", "")
TASK16 = {0: "pl", 1: "bus", 2: "bu", 3: "vpb", 4: "vus", 5: "pickx",
          6: "stopcube", 7: "swingx", 8: "ph", 9: "mc", 10: "ip", 11: "rs",
          12: "binfill", 13: "vpo", 14: "vrp", 15: "vu"}
HLRE = re.compile(r"(?:the\s+)?(first|second|third|fourth|fifth)?\s*"
                  r"highlighted cube at <\s*(\d+)\s*,\s*(\d+)\s*>"
                  r"(?:,\s*which is\s+(\w+))?")
os.makedirs(f"{OUT}/frames", exist_ok=True)
t0 = time.time()

tasks = {json.loads(l)["task_index"]: json.loads(l)["task"]
         for l in open(f"{D}/meta/tasks.jsonl")}


def obj_name(gs_):
    pre = gs_.split("at <")[0].strip()
    m = re.search(r"(?:the|a|another)\s+([\w\- ]+?)\s*$", pre)
    return (m.group(1) if m else "object").strip().replace(" ", "_")


files = sorted(glob.glob(f"{D}/data/chunk-*/*.parquet"))
_only = os.environ.get("ONLY_FILES", "")
if _only:
    files = [files[int(i)] for i in _only.split(",")]
_lim = int(os.environ.get("LIMIT", "0"))
if _lim:
    files = files[::max(1, len(files) // _lim)][:_lim]
print(f"[gen] {len(files)} parquet files", flush=True)

a1 = {s: open(f"{OUT}/agent1_{s}.jsonl", "w") for s in ("train", "val")}
a2 = {s: open(f"{OUT}/agent2_{s}.jsonl", "w") for s in ("train", "val")}
stats = collections.Counter()

for fi, f in enumerate(files):
    tbl = pq.read_table(f)
    cols = {c: tbl[c].to_pylist() for c in
            ("episode_index", "step_idx", "is_demo", "grounded_subgoal",
             "grounded_subgoal_online", "task_index")}
    imgs = tbl["image"].to_pylist()
    idxs = sorted(range(len(cols["step_idx"])),
                  key=lambda i: cols["step_idx"][i])
    e = int(cols["episode_index"][0])
    fam = TASK16.get(e // 100)
    if fam is None:
        continue
    instr = tasks[cols["task_index"][idxs[0]]]
    gs = [cols["grounded_subgoal"][i] or "" for i in idxs]
    # the ONLINE stream: the env's live subgoal, served verbatim to pi0.5 by
    # RoboMME's oracle grounded-symbolic baseline (0.84 SR) once per 16-step
    # chunk — switches at true completion moments and carries live
    # coordinates. This is agent-2's GT output stream.
    gso = [cols["grounded_subgoal_online"][i] or "" for i in idxs]
    dm = [bool(cols["is_demo"][i]) for i in idxs]
    T = len(idxs)
    split = "val" if e % 10 == 0 else "train"
    if os.environ.get("VAL_ONLY") and split != "val":
        continue
    get = lambda t: SR.decode_bgr(imgs[idxs[t]])

    # ---- per-frame detections: REAL detector (Grounding-DINO) cache — the
    # channel that is reproduced verbatim at eval time. HSV is used only
    # inside the offline authoring trackers (covering/moves/highlights QC).
    _dp = os.environ.get("DET_DIR", os.path.expandvars("${GAMMA_DATA}/data/wam_sft_v8/dets_sam3")) + f"/e{e}.json"
    if not os.path.exists(_dp):
        stats["ep_no_dets"] += 1
        continue
    _real = {int(k): [tuple(d) for d in v]
             for k, v in json.load(open(_dp)).items()}
    _grid = sorted(_real)

    def det_at(t):
        return _real[min(_grid, key=lambda g: abs(g - t))]

    def det_str(t):
        return " ".join(f"{n}<{x}, {y}>" for n, x, y in det_at(t)) or "(none)"

    # ---- events: annotation boundaries + augmented observation lines
    # Event boundaries follow the ONLINE stream: it switches at the true
    # completion moment (the visible one), while the recorded annotation
    # trails it by tens of control steps (arm retract / idle). Aligning
    # agent-1's writes to the online switch keeps the bank's completion
    # state in sync with agent-2's target, and makes the event coincide
    # with the change actually visible in the window's frames.
    # v15b (user): vpo/vpb keep the EXACT v13 configuration -- 8-step demo
    # training windows with the unchanged 16-step serve replay (that pairing
    # measured 9/10 on both; the matched-16 pairing regressed them to 3/10).
    SD = 8 if fam in ("vpo", "vpb") else SNAP_DEMO
    ALIGN = os.environ.get("EVENT_ALIGN", "online")
    src = gso if ALIGN == "online" else gs
    bounds = [b for b in range(1, T) if src[b] != src[b-1]]
    events = []          # (t, kind, payload) kind: EV(prefix,sg) | RAW(line)
    for b in bounds:
        sg = src[b-1]
        if sg.lower().strip() in SKIP:
            # the online stream emits "no record" at demo edges; fall back to
            # the recorded annotation so no real event is lost
            sg = gs[b-1]
            if sg.lower().strip() in SKIP:
                continue
        pre = "demo showed" if dm[b-1] else "completed"
        # v15 (user): completion registration back with AGENT-1 (v13 style):
        # press completions stamped one tick later (arm-retract percept).
        b_eff = b
        if pre == "completed" and "press the" in sg.lower():
            b_eff = min(b + SNAP, T - 1)
            stats["press_completion_delayed"] += 1
        events.append((b_eff, "EV", (pre, sg)))
    init_line = ("[event] initial scene: "
                 + (", ".join(f"{n.replace('_', ' ')} at <{x}, {y}>"
                              for n, x, y in det_at(0)) or "empty table")
                 + "  [sam] "
                 + (" ".join(f"{n}<{x}, {y}>" for n, x, y in det_at(0))
                    or "(none)"))
    if fam in ("vu", "vus", "bu", "bus"):
        # Reveal identity + dynamics from the STANDARD PIPELINE
        # (annot_pipeline.py): sites -> occupancy -> cover/transit primitives
        # -> identity propagation, stamped on the frames where the motion is
        # visible.  Verified at G3 = 0.955 binding accuracy, against 0.43 for
        # the heuristic tracker it replaces.
        pev, psites, pat, pbind = AP.build_events(_real, _grid)
        # per-frame bound container positions (real det coords), labels only
        _bound, _cmoves = CB.replay(pev, _real, _grid)
        # v12: after every swap, close the exchange with ONE state-summary line.
        # v11 left agent-2 to resolve a chain of move lines: 1.000 action+object
        # but only 0.491 coordinate accuracy on vus execution ticks.
        # GATED (first cut measured 0.75 with 8/20 ALIASED states -- summaries
        # emitted mid-exchange put two cubes under one container):
        #  - intermediate summaries require a VALID state: pairwise-distinct
        #    coordinates (>=18px) each sitting on a real container detection;
        #  - the FINAL summary (stamped at the last swap) must AGREE with every
        #    binding the oracle's own unmask subgoals name, else it is withheld
        #    (an absent line beats a wrong one).
        _truth = {}
        for _sg in gso:
            _m = re.search(r"pick up the container at <(\d+), (\d+)> that "
                           r"hides the (\w+) cube", _sg or "")
            if _m:
                _truth.setdefault(_m.group(3),
                                  (int(_m.group(1)), int(_m.group(2))))
        _state = {}
        _last_move_t = max([t_ for t_, k_, p_ in pev if k_ == "MOVE_END"],
                           default=None)
        # v14 (user): ONE line per window per moving container -- "moved from
        # <pos in the window's first frame> to <pos in its last frame>", both
        # endpoints real SAM det positions (CB tracks bound containers to
        # detections), so the label is verifiable inside the window's own
        # frames.  No "is being moved" chatter lines.
        def _win_end_of(t):
            w_ = SD if dm[min(max(t - 1, 0), T - 1)] else SNAP
            return max(w_, ((t + w_ - 1) // w_) * w_)
        for _we in sorted({_win_end_of(g) for g in _grid}):
            _w = SD if dm[min(_we - 1, T - 1)] else SNAP
            _fs = [g for g in _grid if _we - _w <= g <= _we]
            if len(_fs) < 2:
                continue
            _b0, _b1 = _bound.get(_fs[0], {}), _bound.get(_fs[-1], {})
            for _c in sorted(set(_b0) & set(_b1)):
                p0, p1 = _b0[_c], _b1[_c]
                if (p0[0]-p1[0])**2 + (p0[1]-p1[1])**2 >= 6**2:
                    events.append((_we, "RAW",
                        f"[event] observed: the {_c} cube container moved "
                        f"from <{p0[0]}, {p0[1]}> to <{p1[0]}, {p1[1]}>  "
                        f"[sam] {_c}_cube_container<{p1[0]}, {p1[1]}>"))
                    stats["pipe_move"] += 1
        def _valid(st, t_):
            xs = list(st.values())
            if len(set(xs)) < len(xs):
                return False
            for i in range(len(xs)):
                for j in range(i+1, len(xs)):
                    if (xs[i][0]-xs[j][0])**2 + (xs[i][1]-xs[j][1])**2 < 18**2:
                        return False
            dets_ = [(x, y) for n_, x, y in det_at(t_) if n_ == "container"]
            return all(any((c[0]-u)**2 + (c[1]-v)**2 <= 16**2 for u, v in dets_)
                       for c in xs)
        for t_, kind, payload in pev:
            if kind == "COVER":
                desc = ", ".join(f"the {c} cube at <{xy[0]}, {xy[1]}>"
                                 for c, (xy, s) in sorted(payload.items()))
                sam = " ".join(f"{c}_cube_container<{s[0]}, {s[1]}>"
                               for c, (xy, s) in sorted(payload.items()))
                events.append((t_, "RAW",
                               f"[event] observed: containers were placed over "
                               f"the cubes — {desc}  [sam] {sam}"))
                stats["pipe_cover"] += 1

        for t_, kind, payload in pev:
            if kind == "COVER":
                for c, (xy, s_) in payload.items():
                    _state[c] = tuple(s_)
            elif kind == "MOVE_END":
                _state[payload[0]] = tuple(payload[1])
                final = (t_ == _last_move_t)
                if final:
                    agree = all(
                        c in _state and
                        (_state[c][0]-xy[0])**2 + (_state[c][1]-xy[1])**2 <= 16**2
                        for c, xy in _truth.items())
                    if not (agree and _valid(_state, t_)):
                        stats["swap_summary_withheld"] += 1
                        continue
                elif not _valid(_state, t_):
                    stats["swap_summary_invalid_skip"] += 1
                    continue
                desc = ", ".join(
                    f"the {c} cube is under the container at "
                    f"<{p_[0]}, {p_[1]}>" for c, p_ in sorted(_state.items()))
                sam = " ".join(f"{c}_cube_container<{p_[0]}, {p_[1]}>"
                               for c, p_ in sorted(_state.items()))
                events.append((_win_end_of(t_), "RAW",
                               f"[event] observed: now {desc}  [sam] {sam}"))
                stats["swap_summary"] += 1
        stats["reveal_tracked" if pev else "reveal_untracked"] += 1
    if any(dm):
        # demo->exec reset: objects move between demo and execution; author
        # an execution-scene line from the first execution frames
        t_ex = max(t for t in range(T) if dm[t]) + 1
        if t_ex + 2 < T:
            dts = det_at(min(t_ex + 2, T-1))
            if dts:
                events.append((t_ex + 2, "RAW",
                               "[event] execution phase begins — scene: "
                               + ", ".join(f"{n.replace('_',' ')} at <{x}, {y}>"
                                           for n, x, y in dts if n != "arm")
                               + "  [sam] "
                               + " ".join(f"{n}<{x}, {y}>" for n, x, y in dts
                                          if n != "arm")))
                stats["exec_scene"] += 1
    if fam == "binfill":
        # cubes spawn over time; author an appearance line per new stable
        # cube cluster (skips ones near the arm = being carried)
        grid = list(range(0, T, FSTEP))
        clusters = [(x, y) for n, x, y in det_at(0) if "cube" in n]
        for gi, t in enumerate(grid[1:-1], 1):
            arm = [(x, y) for n, x, y in det_at(t) if n == "arm"]
            for n, x, y in det_at(t):
                if "cube" not in n:
                    continue
                if any((x-a)**2 + (y-b)**2 < 20**2 for a, b in clusters):
                    continue
                if any((x-a)**2 + (y-b)**2 < 30**2 for a, b in arm):
                    continue
                nxt = det_at(grid[gi + 1])
                if not any(m == n and (x-a)**2 + (y-b)**2 < 12**2
                           for m, a, b in nxt):
                    continue
                clusters.append((x, y))
                colr = n.replace("_cube", "")
                events.append((t, "RAW",
                               f"[event] observed: a new {colr} cube appeared "
                               f"at <{x}, {y}>  [sam] {n}<{x}, {y}>"))
                stats["spawn_lines"] += 1
    if fam == "stopcube":
        # the count this task turns on was absent from memory entirely
        tgt = None
        for t in range(0, min(T, 60), 4):
            for n, x, y in det_at(t):
                if n == "target":
                    tgt = (x, y)
                    break
            if tgt:
                break
        if tgt:
            far, nre, prevd, warned = True, 0, None, False
            for t in range(0, T, 4):
                cu = [(n, x, y) for n, x, y in det_at(t) if "cube" in n]
                if not cu:
                    continue
                n, x, y = cu[0]
                d = ((x-tgt[0])**2 + (y-tgt[1])**2) ** 0.5
                # v12: the cube's APPROACH is what the press has to anticipate.
                # v11 recorded only the crossing, so agent-2 pressed one tick
                # after the cube had already left (all 7 closed-loop failures
                # pressed at t4 while the oracle still said "remain static").
                if far and not warned and prevd is not None and 18 < d <= 45 \
                        and d < prevd - 1:
                    warned = True
                    events.append((t, "RAW",
                                   f"[event] observed: the cube at <{x}, {y}> is "
                                   f"approaching the target at <{tgt[0]}, {tgt[1]}>  "
                                   f"[sam] {n}<{x}, {y}> target<{tgt[0]}, {tgt[1]}>"))
                    stats["stopcube_approach"] += 1
                if d <= 18 and far:
                    nre += 1
                    far = False
                    warned = False
                    events.append((t, "RAW",
                                   f"[event] observed: the cube reached the target at "
                                   f"<{tgt[0]}, {tgt[1]}> (reach #{nre})  "
                                   f"[sam] target<{tgt[0]}, {tgt[1]}>"))
                    stats["stopcube_reach"] += 1
                elif d > 26:
                    if not far:
                        events.append((t, "RAW",
                                       f"[event] observed: the cube left the target at "
                                       f"<{tgt[0]}, {tgt[1]}> and is moving away "
                                       f"(after reach #{nre})  "
                                       f"[sam] {n}<{x}, {y}>"))
                        stats["stopcube_depart"] += 1
                    far = True
                prevd = d
    if fam == "vrp":
        # v12 (user diagnosis): vrp's demo contains a SHUFFLE -- cubes visibly
        # exchange positions late in the video -- and the "correct cube" is the
        # demo-picked cube tracked THROUGH that shuffle.  v11 recorded nothing
        # of it, so with three identical red cubes the identity was
        # unrecoverable from the bank (6/8 closed-loop failures named the
        # wrong cube).  Nearest-neighbour tracking at the cache stride lands
        # the picked cube on the oracle's exec target in 40/40 GT episodes
        # (0 lost tracks), so the bank now records the shuffle hops and closes
        # with one copyable state line, mirroring the vus swap summary.
        _pick = _dropt = None
        for b in range(1, T):
            if dm[b-1] and gso[b] != gso[b-1]:
                _m = COORD.search(gso[b-1] or "")
                if 'pick up' in (gso[b-1] or '').lower() and _m:
                    _pick = (int(_m.group(1)), int(_m.group(2)))
                if 'drop' in (gso[b-1] or '').lower() and _pick and not _dropt:
                    _dropt = b
        if _pick and _dropt and any(dm):
            _dend = max(t for t in range(T) if dm[t])
            def _cubes(t):
                return [(x, y) for n_, x, y in det_at(t) if 'cube' in n_]
            _cur = min(_cubes(_dropt) or [_pick],
                       key=lambda c: (c[0]-_pick[0])**2 + (c[1]-_pick[1])**2)
            # v14 (user): same target shape as bus/vus -- one grounded line
            # per window the tracked cube moved ("moved from <p_first> to
            # <p_last>", both real det positions), then the identity summary
            # AFTER the moves.
            _tr = {}
            for t in range(_dropt, _dend + 4, 4):
                _cs = _cubes(t)
                if _cs:
                    _cur = min(_cs, key=lambda c: (c[0]-_cur[0])**2
                               + (c[1]-_cur[1])**2)
                _tr[t] = _cur
            _wend = None
            for _we in range(SD, _dend + SD, SD):
                _fs = [t for t in _tr if _we - SD <= t <= _we]
                if len(_fs) < 2:
                    continue
                p0, p1 = _tr[min(_fs)], _tr[max(_fs)]
                if (p0[0]-p1[0])**2 + (p0[1]-p1[1])**2 >= 6**2:
                    events.append((min(_we, _dend), "RAWNS",
                                   f"[event] observed: the correct cube moved "
                                   f"from <{p0[0]}, {p0[1]}> to <{p1[0]}, {p1[1]}>  "
                                   f"[sam] correct_cube<{p1[0]}, {p1[1]}>"))
                    stats["vrp_shuffle_move"] += 1
                    _wend = min(_we, _dend)
            events.append((_wend if _wend is not None else _dend, "RAWNS",
                           f"[event] observed: the cube picked in the video is now "
                           f"at <{_cur[0]}, {_cur[1]}> — this is the correct cube  "
                           f"[sam] cube<{_cur[0]}, {_cur[1]}>"))
            stats["vrp_track_summary"] += 1
    _stroke_g = {}
    _stroke_arc = {}
    _seen_strokes = []
    if fam in ("rs", "pl"):
        # v14 (user): the rs demo DRAWS A WHITE ARC on the table.  An HSV
        # white-mask diff between consecutive stride-4 frames gives the arc's
        # growing TIP; the tip's stick-centered angular sweep matches the
        # annotated cw/ccw at 17/18 = 0.94 when |sweep| >= 1.2 rad.  Tips are
        # injected as 'arc_tip' input detections at window-INTERIOR frames
        # only (f%16 in 4,8,12), so the serve loop can recompute them
        # statelessly from the same within-window diffs.
        def _wmask(t):
            a = get(t).astype(np.int16)
            mx = a.max(axis=2); mn = a.min(axis=2)
            m = (mx > 190) & ((mx - mn) < 28)
            m[:40, :] = False
            return m
        _pm = None; _ptt = None; _tips = {}
        _dend_a = max((tt for tt in range(T) if dm[tt]), default=-1)
        for _f in [g_ for g_ in _grid if g_ <= _dend_a]:
            _m = _wmask(_f)
            if _pm is not None and _f - _ptt <= 8:
                _new = _m & ~_pm
                _ys, _xs = np.nonzero(_new)
                # 3..200px: trail segments are thin; larger diffs are the
                # demo gripper's own body moving (verified visually: it was
                # polluting endpoints and the 'above'-region side stats)
                _keep = _ys > 65   # gripper works rows<65; route plane below
                if 3 <= _keep.sum() <= 200:
                    _tips[_f] = (int(_ys[_keep].mean()),
                                 int(_xs[_keep].mean()))
            _pm, _ptt = _m, _f
        for _f, (_tx, _ty) in _tips.items():
            if _f % 16 in (4, 8, 12):
                _real[_f] = list(_real[_f]) + [("arc_tip", _tx, _ty)]
                stats["arc_tip_injected"] += 1
        _traj_pts = []
        # v14 (user): TRACK THE ARC TRAJECTORY IN THE BANK -- one observed
        # line per window while the arc grows; at stroke end the demo line
        # summarizes and judges cw/ccw from the recorded trajectory.
        for _we in range(16, _dend_a + 16, 16):
            _wt = [(f, _tips[f]) for f in sorted(_tips) if _we - 16 < f <= _we]
            if len(_wt) >= 2:
                _p0, _p1 = _wt[0][1], _wt[-1][1]
                if (_p0[0]-_p1[0])**2 + (_p0[1]-_p1[1])**2 >= 4**2:
                    events.append((_wt[-1][0], "RAWNS",
                        f"[event] observed: the arc tip moved from "
                        f"<{_p0[0]}, {_p0[1]}> to <{_p1[0]}, {_p1[1]}>  "
                        f"[sam] arc_tip<{_p1[0]}, {_p1[1]}>"))
                    _traj_pts.append((_wt[0][0] if _wt else 0, _p0))
                    _traj_pts.append((_wt[-1][0], _p1))
                    stats["arc_traj_line"] += 1
    if fam in ("pl", "rs"):
        # Lateral direction is in the arm channel, but NOT inside one window:
        # over 460 pl moves the column displacement gives left-vs-right at 0.91
        # across a whole stroke and only 0.55 across the 8-step window the
        # writer sees (median |dcol| 6px).  So anchor the stroke start in the
        # bank, then state the lateral read at the stroke end -- both halves are
        # then present in the writer's own inputs (bank + current window).
        # Only the lateral half is claimed: fore/aft is +-3..11px and lands at
        # 0.62.  The read is emitted only when it AGREES with the annotation, so
        # a label can never contradict the measurement.
        _t = 0
        while _t < T:
            if not dm[min(_t, T-1)] or "move " not in (gso[_t] or "").lower():
                _t += 1
                continue
            _s = _t
            while _t < T and gso[_t] == gso[_s]:
                _t += 1
            _end = min(_t, T - 1)
            if (fam == "rs" and "circling" in (gso[_s] or "").lower()) \
                    or fam == "pl":
                _stroke_arc.setdefault((gso[_s], _t), None)
            def _snapdet(_p, _s0=_s, _e0=_end):
                # v17 (user): every chain coordinate must be FINDABLE in the
                # writer's inputs.  Candidates are limited to provably
                # visible coords: interior arc_tips within the stroke (they
                # enter the bank via trajectory lines before the demo line)
                # and the emission window's own detections (in frame_dets).
                _cand = [tp for _f, tp in (_traj_pts if fam == "rs" or
                          fam == "pl" else [])
                         if _s0 - 8 <= _f <= _e0 + 8]
                _cand += [(_x, _y) for _tt in range(max(_s0, _e0-16), _e0+1, 4)
                          for _n, _x, _y in det_at(_tt)]
                _best = None
                for (_x, _y) in _cand:
                    _d = (_x-_p[0])**2 + (_y-_p[1])**2
                    if _d <= 144 and (_best is None or _d < _best[0]):
                        _best = (_d, (_x, _y))
                return _best[1] if _best else None
            _dend_local = max((tt for tt in range(T) if dm[tt]), default=-1)
            if _dend_local >= 0:
                _end = min(_end, _dend_local)   # never cross the demo->exec reset
            if _end <= _s:
                continue
            _a0 = [(x, y) for n, x, y in det_at(_s) if n == "arm"]
            _a1 = [(x, y) for n, x, y in det_at(_end) if n == "arm"]
            if not _a0 or not _a1:
                continue
            events.append((_s, "RAWNS",
                           f"[event] observed: the arm is at <{_a0[0][0]}, {_a0[0][1]}> "
                           f"and begins a new stroke  "
                           f"[sam] arm<{_a0[0][0]}, {_a0[0][1]}>"))
            stats["stroke_anchor"] += 1
            _w = re.search(r"(?:move |nearest )(?:\w+-)?(left|right)\b", gso[_s] or "")
            _dc = _a1[0][1] - _a0[0][1]
            _meas = "left" if _dc > 12 else ("right" if _dc < -12 else None)
            # v14 (user): SUPERVISION LINK -- instead of a separate lateral
            # line, the demo direction line itself carries the stroke's arm
            # endpoints ("(arm from <p0> to <p1>)"), so the direction word is
            # verifiable arithmetic over coordinates copied from the input.
            # Attached only when the measured lateral AGREES with the word
            # (0.91 pl / 27:27 rs when measured), never contradicting it.
            if _w and _meas == _w.group(1):
                _stroke_g[(gso[_s], _t)] = (_a0[0], _a1[0])
                stats["stroke_lateral_ok"] += 1
            elif _w:
                stats["stroke_lateral_unconfirmed"] += 1
            if fam == "pl":
                # v17 (user): pl REASONING CHAIN -- grounded endpoints and the
                # measured lateral phrase next to the annotation's word, so
                # the model learns to justify direction from displacement
                # (fore/aft mapping left to the network; lateral is gated).
                _tp = [(_f, _tips[_f]) for _f in sorted(_tips)
                       if _s <= _f <= _end]
                if len(_tp) >= 2:
                    _A2, _B2 = _tp[0][1], _tp[-1][1]
                    _dc2 = _B2[1] - _A2[1]
                    _m2 = ("left" if _dc2 > 4 else
                           ("right" if _dc2 < -4 else None))
                    _wl = _w.group(1) if _w else None
                    _word = (gso[_s] or "").lower().replace("move ", "").strip()
                    if _wl is None or _m2 == _wl:
                        _latp = (f"toward the {_m2}" if _m2
                                 else "straight along the pattern")
                        _As2, _Bs2 = _snapdet(_A2), _snapdet(_B2)
                        if _As2 and _Bs2:
                            _stroke_arc[(gso[_s], _t)] = (
                                _As2, _Bs2, f"{_latp}, move {_word}")
                            stats["arc_pl_grounded"] += 1
                        else:
                            stats["arc_chain_unsnappable"] += 1
                    else:
                        stats["arc_pl_unconfirmed"] += 1
            if fam == "rs" and "circling" in (gso[_s] or "").lower():
                _stk = [(x, y) for tt in range(_s, _end + 1, 4)
                        for n, x, y in det_at(tt) if "stick" in n]
                _tp = [(_f, _tips[_f]) for _f in sorted(_tips)
                       if _s <= _f <= _end]
                if _stk and len(_tp) >= 3:
                    _cx = sum(p[0] for p in _stk) / len(_stk)
                    _cy = sum(p[1] for p in _stk) / len(_stk)
                    _ag = [np.arctan2(p[1][1]-_cy, p[1][0]-_cx) for p in _tp]
                    _sw = 0.0
                    for _u, _v in zip(_ag, _ag[1:]):
                        _d = _v - _u
                        while _d > np.pi: _d -= 2*np.pi
                        while _d < -np.pi: _d += 2*np.pi
                        _sw += _d
                    _wcw = "counterclockwise" not in gso[_s].lower()
                    if abs(_sw) >= 1.2 and _wcw == (_sw > 0):
                        # v17 (user): REASONING CHAIN, vus-style -- numbered
                        # stroke, grounded endpoints, chord direction, arc
                        # side (derived from the validated sweep geometry),
                        # then the conclusion.  Every element observable in
                        # BOTH demo renderings (trail video / re-enactment).
                        _A, _B = _tp[0][1], _tp[-1][1]
                        _lat = ("left to right" if _B[1] - _A[1] < -8 else
                                ("right to left" if _B[1] - _A[1] > 8
                                 else None))
                        # side follows BIJECTIVELY from (chord, word) by the
                        # user's rule: right-to-left below the stick is cw,
                        # left-to-right above is cw -- guaranteeing every
                        # chain is internally coherent geometry (the noisy
                        # mid-angle estimate produced contradictions).
                        if _lat is not None:
                            _side = ("above" if (_lat == "left to right")
                                     == _wcw else "below")
                        else:
                            _side = None
                        _dirw = "clockwise" if _wcw else "counterclockwise"
                        _geo = (f"{_lat}, arc {_side} the stick, "
                                if _lat else "")
                        _As, _Bs = _snapdet(_A), _snapdet(_B)
                        if _As and _Bs:
                            _stroke_arc[(gso[_s], _t)] = (
                                _As, _Bs, f"{_geo}circling {_dirw}: {gso[_s]}")
                            stats["arc_cw_grounded"] += 1
                        else:
                            stats["arc_chain_unsnappable"] += 1
                    else:
                        _stroke_arc[(gso[_s], _t)] = None
                        stats["arc_cw_unconfirmed"] += 1
    if fam == "swingx":
        # v12: agent-2 counts repetitions by reading the bank.  v11 wrote no
        # event for a completed swing, so a missed completion froze the counter
        # one behind and agent-2 re-issued the same side (3 closed-loop
        # failures).  Stamp every target-top crossing with its running count,
        # per side, using the same hysteresis as the stopcube reach detector.
        tg = []
        for t in range(0, min(T, 60), 4):
            tg = [(x, y) for n, x, y in det_at(t) if n == "target"]
            if len(tg) >= 2:
                break
        if len(tg) >= 2:
            tg = sorted(tg, key=lambda p: p[1])          # by column: top/bottom
            side = {0: "right-side", 1: "left-side"}
            cnt = [0, 0]
            far = [True, True]
            for t in range(0, T, 4):
                cu = [(n, x, y) for n, x, y in det_at(t) if "cube" in n]
                if not cu:
                    continue
                n, x, y = cu[0]
                for k, (tx, ty) in enumerate(tg[:2]):
                    d = ((x-tx)**2 + (y-ty)**2) ** 0.5
                    if d <= 20 and far[k]:
                        cnt[k] += 1
                        far[k] = False
                        ordn = ("first", "second", "third", "fourth", "fifth",
                                "sixth")[min(cnt[k]-1, 5)]
                        events.append((t, "RAW",
                                       f"[event] observed: the cube reached the top of "
                                       f"the {side[k]} target at <{tx}, {ty}> for the "
                                       f"{ordn} time  [sam] target<{tx}, {ty}>"))
                        stats["swingx_cross"] += 1
                    elif d > 30:
                        far[k] = True
    if fam == "ph":
        ohl = []
        for s in sorted(set(gs)):
            m = HLRE.search(s)
            if m and (m.group(1), (int(m.group(2)), int(m.group(3)))) not in \
                    [(o[0], o[2]) for o in ohl]:
                ohl.append((m.group(1), m.group(4),
                            (int(m.group(2)), int(m.group(3)))))
        hl = SR.track_highlights(get, gs, ohl)
        stats["ph_lines"] += len(hl)
        for k_, v_ in getattr(SR, "HL_STATS", {}).items():
            stats[f"hl_{k_}"] += v_
        events.extend((t_, "RAW", ln) for t_, ln in hl)
    # DERIVABILITY: an authored (RAW) line must only contain coordinates the
    # writer can actually read off its own inputs, so snap each to the nearest
    # detection at that moment (<=15px). Annotation-derived lines are left
    # alone: their coordinates are the oracle's and must stay intact for
    # agent-2's knowability.
    def _win_frames(t_):
        """The 5 frames agent-1 will actually see for the window that this
        event falls into (last < t_ <= end, frames at stride FSTEP).

        v12: demo-phase windows are SNAP_DEMO long, not SNAP -- using the
        16-step grid there picked frames the writer never sees, which showed
        up as 6 pl motion traces failing the derivability audit."""
        _w = SD if (t_ < T and dm[min(int(t_), T - 1)]) else SNAP
        end = min(int(np.ceil(max(t_, 1) / _w)) * _w, T - 1)
        last = max(end - _w, 0)
        ts = list(range(last, end + 1, FSTEP))
        if ts[-1] != end:
            ts.append(end)
        return ts

    def snap_line(ln, t_):
        dets = [(x, y) for tt in _win_frames(t_)
                for n, x, y in det_at(tt) if n != "arm"]
        if not dets:
            return ln

        def rep(m):
            x, y = int(m.group(1)), int(m.group(2))
            bx, by = min(dets, key=lambda c: (c[0]-x)**2 + (c[1]-y)**2)
            if (bx-x)**2 + (by-y)**2 <= 25**2:
                stats["snapped"] += (bx, by) != (x, y)
                return f"<{bx}, {by}>"
            stats["snap_failed"] += 1
            return m.group(0)
        return COORD.sub(rep, ln)

    events = [(t_, k, (snap_line(p, t_) if k == "RAW" else p))
              for t_, k, p in events]   # RAWNS keeps its arm coords verbatim
    events.sort(key=lambda x: x[0])

    # ---- agent-2 GT + bank-knowability
    bank = []
    bank_xy = []

    def push(step, line):
        # v13 (user fix): NO absolute step prefix -- training registrations sit
        # at near-constant steps, so printed step numbers hand the model a
        # clock shortcut for completion timing.  Order encodes sequence.
        bank.append(line)
        bank_xy.extend((int(a), int(b)) for a, b in COORD.findall(line))

    def a2_gt(tq):
        if dm[min(tq, T-1)]:
            return "hold and wait for the demo to complete"
        sg = gso[min(tq, T-1)]
        # v12 FIX: the oracle stream's waiting strings are REAL targets that
        # pi0.5 consumes verbatim ("remain static" covers 349/390 steps of a
        # stopcube episode).  v11 treated them as skip-tokens and substituted
        # the NEXT subgoal, so agent-2 was trained to press immediately --
        # zero "remain static" targets existed in the stopcube training data,
        # which is the whole 7/7 closed-loop premature-press failure.  Only
        # "no record" (a demo-edge glitch token) and empty keep the fallback.
        if sg.lower().strip() in ("static", "remain static",
                                   "all tasks completed"):
            # stopcube anticipation (user-approved v14): the oracle's press
            # onset is look-ahead-timed, so a model that flips on observable
            # evidence is one tick late and the press misses the crossing
            # (v13 closed loop 1/10 with otherwise perfect holds).  Issue the
            # press subgoal one tick BEFORE the oracle flip: at the last
            # waiting tick before the flip, the target becomes the press.
            if fam == "stopcube":
                nxt = next((gso[t] for t in range(tq + 1, min(tq + 17, T))
                            if "press the button to stop" in gso[t].lower()), None)
                if nxt:
                    stats["stopcube_press_advanced"] += 1
                    return nxt
            stats["a2_static_kept"] += 1
            return sg
        # v15: press-delay gap hold restored (v13): if the previous subgoal
        # was a press whose delayed completion is not yet in the bank, keep
        # pressing -- agent-2 must never advance without the completion line.
        if tq > 0:
            u = min(tq, T - 1)
            v = u
            while v > 0 and gso[v - 1] == gso[u]:
                v -= 1
            prev_sg = gso[v - 1] if v > 0 else None
            if prev_sg and prev_sg != gso[u] and "press the" in prev_sg.lower() \
                    and u - v <= 31 \
                    and not any("completed: press" in ln and
                                prev_sg.split(" at <")[0].lower() in ln.lower()
                                for ln in bank):
                stats["a2_press_gap_held"] += 1
                return prev_sg
        if sg.lower().strip() in SKIP:
            sg = next((gso[t] for t in range(tq + 1, T)
                       if gso[t].lower().strip() not in SKIP), None)
            if sg is None:
                return "all tasks completed; remain static"
        m = COORD.search(sg)
        if m:
            gx, gy = int(m.group(1)), int(m.group(2))
            if not any((gx-a)**2 + (gy-b)**2 <= 30**2 for a, b in bank_xy):
                # coordinate not yet observable (undetectable object class):
                # keep the action, drop the coordinate
                stats["a2_coord_stripped"] += 1
                return re.sub(r"\s*at <\s*\d+\s*,\s*\d+\s*>", "", sg)
        return sg

    def save_frames(ts):
        out = []
        for t in ts:
            pth = f"frames/e{e}_t{t}.jpg"
            fp = f"{OUT}/{pth}"
            if not os.path.exists(fp):
                raw = imgs[idxs[t]]
                raw = raw["bytes"] if isinstance(raw, dict) else raw
                Image.open(io.BytesIO(raw)).convert("RGB").save(fp, quality=87)
            out.append(pth)
        return out

    def wr(fh, rec):
        fh.write(json.dumps(rec) + "\n")

    # ---- tick grids: offset 0 is the eval-time grid; extra offsets
    # (frame-aligned with the stride-4 detection cache) are training-only
    # augmentation toward policy-training sample density
    OFFS = [int(x) for x in os.environ.get("OFFSETS", "0").split(",")]
    if split == "val":
        OFFS = [0]          # val stays canonical (eval-time grid only)
    for OFF in OFFS:
      bank.clear()
      bank_xy.clear()
      # tick 0 (bank always starts with the initial scene; record written
      # once, on the canonical grid)
      if OFF == 0:
        wr(a1[split], dict(instruction=instr, family=fam, ep=e, span=[0, 0],
                           phase=("demo" if dm[0] else "execution"),
                           frames=save_frames([0]),
                           frame_dets=[f"+0: {det_str(0)}"], bank=[],
                           expected_next="(none yet)", target=init_line,
                           offset=0))
        stats["a1"] += 1
      push(0, init_line)
      if OFF == 0:
        wr(a2[split], dict(instruction=instr, family=fam, ep=e, tq=0,
                           phase=("demo" if dm[0] else "execution"),
                           bank=list(bank), target=a2_gt(0), offset=0,
                           frames=save_frames([0]),
                           frame_det=f"now: {det_str(0)}"))
        stats["a2"] += 1

      prev_obj = [None, None]
      last = 0
      dend = (max(t for t in range(T) if dm[t]) + 1) if any(dm) else 0
      ends = ([OFF] if OFF > 0 else [])
      ends += [t for t in range(OFF + SD, min(dend, T), SD)]
      ends += list(range(max(OFF + SNAP, (dend // SNAP + 1) * SNAP if dend else OFF + SNAP), T, SNAP))
      ends = sorted(set(ends))
      if not ends or ends[-1] != T - 1:
          ends.append(T - 1)
      for end in ends:
        # 5 frames per 16-step window: both boundaries included (t, t+4, ...,
        # t+16); the boundary frame overlaps the previous window by design
        ts = list(range(last, end + 1, FSTEP)) or [end]
        if ts[-1] != end:
            ts.append(end)
        win_cl = {}
        for t in ts:
            for n, x, y in det_at(t):
                if n == "arm":
                    continue
                cl = win_cl.setdefault(n, [])
                for c in cl:
                    if abs(c[0]-x) < 18 and abs(c[1]-y) < 18:
                        c[2] += 1
                        break
                else:
                    cl.append([float(x), float(y), 1])
        dreg = SR.snapshot(win_cl)
        cmap = {}
        for t in range(min(end + 1, T)):
            m = COORD.search(gs[t])
            if m:
                cmap.setdefault(obj_name(gs[t]),
                                (int(m.group(1)), int(m.group(2))))

        def render(kind, payload, t_=0):
            if kind in ("RAW", "RAWNS"):
                return payload
            pre, sg_ = payload
            sg_0 = sg_
            m_ = COORD.search(sg_)
            nm_ = obj_name(sg_)
            if nm_ == "object" and prev_obj[0]:
                nm_ = prev_obj[0]
            if m_:
                stats["sam_direct"] += 1
                hit = (int(m_.group(1)), int(m_.group(2)))
            else:
                hit = next((xy for k, xy in cmap.items()
                            if k == nm_ or k in nm_ or nm_ in k), None)
                if hit is not None:
                    stats["sam_cmap"] += 1
                else:
                    hit = SR.fill_from_registry(nm_, dreg)
                    if hit is None and prev_obj[0] == nm_:
                        hit = prev_obj[1]
                    stats["sam_det_fill" if hit else "sam_none"] += 1
            prev_obj[0], prev_obj[1] = nm_, hit
            sam = f"{nm_}<{hit[0]}, {hit[1]}>" if hit else "(none)"
            # v12 (user diagnosis, unmask family): put-down completions carried
            # no landing coordinate, were never oversampled and never
            # evidence-gated -- and the 2 closed-loop misses stranded the arm
            # mid-air for the rest of the episode.  The sentence now names the
            # landing spot so agent-2 can cross-check it, and the evidence gate
            # below requires the object to actually sit there.
            if pre == "completed" and hit and not m_                     and ("put down" in sg_.lower() or "put it down" in sg_.lower()):
                sg_ = f"{sg_} at <{hit[0]}, {hit[1]}>"
                stats["putdown_coord"] += 1
            ln_ = f"[event] {pre}: {sg_}  [sam] {sam}"
            if pre == "demo showed":
                # v12: ground demo lines in what the window actually shows.
                # v11 emitted the annotation's video coordinate verbatim, which
                # the writer cannot verify, so at serve time it guessed among
                # identical objects: 6 of 8 vrp and 3 of 4 vpo closed-loop
                # failures had a demo line naming the wrong one of N identical
                # cubes, which agent-2 then copied faithfully.
                ln_ = snap_line(ln_, t_)
                _g = _stroke_g.get((sg_0, t_))
                if _g:
                    (_gx0, _gy0), (_gx1, _gy1) = _g
                    _sfx = (f" (arm from <{_gx0}, {_gy0}> "
                            f"to <{_gx1}, {_gy1}>)")
                    ln_ = ln_.replace("  [sam]", _sfx + "  [sam]", 1)
                    stats["dir_grounded"] += 1
                if ((fam == "rs" and "circling" in sg_0.lower())
                        or (fam == "pl" and sg_0.lower().startswith("move "))) \
                        and (sg_0, t_) not in _stroke_arc:
                    return None   # duplicate re-entry boundary, not a stroke
                if (sg_0, t_) in _stroke_arc:
                    _ga = _stroke_arc[(sg_0, t_)]
                    _k = len(_seen_strokes) + 1
                    _seen_strokes.append(t_)
                    if _ga:
                        (_ax0, _ay0), (_bx0, _by0), _desc = _ga
                        ln_ = (f"[event] demo showed: stroke {_k} from "
                               f"<{int(_ax0)}, {int(_ay0)}> to "
                               f"<{int(_bx0)}, {int(_by0)}> — {_desc}  [sam] "
                               f"arc_tip<{int(_bx0)}, {int(_by0)}>")
                        stats["arc_chain_line"] += 1
                    else:
                        ln_ = (f"[event] demo showed: stroke {_k}: {sg_0}"
                               f"  [sam] (none)")
            return ln_

        win_ev = [(t_, _l) for t_, k, p in events if last < t_ <= end
                  for _l in [render(k, p, t_)] if _l is not None]
        win_ev.sort(key=lambda x: x[0])   # chronological; stable for ties
        # SPEC RULE, enforced mechanically: an event may only be emitted at a
        # tick whose own five frames contain its evidence.  This makes stamping
        # imprecision harmless -- an event simply moves to (or is dropped from)
        # the window that actually shows it.  v10 measured MOVE dynamics 0.36
        # and HIGHLIGHT 0.54 without this gate.
        if win_ev:
            _wd = [det_at(t) for t in ts]
            _prev = [d for t in _grid if last - SNAP <= t < last for d in _real[t]]
            _sites = [(u, v) for _, u, v in _prev]

            def _evident(ln):
                l = ln.lower()
                cs = [(int(a_), int(b_)) for a_, b_ in COORD.findall(ln)]
                if "appeared at" in l and "highlight" not in l and cs:
                    x, y = cs[0]
                    was = any((x-u)**2 + (y-v)**2 <= 14**2 for _, u, v in _prev)
                    now = any((x-u)**2 + (y-v)**2 <= 14**2
                              for f in _wd for _, u, v in f)
                    return now and not was
                if "highlight appeared" in l and cs:
                    x, y = cs[0]
                    return any(n == "highlight" and (x-u)**2 + (y-v)**2 <= 22**2
                               for f in _wd for n, u, v in f)
                if "arc tip moved" in l and len(cs) >= 2:
                    dx, dy = cs[-1]
                    return any(n == "arc_tip"
                               and (u-dx)**2 + (v-dy)**2 <= 12**2
                               for f in _wd for n, u, v in f)
                if ("is being moved" in l or "moved to" in l
                        or "moved from" in l):
                    # v14: DISPLACEMENT test, not off-site test.  vus demo
                    # swaps move 7-18px per 8-step window, so the old
                    # ">18px from every previous position" rule dropped ALL
                    # vus move lines (bus's faster 16-step exec windows
                    # passed).  Evidence = some container sits >=6px from its
                    # NEAREST previous-window container (SAM jitter <=2px).
                    _cls = "container" if "container" in l else "cube"
                    _pc = [(u, v) for n_, u, v in
                           [d for t in _grid if last - SNAP <= t < last
                            for d in _real[t]] if _cls in n_]
                    if not _pc:
                        return True
                    return any(_cls in n and min(
                        (u-su)**2 + (v-sv)**2 for su, sv in _pc) >= 6**2
                        for f in _wd for n, u, v in f)
                if "were placed over the cubes" in l:
                    # CONJUNCTIVE span rule (both halves from the SAM stream):
                    # at some claimed site, a cube mask VANISHES across this
                    # window while a container mask APPEARS there.  A cube-
                    # count-only test false-positives on ordinary picks (a
                    # grasped cube also vanishes, with no container involved).
                    # span = first frame vs last frame of the window
                    f0, f1 = _wd[0], _wd[-1]
                    for x, y in cs:
                        cube0 = any(n.endswith("_cube") and (x-u)**2 + (y-v)**2 <= 16**2
                                    for n, u, v in f0)
                        cube1 = any(n.endswith("_cube") and (x-u)**2 + (y-v)**2 <= 16**2
                                    for n, u, v in f1)
                        cont0 = any(n == "container" and (x-u)**2 + (y-v)**2 <= 16**2
                                    for n, u, v in f0)
                        cont1 = any(n == "container" and (x-u)**2 + (y-v)**2 <= 16**2
                                    for n, u, v in f1)
                        if cube0 and not cube1 and cont1 and not cont0:
                            return True
                    return False
                if ("put down" in l or "put it down" in l) and cs:
                    # landed: a detection sits at the claimed spot in BOTH of
                    # the window's last two frames (stable, not mid-carry)
                    x, y = cs[-1]
                    return all(any((x-u)**2 + (y-v)**2 <= 16**2
                                   for _, u, v in f) for f in _wd[-2:])
                if "reached the target" in l and cs:
                    tx, ty = cs[0]
                    return any("cube" in n and (u-tx)**2 + (v-ty)**2 <= 22**2
                               for f in _wd for n, u, v in f)
                return True

            _kept, _seen = [], set()
            for t_, ln in win_ev:
                key = re.sub(r"<[^>]*>", "", ln)
                if key in _seen:          # one line per event kind per tick
                    continue
                if not _evident(ln):
                    stats["evidence_dropped"] += 1
                    continue
                _seen.add(key)
                _kept.append((t_, ln))
            win_ev = _kept
        # v17 (user): knowability enforced EXACTLY -- a chain line's
        # coordinates must appear in this record's visible inputs
        # (bank[-8:] + frame_dets); otherwise demote to numbered-plain.
        if fam in ("rs", "pl") and win_ev:
            _vis = " ".join(bank[-8:]) + " " + " ".join(
                f"+{t2 - last}: {det_str(t2)}" for t2 in ts)
            _kept2 = []
            for _t2, _l2 in win_ev:
                _m2v = re.match(r"\[event\] demo showed: (stroke \d+) from "
                                r"<\d+, \d+> to <\d+, \d+> — ", _l2)
                if _m2v:
                    _cs2 = re.findall(r"<\d+, \d+>",
                                      _l2.split("  [sam]")[0])
                    if not all(_c2 in _vis for _c2 in _cs2):
                        if "circling " in _l2:      # rs chain
                            _sg2 = _l2.split("circling ", 1)[1] \
                                      .split(": ", 1)[-1].split("  [sam]")[0]
                        else:                        # pl chain
                            _sg2 = "move " + _l2.split(", move ", 1)[-1] \
                                                .split("  [sam]")[0]
                        _l2 = (f"[event] demo showed: {_m2v.group(1)}: "
                               f"{_sg2}  [sam] (none)")
                        stats["chain_demoted_unfindable"] += 1
                _kept2.append((_t2, _l2))
            win_ev = _kept2
        expected = a2_gt(last)
        # v17 (user): bus dense frames REVERTED -- bu presses fine at the
        # standard 5-frame windows, and bus's wall is the invisible press
        # registration, not evidence density.  Uniform frames everywhere.
        ts_r = ts
        _tgt = "\n".join(ln for _, ln in win_ev) if win_ev else "NONE"
        _rec = dict(
            instruction=instr, family=fam, ep=e, span=[int(last), int(end)],
            phase=("demo" if dm[min(end, T-1)] else "execution"),
            frames=save_frames(ts_r),
            frame_dets=[(f"+{t - last}: {det_str(t)}" if (t - last) % FSTEP == 0
                         else f"+{t - last}: (sam skipped)") for t in ts_r],
            bank=bank[-8:], expected_next=expected,
            target=_tgt, offset=OFF)
        # OVERSAMPLING (train only): the rare targets the closed-loop eval
        # showed agent-1 failing to emit -- swap/binding lines were 0.4% of
        # targets and were written in 1/10 bus and 0/10 vus episodes, and
        # multi-line windows lost every line after the first.
        _rep = 1
        if split == "train":
            if any(k in _tgt for k in ("moved to", "moved from", "being moved",
                                        "is now under", "is under the container")):
                _rep = OVER_SWAP
            elif any(k in _tgt for k in ("completed: put down", "completed: put it down",
                                          "completed: pick up the container")):
                _rep = OVER_COMPLETE
            elif len(_tgt.split("\n")) > 1:
                _rep = OVER_MULTI
            elif "reached the target" in _tgt:
                _rep = OVER_CROSS
        for _ in range(_rep):
            wr(a1[split], _rec)
        stats["a1"] += _rep
        if _rep > 1:
            stats["oversampled"] += _rep - 1
        stats["a1_write" if win_ev else "a1_none"] += 1
        for t_, ln in win_ev:
            push(end, ln)
        wr(a2[split], dict(
            instruction=instr, family=fam, ep=e, tq=int(end),
            phase=("demo" if dm[min(end, T-1)] else "execution"),
            bank=list(bank), target=a2_gt(end), offset=OFF,
            frames=save_frames([end]),
            frame_det=f"now: {det_str(end)}"))
        stats["a2"] += 1
        last = end
    stats[f"fam_{fam}"] += 1
    if (fi + 1) % 50 == 0:
        print(f"[gen] {fi+1}/{len(files)} eps {dict(stats)} "
              f"({(time.time()-t0)/60:.1f} min)", flush=True)

for fh in list(a1.values()) + list(a2.values()):
    fh.close()
print(f"[gen] DONE {dict(stats)}; {(time.time()-t0)/60:.1f} min", flush=True)

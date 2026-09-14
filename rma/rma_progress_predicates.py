#!/usr/bin/env python
"""Non-privileged progress predicates for RMA subtasks from the robot's own
8-d state (ee xyz | ee axis-angle | gripper q6,q7), tick = 10 control steps.
  pick  : gripper closes on something (q6 < 0.036 for 3 steps) and the ee then
          rises >= 3 cm above its height at closure  (mirrors the oracle's
          'object risen 3 cm' rule, read from the hand instead of the object)
  place : after a pick, the gripper re-opens (q6 > 0.037 for 3 steps)
  pour  : gripper closed and the wrist tilted (gripper approach axis > 55 deg
          from straight down) for >= 8 steps; completion when it levels again
  open/close (drawer, microwave door): a handle grasp (q6 < 0.020) followed by
          release; push-style strokes (gripper stays open) are NOT covered here
Audit: predicted completion step vs recorded segment end, per subtask type.
"""
import os, sys, json, pickle, re, collections
import numpy as np
from scipy.spatial.transform import Rotation as Rot
O = os.path.expandvars('${ROBOMME_ROOT}/data/rma_preprocessed_data')
TICK = 10
_CAM = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rma_agentview_camera.json')))
_M = np.asarray(_CAM['matrix'], float); _HW = tuple(_CAM['hw']); LOC_R = float(os.environ.get('LOC_R', '30'))

def project(p):
    w = np.array([p[0], p[1], p[2], 1.0]); c = _M @ w; pix = c[:2] / c[2]
    return int(np.clip(round(pix[1]), 0, _HW[0] - 1)), int(np.clip(round(pix[0]), 0, _HW[1] - 1))

def near(xyz, target):
    if target is None: return True
    r, c = project(xyz); return ((r - target[0]) ** 2 + (c - target[1]) ** 2) ** 0.5 <= LOC_R
SEQUENTIAL = os.environ.get('SEQUENTIAL', '0') == '1'

def kind_of(instr):
    i = instr.lower()
    if i.startswith('pick'): return 'pick'
    if i.startswith('place') or i.startswith('put'): return 'place'
    if i.startswith('pour'): return 'pour'
    if i.startswith('open'): return 'open'
    if i.startswith('close'): return 'close'
    return 'other'

def tilt_deg(aa):
    R = Rot.from_rotvec(aa).as_matrix(); z = R @ np.array([0, 0, 1.0])   # gripper approach axis (world)
    return float(np.degrees(np.arccos(np.clip(-z[2], -1, 1))))          # 0 = pointing straight down

def sustained(mask, k):
    out = np.zeros_like(mask); run = 0
    for i, m in enumerate(mask):
        run = run + 1 if m else 0
        out[i] = run >= k
    return out

def grasp_mask(st, band=(0.006, 0.037)):
    q = st[:, 6]
    return sustained((q > band[0]) & (q < band[1]) & (np.r_[True, np.abs(np.diff(q)) < 0.0015]), 5)

def predict(st, segs, targets=None, widths=None):
    """v6b = v5 + location gates: place/pour at non-drawer fixtures complete only with the
    end-effector within LOC_R px (image plane) of the step's fixture position
    (targets[i] = (row, col) training-set median from the plan file; None = no gate).
    v5 (deployment order: each predicate scans from the last verified
    completion, so it only ever sees the current plan step's motion).
    pick : grasp (gripper stops closing on something) then ee risen >= 3 cm
    place: a grasp WITH a lift (>= 3 cm) since the last completion, held >= 20
           steps, then release (q6 > 0.037, 3 steps)  -- a handle pull has no lift
    pour : wrist rotated >= 25 deg away from the carry orientation (tilt at the
           scan start) for >= 15 steps, done when back within 10 deg (5 steps);
           sign-free: the sauce tilts 0->45, the side-grasped milk 63->17
    open : handle grasp in drawer/door posture (tilt > 75) then a stroke >= 0.10 m
           from the grasp point; done when the ee stops
    close: gripper closed on the handle/door edge, a stroke that shortens the
           distance to the matching open's grasp point by >= 0.10 m, ending
           within 0.12 m of it; done when the ee stops"""
    q = st[:, 6]; z = st[:, 2]; xy = st[:, :2]; tilt = np.array([tilt_deg(a_) for a_ in st[:, 3:6]])
    opened = sustained(q > 0.037, 3); posture = tilt > 75; closedq = sustained(q < 0.037, 5)
    speed = np.r_[0, np.linalg.norm(np.diff(xy, axis=0), axis=1)]; still = sustained(speed < 0.001, 5)
    preds = []; n = len(st); handle_pos = {}; prev_p = None
    for si, (a, ln, ins) in enumerate(segs):
        tgt = targets[si] if targets is not None and si < len(targets) else None
        band = widths[si] if widths is not None and si < len(widths) and widths[si] else (0.006, 0.037)
        grasp = grasp_mask(st, band); inband = (q > band[0]) & (q < band[1])
        if tgt is not None and 'drawer' in ins.lower(): tgt = None      # drawer medians are post-retract; ungated
        b = a + ln - 1; k = kind_of(ins); p = None; a_rec = a
        if SEQUENTIAL and prev_p is not None: a = prev_p + 1
        if k == 'pick':
            g = [i for i in range(a, n) if grasp[i] and np.any(q[a:i + 1] > 0.037)]   # a NEW grasp: gripper was open since the scan start
            if g:
                c = g[0]; zc = z[c]
                up = [i for i in range(c, n) if z[i] >= zc + 0.03 and q[i] < 0.037]
                p = up[0] if up else None
        elif k == 'place':
            g = [i for i in range(max(0, a - 120), n) if grasp[i]]
            lifted = None
            for c in g[:1]:
                pass
            i0 = max(0, a - 120)
            if inband[a]:                                   # already holding something: walk back to the grasp start
                s0 = a
                while s0 > 0 and inband[s0 - 1]: s0 -= 1
                i0 = s0
            # first grasp-with-lift after i0
            c = next((i for i in range(i0, n) if grasp[i]), None)
            while c is not None:
                zc = z[c]
                up = next((i for i in range(c, n) if q[i] > 0.037 or z[i] >= zc + 0.03), None)
                if up is not None and z[up] >= zc + 0.03 and q[up] <= 0.037: lifted = up; break
                c = next((i for i in range((up or c) + 1, n) if grasp[i]), None)
            if lifted is not None:
                op = [i for i in range(max(lifted + 20, a), n) if opened[i] and (i == 0 or not opened[i - 1])]   # release moments
                p = next((i for i in op if near(st[i, :3], tgt)), None)
        elif k == 'pour':
            base = tilt[a]
            dev = sustained(np.abs(tilt - base) >= 25, 15)
            t = [i for i in range(a, n) if dev[i] and near(st[i, :3], tgt)]
            if t:
                back = sustained(np.abs(tilt - base) <= 10, 5)
                e = next((i for i in range(t[0], n) if back[i]), None)
                p = e if e is not None else None
        elif k == 'open':
            gh = grasp_mask(st, (0.006, 0.037))
            g = [i for i in range(a, n) if gh[i] and posture[i] and np.any(q[a:i + 1] > 0.037)]
            if g:
                c = g[0]; origin = xy[c]
                far = [i for i in range(c, n) if np.linalg.norm(xy[i] - origin) >= 0.10]
                if far:
                    e = far[0]
                    while e + 1 < n and not still[e + 1]: e += 1
                    p = e
                key = re.sub(r'\s+again$', '', ins.lower().replace('open ', '')); handle_pos[key] = origin
        elif k == 'close':
            key = re.sub(r'\s+(again|final)$', '', ins.lower().replace('close ', ''))
            origin = handle_pos.get(key)
            if origin is not None:
                d = np.linalg.norm(xy - origin, axis=1)
                i = a
                while i < n and p is None:
                    if closedq[i]:
                        d0 = d[i]; j = i
                        while j + 1 < n and closedq[j + 1]: j += 1
                        seg_d = d[i:j + 1]; kmin = i + int(np.argmin(seg_d))
                        if d0 - d[kmin] >= 0.06 and d[kmin] <= 0.12:
                            e = kmin
                            while e + 1 <= j and not still[e + 1]: e += 1
                            p = e
                        i = j + 1
                    else: i += 1
        preds.append((k, ins, a_rec, b, p)); prev_p = p if p is not None else prev_p
    return preds

PLANS = {int(k): v for k, v in json.load(open(os.path.expandvars('${GAMMA_ROOT}/rma/eval_stack/rma_oracle_plans.json'))).items()}

if __name__ == '__main__':
    n_per_task = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    L = json.load(open(f'{O}/meta/episode_lengths.json')); T = json.load(open(f'{O}/meta/episode_task.json')); SEG = json.load(open(f'{O}/meta/segments.json'))
    order = sorted(L, key=int); off = {}; o = 0
    for e in order: off[e] = o; o += L[e]
    by_task = collections.defaultdict(list)
    for e in order:
        if len(by_task[T[e]['task']]) < n_per_task: by_task[T[e]['task']].append(e)
    err = collections.defaultdict(list); miss = collections.Counter(); tot = collections.Counter(); rows = []
    inside = collections.Counter()
    for task in sorted(by_task, key=lambda t: int(t[4:])):
        for e in by_task[task]:
            st = np.stack([np.asarray(pickle.load(open(f'{O}/data/{i}.pkl', 'rb'))['state'], dtype=float) for i in range(off[e], off[e] + L[e])])
            segs = list(zip(SEG[e]['seg_start'], SEG[e]['seg_len'], SEG[e]['instructions']))
            plan = PLANS.get(int(task[4:]), [])
            tg = [next((tuple(s_['median']) for s_ in plan if s_['instr'].strip().lower() == ins.strip().lower() and s_.get('kind') != 'pick'), None) for _, _, ins in segs]
            wd = None
            if os.environ.get('USE_WIDTHS', '0') == '1':
                W = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'grasp_widths.json')))
                def _band(ins):
                    i = ins.lower(); o = next((x for x in ('cookies', 'tomato sauce', 'butter', 'popcorn', 'cream', 'chocolate', 'pudding', 'milk', 'wine', 'orange', 'sauce') if x in i), None)
                    k_ = kind_of(ins)
                    if o is None or k_ not in ('pick', 'place'): return None
                    ent = W.get(('carry:' + o) if k_ == 'place' else o) or W.get(o)
                    return (max(0.006, ent['p10'] - 0.004), min(0.0395, ent['p90'] + 0.004)) if ent else None
                wd = [_band(ins) for _, _, ins in segs]
            for k, ins, a, b, p in predict(st, segs, tg, wd):
                tot[k] += 1
                if p is None: miss[k] += 1; rows.append((task, e, k, ins, a, b, None)); continue
                err[k].append((p - b) / TICK); rows.append((task, e, k, ins, a, b, p))
                inside[k] += (a <= p <= b + 20)
    print('type   n   miss  inside-own-segment   median err(ticks)   |err|<=2 ticks   |err|<=4')
    for k in ('pick', 'place', 'pour', 'open', 'close', 'other'):
        if not tot[k]: continue
        a = np.array(err[k]) if err[k] else np.array([])
        print(f'{k:6s} {tot[k]:3d}  {miss[k]:3d}     {inside[k]/tot[k]:.2f}            {np.median(a) if len(a) else float("nan"):+6.1f}          {np.mean(np.abs(a) <= 2) if len(a) else 0:.2f}            {np.mean(np.abs(a) <= 4) if len(a) else 0:.2f}')
    json.dump(rows, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'predicate_audit.json'), 'w'))
    bad = [r for r in rows if r[6] is None or not (r[4] <= r[6] <= r[5] + 20)]
    print(f'\nworst/missed ({len(bad)}):'); [print('  ', r) for r in bad[:25]]

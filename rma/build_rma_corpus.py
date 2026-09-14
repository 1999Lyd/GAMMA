#!/usr/bin/env python
"""RMA writer/reasoner corpus (v1) in the wam_sft_v17 record schema.
Tick = 10 control steps; window = 3 grid frames (t-10, t-5, t). Per frame the
writer sees the r6 SAM-3 detections (top-1 per concept, score >= DET_THR) plus a
proprioceptive readout projected into the same frame:
    ee<row, col> grip:open|closed(q) z:0.74 tilt:12
Targets (mechanically verified, evidence-gated by rma_progress_predicates v4):
  tick 1                     [event] initial scene: <name> at <r, c>, ...  [sam] ...
  completion tick of step j  [event] completed: <subtask>  [sam] ee<r, c> grip:...
  otherwise                  NONE   (kept at most NONE_RATIO x event ticks / episode)
expected_next = the plan step in progress at the tick start (plan-pinned reasoner);
bank = ground-truth lines admitted so far (stage-1 corpus).
Reasoner records: bank -> plan step in progress at the next tick (last step is
repeated after the plan completes, as the oracle feed does).
Splits: seeds 100-131 train, 132-139 val.  Env: OUT, SEEDS (default 40), DET_THR.
"""
import os, sys, json, re, glob, random, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expandvars('${ROBOMME_ROOT}/src/mme_vla_suite/dataset_builder'))
import rma_progress_predicates as pp
from rma_h5_utils import discover_task_dirs, discover_episodes, load_episode
from scipy.spatial.transform import Rotation as Rot

F = os.path.expandvars('${GAMMA_DATA}/data/rma_sft_v1'); RAW = os.path.expandvars('${GAMMA_DATA}/data/RoboMemArena')
OUT = os.environ.get('OUT', f'{F}/corpus_v1'); NSEED = int(os.environ.get('SEEDS', '40')); DET_THR = float(os.environ.get('DET_THR', '0.15'))
NONE_RATIO = float(os.environ.get('NONE_RATIO', '3')); TICK = 10; VAL_SEEDS = set(range(132, 140))
CAM = json.load(open(f'{os.path.dirname(os.path.abspath(__file__))}/rma_agentview_camera.json')); M = np.asarray(CAM['matrix'], float); HW = tuple(CAM['hw'])
sys.path.insert(0, os.path.expandvars('${ROBOMME_ROOT}/src/mme_vla_suite/dataset_builder'))
from rma_task_prompts import TASK_PROMPTS
PRETTY = {'tomato_sauce': 'tomato sauce', 'orange_juice': 'orange juice', 'drawer_open': 'open drawer', 'drawer_content': 'drawer content', 'microwave_open': 'open microwave'}

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

def scene_line(dets0):
    best = {}
    for name, r, c, sc, *_ in dets0:
        if name == 'arm' or sc < DET_THR: continue
        if name not in best or sc > best[name][2]: best[name] = (r, c, sc)
    items = [f"{PRETTY.get(n, n).replace('_', ' ')} at <{r}, {c}>" for n, (r, c, _) in sorted(best.items(), key=lambda kv: -kv[1][2])]
    sam = ' '.join(f'{n}<{r}, {c}>' for n, (r, c, _) in best.items())
    return f"[event] initial scene: {', '.join(items) if items else 'no objects detected'}  [sam] {sam or '(none)'}"

def main():
    os.makedirs(OUT, exist_ok=True); random.seed(0)
    w1 = {s: open(f'{OUT}/agent1_{s}.jsonl', 'w') for s in ('train', 'val')}; w2 = {s: open(f'{OUT}/agent2_{s}.jsonl', 'w') for s in ('train', 'val')}
    stats = collections.Counter(); audit = []
    for td in sorted(discover_task_dirs(RAW), key=lambda d: d['task_num']):
        T = td['task_num']; instr = TASK_PROMPTS[f'task{T}'].strip().lower()
        for ep in sorted(discover_episodes(td['path']), key=lambda e: e['seed']):
            S = ep['seed']
            if S >= 100 + NSEED: continue
            cache = f'{F}/dets_sam3/task{T}_seed{S}.json'
            if not os.path.exists(cache): stats['no_cache'] += 1; continue
            dets = json.load(open(cache))['dets']; e = load_episode(ep, load_images=False)
            st = e['state']; L = len(st); segs = list(zip(e['seg_start'], e['seg_len'], e['instructions']))
            plan = [ins.strip().lower() for ins in e['instructions']]
            preds = pp.predict(st, segs)                       # sequential inside (SEQUENTIAL env respected by module)
            comp = {}                                          # completion step -> plan index
            for j, (k, ins, a, b, p) in enumerate(preds):
                if p is None: stats['missing_completion'] += 1; audit.append((T, S, j, ins)); continue
                comp[p] = j
            split = 'val' if S in VAL_SEEDS else 'train'; fam = td['suite']
            bank = [scene_line(dets.get('0', []))]; j_done = 0
            ticks = list(range(1, (L - 1) // TICK + 1)); recs = []
            for k in ticks:
                t0, t1 = (k - 1) * TICK, k * TICK
                # nearest grid frame to each anchor t0 / t0+5 / t1 (the extractor's
                # stride-5 grid restarts at every recorded subtask, so labels jitter
                # by <= 4 steps; serve time uses exactly +0/+5/+10)
                gsteps = sorted(int(x) for x in dets if int(x) <= t1 + 4)
                fr = sorted({min(gsteps, key=lambda t: (abs(t - anc), -t)) for anc in (t0, t0 + 5, t1)})
                frame_dets = [f"{t - t0:+d}: {det_string(dets[str(t)])} {state_string(st[min(t, L - 1)])}".strip() for t in fr]
                done_here = [comp[p] for p in range(t0 + 1, t1 + 1) if p in comp]
                expected = plan[min(j_done, len(plan) - 1)]
                lines = []
                if k == 1: lines.append(bank[0])
                for j in done_here:
                    s_end = st[min(t1, L - 1)]
                    lines.append(f"[event] completed: {plan[j]}  [sam] {state_string(s_end)}")
                target = '\n'.join(lines) if lines else 'NONE'
                rec = {'instruction': instr, 'family': fam, 'ep': f'task{T}_seed{S}', 'span': [t0, t1], 'phase': 'execution',
                       'frames': [f'frames/task{T}_seed{S}/f{t}.jpg' for t in fr], 'frame_dets': frame_dets,
                       'bank': list(bank[-8:]) if k > 1 else [], 'expected_next': expected, 'target': target, 'offset': 0}
                if done_here: j_done = max(done_here) + 1
                for ln in lines[1:] if k == 1 else lines: bank.append(ln)
                nxt = plan[min(j_done, len(plan) - 1)]
                rec2 = {'instruction': instr, 'family': fam, 'ep': f'task{T}_seed{S}', 'tq': t1, 'phase': 'execution', 'bank': list(bank),
                        'target': nxt, 'offset': 0, 'frames': [f'frames/task{T}_seed{S}/f{fr[-1]}.jpg'], 'frame_det': f"now: {state_string(st[min(t1, L - 1)])}", 'target_oracle': nxt}
                recs.append((rec, rec2, bool(lines)))
            n_ev = sum(1 for r in recs if r[2]); keep_none = int(NONE_RATIO * n_ev)
            none_idx = [i for i, r in enumerate(recs) if not r[2]]; random.shuffle(none_idx); keep = set(none_idx[:keep_none])
            for i, (rec, rec2, ev) in enumerate(recs):
                if ev or i in keep:
                    w1[split].write(json.dumps(rec) + '\n'); stats[f'a1_{split}_event' if ev else f'a1_{split}_none'] += 1
                w2[split].write(json.dumps(rec2) + '\n'); stats[f'a2_{split}'] += 1
            stats['episodes'] += 1
    for f in list(w1.values()) + list(w2.values()): f.close()
    json.dump({'stats': dict(stats), 'missing': audit}, open(f'{OUT}/build_stats.json', 'w'), indent=1)
    print(json.dumps(dict(stats), indent=1)); print('missing completions:', len(audit), audit[:10])

if __name__ == '__main__':
    os.environ.setdefault('SEQUENTIAL', '1'); pp.SEQUENTIAL = True
    main()

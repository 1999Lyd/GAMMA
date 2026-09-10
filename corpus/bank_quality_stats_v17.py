import os
#!/usr/bin/env python
"""Tick-by-tick statistical comparison of agent-1's PREDICTED bank against the
ORACLE (GT) bank on val episodes.

Per tick t the two cumulative bank states are matched line-by-line
(normalised text + coords within 12px).  Reports, per task and overall:
  state_precision / state_recall : averaged over all ticks
  final_recall                   : GT lines present in the pred bank at episode end
  timing                         : tick offset (pred - gt) for matched emissions
  coord_err                      : per matched line with coordinates
  completion_by_need             : GT completion lines present in the pred bank
                                   by the tick agent-2 first needs them (+1 tick grace)
  divergence_onset               : first tick where state_recall < 1
"""
import json, re, collections
import numpy as np

V=os.path.expandvars('${GAMMA_DATA}/data/wam_sft_v17')
CO=re.compile(r"<\s*(\d+)\s*,\s*(\d+)\s*>")
TOL=12

def norm(ln):
    s=CO.sub("<>", ln)
    s=re.sub(r"\s+"," ",s.replace("[event]","").split("  [sam]")[0]).strip().lower()
    return s
def coords(ln):
    return [(int(a),int(b)) for a,b in CO.findall(ln.split("  [sam]")[0])]
def match(g,p):
    if norm(g)!=norm(p): return False
    cg,cp=coords(g),coords(p)
    if len(cg)!=len(cp): return False
    return all(abs(x-u)<=TOL and abs(y-v)<=TOL for (x,y),(u,v) in zip(cg,cp))

gt=collections.defaultdict(list)
for l in open(f'{V}/agent1_val.jsonl'):
    r=json.loads(l)
    if r.get('offset',0)==0 and r['target'].strip()!='NONE':
        for ln in r['target'].split('\n'):
            gt[(r['ep'])].append((r['span'][1], ln, r['family']))
pred=collections.defaultdict(list)
fam_of={}
for l in open(f'{V}/agent1_rollout_val.jsonl'):
    r=json.loads(l); fam_of[r['ep']]=r['family']
    if r['pred'].strip().upper()!='NONE':
        for ln in r['pred'].split('\n'):
            if ln.strip(): pred[r['ep']].append((r['tq'], ln.strip()))

P=collections.defaultdict(list); R=collections.defaultdict(list)
FR=collections.defaultdict(list); TIM=collections.defaultdict(list)
CE=collections.defaultdict(list); DIV=collections.defaultdict(list)
CBN=collections.defaultdict(lambda:[0,0])
for ep in sorted(gt):
    fam=fam_of.get(ep) or gt[ep][0][2]
    gl=[(t,ln) for t,ln,_ in gt[ep]]; pl=pred.get(ep,[])
    ticks=sorted({t for t,_ in gl}|{t for t,_ in pl})
    div=None
    for t in ticks:
        G=[ln for tt,ln in gl if tt<=t]; Pd=[ln for tt,ln in pl if tt<=t]
        if not G and not Pd: continue
        used=set(); hit=0
        for g_ in G:
            for j,p_ in enumerate(Pd):
                if j in used: continue
                if match(g_,p_): used.add(j); hit+=1; break
        rec=hit/max(len(G),1); prc=hit/max(len(Pd),1) if Pd else (1.0 if not G else 0.0)
        R[fam].append(rec); P[fam].append(prc)
        if div is None and rec<1.0: div=t
    DIV[fam].append(div if div is not None else -1)
    # final state + timing + coord error
    G=[(t,ln) for t,ln in gl]; Pd=pl
    used=set(); 
    fr=0
    for tg,g_ in G:
        found=None
        for j,(tp,p_) in enumerate(Pd):
            if j in used: continue
            if match(g_,p_): found=(j,tp); break
        if found:
            used.add(found[0]); fr+=1
            TIM[fam].append((found[1]-tg)//16)
            cg,cp=coords(g_),coords(Pd[found[0]][1])
            for (x,y),(u,v) in zip(cg,cp): CE[fam].append(float(np.hypot(x-u,y-v)))
        # completion-by-need: needed at its GT tick + 1
        if 'completed:' in g_:
            CBN[fam][1]+=1
            ok=any(j in used and Pd[j][0]<=tg+16 and match(g_,Pd[j][1]) for j in range(len(Pd)))
            CBN[fam][0]+=ok
    FR[fam].append(fr/max(len(G),1))

# completion precision: of the writer's emitted completion lines, how many
# match a GT completion (same normalised content, emission within +-1 tick)
CP=collections.defaultdict(lambda:[0,0])
for ep in sorted(pred):
    fam=fam_of.get(ep)
    gl=[(t,ln) for t,ln,_ in gt[ep] if 'completed:' in ln]
    for tp,pln in pred.get(ep,[]):
        if 'completed:' not in pln: continue
        CP[fam][1]+=1
        ok_=any(norm(g)==norm(pln) and abs(tp-tg)<=16 for tg,g in gl)
        CP[fam][0]+=ok_
print(f"{'task':10s} {'complPRECISION (emitted completions correct+timed)':>50s}")
tot=[0,0]
for f in sorted(CP):
    c=CP[f]; tot[0]+=c[0]; tot[1]+=c[1]
    flag='  <-- BELOW 0.9 BAR' if c[1] and c[0]/c[1]<0.9 else ''
    print(f"{f:10s} {c[0]}/{c[1]} = {c[0]/max(c[1],1):.3f}{flag}")
print(f"{'ALL':10s} {tot[0]}/{tot[1]} = {tot[0]/max(tot[1],1):.3f}   (user bar: >=0.90)")
print()
print(f"{'task':10s} {'stateP':>7s} {'stateR':>7s} {'finalR':>7s} {'complBYneed':>12s} "
      f"{'lag(med tk)':>11s} {'coordE(med)':>11s} {'divergence@tick':>16s}")
tots=[[] for _ in range(6)]
for f in sorted(R):
    c=CBN[f]
    dv=[d//16 for d in DIV[f] if d>0]
    row=(np.mean(P[f]), np.mean(R[f]), np.mean(FR[f]),
         c[0]/max(c[1],1), np.median(TIM[f]) if TIM[f] else 0,
         np.median(CE[f]) if CE[f] else 0)
    print(f"{f:10s} {row[0]:7.3f} {row[1]:7.3f} {row[2]:7.3f} {row[3]:12.3f} "
          f"{row[4]:11.1f} {row[5]:11.1f} "
          f"{'never' if not dv else f'median t{int(np.median(dv))}':>16s}")
    for i,v in enumerate(row): tots[i].append(v)
print(f"{'MEAN':10s} " + " ".join(f"{np.mean(t):7.3f}" if i<4 else f"{np.mean(t):11.1f}"
      for i,t in enumerate(tots)))

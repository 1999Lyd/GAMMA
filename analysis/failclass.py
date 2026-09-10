#!/usr/bin/env python3
"""Failure-class diagnosis of the 0.8B v21c harness-on run, per task, from
serve traces (last run per episode) joined with progress.json outcomes."""
import json, re, glob, os, sys
from collections import defaultdict, Counter
E=os.path.expandvars("${ROBOMME_ROOT}/examples/robomme/runs/evaluation")
FQ=os.path.expandvars("${GAMMA_DATA}/traces")
LANES={"W":"BinFill PickXtimes ButtonUnmask ButtonUnmaskSwap","X":"VideoUnmask VideoRepick MoveCube PatternLock",
       "Y":"StopCube SwingXtimes PickHighlight VideoUnmaskSwap","Z":"VideoPlaceButton VideoPlaceOrder InsertPeg RouteStick"}
c=lambda s: [(int(a),int(b)) for a,b in re.findall(r'<\s*(\d+)\s*,\s*(\d+)', s or '')]
strip=lambda s: re.sub(r'\s*at\s*<[^>]*>','',(s or '')).strip().lower()
only=set(sys.argv[1:])
for lane,tasks in LANES.items():
    prog=json.load(open(f"{E}/wam_0p8b_on20_lane{lane}/ckpt79999/seed7/oracle/progress.json"))
    runs=defaultdict(dict); cur=None
    for line in open(f"{FQ}/wam_live_bank_0p8b_on19_lane{lane}.jsonl"):
        r=json.loads(line)
        if r.get('kind')=='reset': cur=(r['task'],r['ep']); runs[r['task']][r['ep']]=[]
        elif r.get('kind')=='tick' and cur: runs[cur[0]][cur[1]].append(r)
    for task in tasks.split():
        if only and task not in only: continue
        P=prog.get(task,{}); classes=Counter(); detail=[]
        for ep_s,ok in P.items():
            if not isinstance(ok,bool): continue
            ep=int(ep_s); T=runs[task].get(ep,[])
            if ok: classes['success']+=1; continue
            if not T: classes['no-trace']+=1; continue
            O=[strip(t.get('oracle')) for t in T]; A=[strip(t.get('agent2_out')) for t in T]
            match=[o==a for o,a in zip(O,A)]
            cm=sum(match)/len(T)
            bad=tot=0
            for t,m in zip(T,match):
                if m and c(t.get('oracle')) and c(t.get('agent2_out')):
                    tot+=1; (ox,oy),(ax,ay)=c(t['oracle'])[0],c(t['agent2_out'])[0]
                    bad+= abs(ox-ax)+abs(oy-ay)>14
            cb=bad/tot if tot else 0
            # behind: a2 content equals an EARLIER oracle content; ahead: equals a LATER one
            behind=ahead=0; run_b=0; maxb=0
            for i,(o,a) in enumerate(zip(O,A)):
                if o==a: run_b=0; continue
                if a in O[:i]: behind+=1; run_b+=1; maxb=max(maxb,run_b)
                elif a in O[i+1:]: ahead+=1; run_b=0
                else: run_b=0
            if cm>=0.8 and cb<=0.3: k='policy/other (a2 matched oracle)'
            elif cb>0.5: k='wrong coordinate'
            elif maxb>=8: k='STALL behind oracle'
            elif ahead>=3: k='PREMATURE ahead of oracle'
            else: k='content wrong'
            classes[k]+=1
        n=sum(classes.values())
        print(f"{task:<18} n={n:>2}  " + ", ".join(f"{k}={v}" for k,v in classes.most_common()))

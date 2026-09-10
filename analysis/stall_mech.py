import os
#!/usr/bin/env python3
"""For STALL-behind episodes: was a writer completion claim HELD by a gate
(claimed, never admitted) or did the writer stay SILENT (no claim)?"""
import json, re, sys
from collections import defaultdict, Counter
E=os.path.expandvars("${ROBOMME_ROOT}/examples/robomme/runs/evaluation")
FQ=os.path.expandvars("${GAMMA_DATA}/traces")
LANE={"BinFill":"W","PickXtimes":"W","ButtonUnmaskSwap":"W","SwingXtimes":"Y","VideoPlaceOrder":"Z","ButtonUnmask":"W","VideoRepick":"X"}
strip=lambda s: re.sub(r'\s*at\s*<[^>]*>','',(s or '')).strip().lower()
for task in sys.argv[1:]:
    lane=LANE[task]
    prog=json.load(open(f"{E}/wam_0p8b_on20_lane{lane}/ckpt79999/seed7/oracle/progress.json")).get(task,{})
    runs=defaultdict(list); cur=None
    for line in open(f"{FQ}/wam_live_bank_0p8b_on19_lane{lane}.jsonl"):
        if f'"{task}"' not in line: continue
        r=json.loads(line)
        if r.get('task')!=task: continue
        if r['kind']=='reset': cur=r['ep']; runs[cur]=[]
        elif r['kind']=='tick' and cur is not None: runs[cur].append(r)
    mech=Counter(); ex=[]
    for ep_s,ok in prog.items():
        if ok is not False: continue
        T=runs.get(int(ep_s),[])
        O=[strip(t.get('oracle')) for t in T]; A=[strip(t.get('agent2_out')) for t in T]
        # find the longest behind-run window
        best=(0,0,0); run=0; start=0
        for i,(o,a) in enumerate(zip(O,A)):
            if o!=a and a in O[:i]:
                if run==0: start=i
                run+=1
                if run>best[0]: best=(run,start,i)
            else: run=0
        if best[0]<8: continue
        n,s,e=best
        # the oracle subgoal the plan is stuck BEHIND (what should have completed)
        stuck_on=A[s]
        claims=[]; admitted=set()
        for t in T[s:e+1]:
            a1=t.get('agent1_out') or ''
            for ln in a1.split('\n'):
                if 'completed:' in ln: claims.append(strip(ln.split('completed:')[1].split('[sam]')[0]))
            for ln in t.get('bank',[]):
                if 'completed:' in ln: admitted.add(strip(ln.split('completed:')[1].split('[sam]')[0]))
        held=[cl for cl in claims if cl not in admitted]
        if not claims: k='writer SILENT (no completion claim)'
        elif held: k='claim HELD by gate'
        else: k='claim admitted but a2 still behind'
        mech[k]+=1
        if len(ex)<3: ex.append((ep_s,n,stuck_on[:50],k,(held[:1] or claims[:1])))
    print(f"== {task}: stalls by mechanism: {dict(mech)}")
    for e in ex: print("   ", e)

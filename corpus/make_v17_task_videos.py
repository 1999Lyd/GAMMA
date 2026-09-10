#!/usr/bin/env python
"""bus+vus only, v17 recipe: episode frames with the agent-1
training target per tick, the oracle subgoal, and the accumulating bank."""
import glob, io, json, os, shutil, time
import numpy as np, cv2
from PIL import Image
import pyarrow.parquet as pq

V=os.path.expandvars('${GAMMA_DATA}/data/wam_sft_v17')
D=os.path.expandvars('${GAMMA_DATA}/data/robomme_lerobot')
OUTD=os.path.expandvars('${GAMMA_DATA}/results/v17_training_videos')
os.makedirs(OUTD, exist_ok=True)
TASK16={0:"pl",1:"bus",2:"bu",3:"vpb",4:"vus",5:"pickx",6:"stopcube",7:"swingx",8:"ph",9:"mc",10:"ip",11:"rs",12:"binfill",13:"vpo",14:"vrp",15:"vu"}
files=sorted(glob.glob(D+'/data/chunk-*/*.parquet'))
recs_by_ep={}
for l in open(f'{V}/agent1_val.jsonl'):
    r=json.loads(l)
    if r.get('offset',0)==0:
        recs_by_ep.setdefault(r['ep'],[]).append(r)
a2_by={}
for l in open(f'{V}/agent2_val.jsonl'):
    r=json.loads(l)
    if r.get('offset',0)==0:
        a2_by[(r['ep'],r['tq'])]=r['target']
def wrap(s,w=60):
    out,line=[],""
    for word in (s or "").split():
        if len(line)+len(word)+1>w: out.append(line); line=word
        else: line=word if not line else line+" "+word
    if line: out.append(line)
    return out or ["(none)"]
A=lambda s:s.replace('—','-').replace('·','.')
font=cv2.FONT_HERSHEY_SIMPLEX
t0=time.time()
for blk,fam in TASK16.items():
    ep=blk*100
    recs=sorted(recs_by_ep.get(ep,[]),key=lambda r:(r['span'][1],r['span'][0]))
    if not recs: print(f"skip {fam}: no records for ep{ep}"); continue
    out=f'{OUTD}/{fam}_ep{ep}_v17_targets.mp4'
    tb=pq.read_table(files[ep])
    si=tb["step_idx"].to_pylist(); ix=sorted(range(len(si)),key=lambda i:si[i])
    im=tb["image"].to_pylist()
    gso=[tb["grounded_subgoal_online"].to_pylist()[i] or "" for i in ix]
    T=len(ix)
    TMP=out+'.frames'; shutil.rmtree(TMP,ignore_errors=True); os.makedirs(TMP)
    bank=[]; fi=0; ri=0
    for t in range(0,T,4):
        while ri<len(recs)-1 and recs[ri]['span'][1] < t: 
            if recs[ri]['target'].strip()!='NONE':
                for ln in recs[ri]['target'].split('\n'):
                    bank.append(ln.split('  [sam]')[0].replace('[event] ',''))
            # item 10: agent-2's declared completions enter the bank at the
            # end of their window (after its judgment) -- show them here
            _pt=a2_by.get((ep,recs[ri]['span'][1])) or ''
            if _pt.startswith('completed:') and ' | next:' in _pt:
                for _c in _pt[len('completed: '):].split(' | next:')[0].split('; '):
                    if _c.strip(): bank.append(f"completed: {_c.strip()}")
            ri+=1
        r=recs[ri]
        raw=im[ix[t]]; raw=raw["bytes"] if isinstance(raw,dict) else raw
        img=np.array(Image.open(io.BytesIO(raw)).convert("RGB"))[:,:,::-1].copy()
        big=cv2.resize(img,(640,640),interpolation=cv2.INTER_CUBIC)
        # overlay the SAM detections agent-1 receives for this frame
        fds=r.get('frame_dets') or []
        a0=r['span'][0]
        fdi=min(max((t-a0)//4,0),len(fds)-1) if fds else -1
        if fdi>=0:
            import re as _re
            for m in _re.finditer(r"(\w+)<(\d+), (\d+)>",fds[fdi]):
                n_,rr_,cc_=m.group(1),int(m.group(2)),int(m.group(3))
                px,py=int(cc_*2.5),int(rr_*2.5)   # dets are <row, col>
                cv2.circle(big,(px,py),7,(0,220,255),2)
                cv2.putText(big,n_[:14],(px+8,py-6),font,0.38,(0,220,255),1,cv2.LINE_AA)
        canvas=np.full((700,1280,3),22,np.uint8)
        canvas[30:670,10:650]=big
        cv2.putText(canvas,f"{fam} ep{ep}  step {t}  window {r['span'][0]}-{r['span'][1]}  [{r['phase']}]",(10,22),font,0.55,(255,255,255),1,cv2.LINE_AA)
        y=[50]
        def put(lines,color,size=0.5,dy=22,x=668):
            for ln in lines:
                cv2.putText(canvas,A(ln),(x,y[0]),font,size,color,1,cv2.LINE_AA); y[0]+=dy
        put(["ORACLE SUBGOAL:"],(120,200,120)); put(wrap(gso[min(t,T-1)]),(180,255,180))
        y[0]+=6
        put(["SAM INPUT to agent-1 (per frame, <row, col>):"],(0,220,255),0.45,18)
        for j,fd in enumerate(fds):
            c=(0,220,255) if j==fdi else (150,170,170)
            for wl in wrap(fd,74)[:2]:
                put([wl],c,0.36,15)
        y[0]+=6
        put(["AGENT-1 TRAINING TARGET:"],(120,200,255))
        if r['target'].strip()=='NONE': put(["NONE"],(140,140,140),0.55)
        else:
            for ln in r['target'].split('\n'):
                c=(80,230,120) if 'completed' in ln else (60,230,230)
                put(wrap(ln),c); y[0]+=2
        y[0]+=8
        # a2 acts at the WINDOW END: show its output only on the final frame
        # of the window (judged from that frame; drives the next 16 steps)
        a2t=a2_by.get((ep,r['span'][1]))
        if a2t is not None and t>=r['span'][1]:
            put([f"AGENT-2 OUTPUT at step {r['span'][1]} (from this frame; drives next chunk):"],(255,200,120),0.42,18)
            c2=(120,255,255) if a2t.startswith('completed:') else (220,220,220)
            put(wrap(a2t,66)[:3],c2,0.4,16)
        y[0]=max(y[0]+10,430)
        put(["BANK so far:"],(150,150,150),0.45,20)
        for b in bank[-11:]: put(wrap(b,66)[:2],(200,200,160),0.4,17)
        cv2.imwrite(f"{TMP}/f{fi:05d}.png",canvas); fi+=1
        if t==r['span'][1] and (r['target'].strip()!='NONE'
                                or (a2_by.get((ep,r['span'][1])) or '').startswith('completed:')):
            for _ in range(8): cv2.imwrite(f"{TMP}/f{fi:05d}.png",canvas); fi+=1
    rc=os.system(fos.path.expandvars("ffmpeg -y -loglevel error -framerate 10 -i {TMP}/f%05d.png -c:v libx264 -pix_fmt yuv420p -crf 24 {out}"))
    shutil.rmtree(TMP,ignore_errors=True)
    print(f"{fam}: {out.split('/')[-1]} ({(time.time()-t0)/60:.1f} min)", flush=True)
print("ALL TASK VIDEOS DONE")

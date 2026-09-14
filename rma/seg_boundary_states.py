import json, pickle, collections, numpy as np
root='data/rma_preprocessed_data'
seg=json.load(open(f'{root}/meta/segments.json')); lens=json.load(open(f'{root}/meta/episode_lengths.json')); et=json.load(open(f'{root}/meta/episode_task.json'))
eps=sorted(seg, key=int); off={}; c=0
for e in eps: off[e]=c; c+=lens[e]
def load(e,s): return pickle.load(open(f'{root}/data/{off[e]+s}.pkl','rb'))
bytask=collections.defaultdict(list)
for e in eps: bytask[et[e]['task']].append(e)
rows=collections.defaultdict(list)
for t,es in sorted(bytask.items(), key=lambda kv:int(kv[0][4:])):
    for e in es[:5]:
        S=seg[e]
        for s0,L,ins in zip(S['seg_start'],S['seg_len'],S['instructions']):
            end=s0+L-1
            de=load(e,end); dm=load(e,max(s0,end-30)); ds=load(e,s0)
            w=lambda d: float(d['state'][6]-d['state'][7]); z=lambda d: float(d['state'][2]); rv=lambda d: np.asarray(d['state'][3:6],float)
            rows[ins.split()[0]].append((w(de), w(de)-w(dm), z(de)-z(ds), w(ds), float(np.linalg.norm(rv(de)-rv(ds)))))
for kind,r in rows.items():
    a=np.array(r)
    print(f'{kind:6s} n={len(a):4d} | width@end med {np.median(a[:,0]):.3f} [p10 {np.percentile(a[:,0],10):.3f} p90 {np.percentile(a[:,0],90):.3f}] open(>=0.07) {np.mean(a[:,0]>=0.07):.2f} | dwidth last30 med {np.median(a[:,1]):+.3f} | dz(end-start) med {np.median(a[:,2]):+.3f} lifted>3cm {np.mean(a[:,2]>0.03):.2f} | width@start med {np.median(a[:,3]):.3f} | |drot| end-start med {np.median(a[:,4]):.2f}')

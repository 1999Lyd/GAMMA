import os
#!/usr/bin/env python
"""Appendix figure: three more harness cases in the grammar of Fig. 3, all
from real serve traces of the 9B three-seed runs (seed 7) plus demo frames of
the same episodes. Row A: StopCube arrival-count trigger (extractor-owned
timing). Row B: PatternLock dot-route reader (REJECT + supersede). Row C:
VideoUnmaskSwap move chain admitted on displacement evidence + lookup."""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image
SP=os.path.expandvars("${GAMMA_WORK}")
GREEN="#1a8a1a"; RED="#cc2222"; DARK="#3a3028"; YEL="#ffd400"; BLUE="#2b6fd6"
plt.rcParams.update({"font.family":"DejaVu Sans"})
fig=plt.figure(figsize=(8.2,9.6),dpi=200)
def label(ax,txt,x=0.03,y=0.95):
    ax.text(x,y,txt,transform=ax.transAxes,fontsize=7,color="w",va="top",bbox=dict(facecolor=DARK,edgecolor="none",pad=1.8))
def rowtitle(y,txt):
    fig.text(0.012,y,txt,fontsize=9,fontweight="bold",color=DARK,va="top")
def verdict(x,y,word,col,sub):
    fig.text(x,y,word,fontsize=12.5,fontweight="bold",color=col,va="top")
    fig.text(x+0.105,y-0.004,sub,fontsize=6.9,va="top",linespacing=1.3)

# ================= Row A: StopCube =================
rowtitle(0.985,"A   StopCube: the press is timed by an evidence extractor, not the VLM")
ser=json.load(open(f"{SP}/sc_ep0_series.json"))
steps=ser["steps"]; dist=ser["dist"]
DG=os.path.expandvars("${ROBOMME_ROOT}/examples/robomme/runs/evaluation/qwenvl_groundsg_diag/ckpt79999/seed7/qwenvl/StopCube/ep0")
def near(step): i=min(range(len(steps)),key=lambda k:abs(steps[k]-step)); return ser["cube"][i],ser["tgt"][i]
for k,(st,cap) in enumerate([(32,"1st arrival, step 32"),(112,"2nd, step 112"),(192,"3rd, step 192"),(256,"trigger fires, step 256")]):
    ax=fig.add_axes([0.012+k*0.155,0.80,0.148,0.15]); im=np.array(Image.open(f"{DG}/step_{st}_image.png").convert("RGB"))
    r0,r1,c0,c1=40,200,30,236; ax.imshow(im[r0:r1,c0:c1],extent=[c0,c1,r1,r0]); ax.axis("off")
    cube,tgt=near(st)
    ax.plot([cube[1]],[cube[0]],"o",ms=8,mfc="none",mec=YEL,mew=1.6); ax.plot([tgt[1]],[tgt[0]],"s",ms=8,mfc="none",mec="w",mew=1.4)
    label(ax,cap)
axt=fig.add_axes([0.012,0.695,0.60,0.085])
axt.plot(steps,dist,color="0.35",lw=1); axt.axhline(20,color="0.7",lw=0.8,ls=":")
for a in (38,118,198): axt.plot([a],[min(d for s,d in zip(steps,dist) if abs(s-a)<=8)],"o",color=BLUE,ms=5)
axt.axvline(256,color=RED,lw=1.2); axt.axvline(284,color="0.4",lw=1,ls="--")
axt.text(256,max(dist)*0.92,"trigger 256",color=RED,fontsize=6.5,ha="right"); axt.text(282,max(dist)*0.70,"predicted 4th: 284",color="0.4",fontsize=6.5,ha="right")
axt.set_xlabel("control step",fontsize=6.5); axt.set_ylabel("cube–target px",fontsize=6.5); axt.tick_params(labelsize=6)
fig.text(0.64,0.945,"instruction: “press the button to stop the cube just as it\nreaches the target for the fourth time”",fontsize=6.9,va="top",linespacing=1.3)
fig.text(0.64,0.885,"evidence (detections): cube within 20 px of the target over\nsteps 32–44, 112–124, 192–204 → period 80 → 4th ≈ 284",fontsize=6.9,va="top",linespacing=1.3)
fig.text(0.64,0.83,"extractor fires the press 28 steps ahead (the measured\noracle lead): step 256 — the oracle switches at the same tick",fontsize=6.9,va="top",linespacing=1.3)
verdict(0.64,0.765,"REJECT",RED,"the reasoner's own press claims\nbefore the trigger (suppressed)")
verdict(0.64,0.715,"ADMIT",GREEN,"the extractor's press at 256;\nepisode succeeds")

# ================= Row B: PatternLock =================
rowtitle(0.655,"B   PatternLock: the writer's route words vs. the dot-route reader")
pf=np.load(f"{SP}/pl_ep1_frames.npy")
strokes=[("right",(65,128),(65,105)),("forward-left",(65,105),(83,128)),("backward-left",(83,128),(65,150))]
for k,(fi,(w,A,B)) in enumerate(zip([30,60,95],strokes)):
    ax=fig.add_axes([0.012+k*0.2,0.455,0.19,0.185]); r0,r1,c0,c1=30,120,70,190
    ax.imshow(pf[fi][r0:r1,c0:c1],extent=[c0,c1,r1,r0]); ax.axis("off")
    ax.add_patch(FancyArrowPatch((A[1],A[0]),(B[1],B[0]),arrowstyle="-|>",color=YEL,lw=1.8,mutation_scale=10))
    ax.text(c0+3,r0+8,f"stroke {k+1}: reader → {w}",fontsize=6.6,color="w",va="center",bbox=dict(facecolor=DARK,edgecolor="none",pad=1.6))
fig.text(0.64,0.635,"evidence (detections): the grid lights the route's dots in\norder; each stroke's endpoints A→B are the lit dots",fontsize=6.9,va="top",linespacing=1.3)
fig.text(0.64,0.578,"writer's demo lines (a finetuned VLM):",fontsize=6.9,va="top",color="0.2")
fig.text(0.64,0.553,"“stroke 1: move left”   “stroke 2: move forward”\n(third stroke never written)",fontsize=6.9,va="top",color=RED,style="italic",linespacing=1.3)
fig.text(0.64,0.503,"reader: right / forward-left / backward-left",fontsize=6.9,va="top",color=GREEN,fontweight="bold")
verdict(0.64,0.47,"REJECT",RED,"writer lines (wrong ×2,\nmissing ×1)")
verdict(0.64,0.425,"ADMIT",GREEN,"reader's three lines;\n10/10 episodes vs. oracle")

# ================= Row C: VideoUnmaskSwap =================
rowtitle(0.395,"C   VideoUnmaskSwap: a move chain admitted on displacement evidence, then a lookup")
df=np.load(f"{SP}/vus_ep0_demo_frames.npy")
ex=np.array(Image.open(os.path.expandvars("${ROBOMME_ROOT}/examples/robomme/runs/evaluation/qwenvl_groundsg_diag/ckpt79999/seed7/qwenvl/VideoUnmaskSwap/ep0/step_0_image.png")).convert("RGB"))
blue=[(106,88),(93,88),(82,104),(82,122)]; green=[(82,122),(95,124),(108,108),(107,88)]
panels=[(df[27],"demo: cubes visible",None),(df[45],"containers placed",None),(df[113],"after 6 moves (chain)",True),(ex,"execution, tick 1: lookup","cite")]
for k,(im,cap,mode) in enumerate(panels):
    ax=fig.add_axes([0.012+k*0.155,0.20,0.148,0.185]); r0,r1,c0,c1=60,150,55,165
    ax.imshow(im[r0:r1,c0:c1],extent=[c0,c1,r1,r0]); ax.axis("off"); label(ax,cap)
    if mode is True:
        for chain,col in ((blue,BLUE),(green,GREEN)):
            for (a,b) in zip(chain,chain[1:]):
                ax.add_patch(FancyArrowPatch((a[1],a[0]),(b[1],b[0]),arrowstyle="-|>",color=col,lw=1.4,mutation_scale=8,connectionstyle="arc3,rad=0.15"))
    if mode=="cite":
        ax.plot([87],[106],"s",ms=10,mfc="none",mec=GREEN,mew=1.8); ax.text(60,145,"cited <106, 87>",fontsize=6.4,color=GREEN,fontweight="bold",bbox=dict(facecolor="w",edgecolor="none",pad=1,alpha=0.85))
        ax.plot([87],[105],"+",ms=6,color="w",mew=1.2)
fig.text(0.64,0.38,"bank lines (writer, each admitted only with a measured\ncontainer displacement in its own window):",fontsize=6.9,va="top",linespacing=1.3)
fig.text(0.64,0.335,"blue container moved\n  <106,88>→<93,88>→<82,104>→<82,122>\ngreen container moved\n  <82,122>→<95,124>→<108,108>→<107,88>\nnow the green cube is under <107,88>",fontsize=6.1,va="top",linespacing=1.35,family="DejaVu Sans Mono")
fig.text(0.64,0.245,"reasoner (bank only): “pick up the container at <106, 87>\nthat hides the green cube”   oracle: <105, 87>",fontsize=6.9,va="top",linespacing=1.3)
verdict(0.64,0.198,"ADMIT",GREEN,"six move lines + summary;\nthe lookup matches the oracle,\nepisode succeeds")
fig.text(0.012,0.148,"blue / green arrows: the admitted move chain of each container; white square: target; yellow: detector-tracked object",fontsize=6.4,color="0.35",va="top")
fig.savefig(os.path.expandvars("${GAMMA_WORK}/figures/harness_more.pdf"),bbox_inches="tight")
fig.savefig(os.path.expandvars("${GAMMA_WORK}/figures/harness_more.png"),dpi=300,bbox_inches="tight"); fig.savefig(f"{SP}/harness_more_draft.png",dpi=160,bbox_inches="tight"); print("saved")

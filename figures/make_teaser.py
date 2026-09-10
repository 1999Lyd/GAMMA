import os
#!/usr/bin/env python
"""Teaser figure: same episode (VideoUnmaskSwap ep0), single-VLM keyframe
baseline fails vs GAMMA succeeds, with mini pipeline flows and real bank lines."""
import cv2, glob, numpy as np
from PIL import Image, ImageDraw, ImageFont

O=os.path.expandvars('${ROBOMME_ROOT}/examples/robomme/runs/evaluation')
SC=os.path.expandvars('${GAMMA_WORK}/teaser')
FD='/usr/share/fonts/truetype/dejavu'
F   = lambda s: ImageFont.truetype(f'{FD}/DejaVuSans.ttf', s)
FB  = lambda s: ImageFont.truetype(f'{FD}/DejaVuSans-Bold.ttf', s)
FM  = lambda s: ImageFont.truetype(f'{FD}/DejaVuSansMono.ttf', s)

def frames(src, idxs):
    cap=cv2.VideoCapture(src); out=[]
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES,i); ok,fr=cap.read(); assert ok, i
        img=fr[-256:, :256]                      # front view
        out.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return out

kf_src  = glob.glob(f'{O}/evalsafe_mme_frs_kf_s42/ckpt20000/seed7/videos/VideoUnmaskSwap_ep0_fail_*.mp4')[0]
our_src = glob.glob(f'{O}/wam_v18_full30_laneD/ckpt79999/seed7/oracle/videos/VideoUnmaskSwap_ep0_success_*.mp4')[0]

KF_IDX  = [2, 60, 104, 163, 206, 224]
OUR_IDX = [4, 61, 104, 148, 320, 435]
LABELS  = ['demo: cubes shown','demo: covered','demo: shuffled ('+chr(0x00d7)+'6)',
           'exec: 1st pick','exec: 2nd pick','outcome']

kf_f, our_f = frames(kf_src, KF_IDX), frames(our_src, OUR_IDX)

# ---- geometry ----
T=232          # tile size
GAP=10
LEFTW=260      # label column
N=6
W = LEFTW + N*T + (N-1)*GAP + 24
ROWH = T + 34
BANKH = 118
HDR = 46
H = HDR + ROWH + 16 + ROWH + BANKH + 14
img = Image.new('RGB',(W,H),(255,255,255))
d = ImageDraw.Draw(img)

RED=(196,30,30); GREEN=(20,140,60); GREY=(90,90,90); DGREY=(45,45,45)

def strip(y0, tiles, border, labels=True):
    for j,t in enumerate(tiles):
        x0 = LEFTW + j*(T+GAP)
        tile = Image.fromarray(cv2.resize(t,(T,T),interpolation=cv2.INTER_AREA))
        img.paste(tile,(x0,y0))
        d.rectangle([x0-1,y0-1,x0+T,y0+T], outline=border, width=3 if border!=GREY else 1)
        if labels and j<len(tiles)-1:
            d.text((x0+T//2, y0+T+6), LABELS[j], font=F(15), fill=DGREY, anchor='ma')
        if j<len(tiles)-1:
            ax=x0+T+GAP//2
            d.line([ax-2,y0+T//2,ax+3,y0+T//2],fill=(150,150,150),width=0)

def rowlabel(y0, title, col, flow):
    d.text((10,y0+2), title, font=FB(19), fill=col)
    yy=y0+34
    for i,(txt,em) in enumerate(flow):
        d.text((10,yy), txt, font=FB(14) if em else F(14), fill=DGREY if em else GREY)
        yy+=21

# header
d.text((10,8),'Same episode, same frozen '+chr(0x03c0)+'0.5 policy '+chr(0x2014)+' only the memory stage differs',
       font=FB(21), fill=(0,0,0))
d.text((W-14,14),'VideoUnmaskSwap, test episode 0', font=F(15), fill=GREY, anchor='ra')

# row 1: keyframe baseline
y1=HDR
rowlabel(y1,'Keyframe memory',RED,
   [('(single VLM, MemER-style)',0),
    ('frames '+chr(0x2192)+' keyframe selector',0),
    (chr(0x2192)+' VLM '+chr(0x2192)+' subgoal',0),
    ('opaque selection; decisive',1),('swap frames dropped;',1),('coordinates guessed',1),
    ('task success 0.38',1)])
strip(y1, kf_f, RED)
# outcome mark on last tile
x_last=LEFTW+5*(T+GAP)
d.text((x_last+T-10,y1+8), chr(0x2717), font=FB(46), fill=RED, anchor='ra')
d.text((x_last+T//2, y1+T+6), 'picks the wrong container', font=FB(15), fill=RED, anchor='ma')

# row 2: ours
y2=HDR+ROWH+16
rowlabel(y2,'GAMMA (ours)',GREEN,
   [('frames '+chr(0x2192)+' detector + writer',0),
    (chr(0x2192)+' text bank '+chr(0x2192)+' reasoner',0),
    (chr(0x2192)+' grounded subgoal',0),
    ('every line evidence-checked',1),('(propose'+chr(0x2013)+'verify);',1),('nothing discarded',1),
    ('task success 0.80 (oracle 0.99)',1)])
strip(y2, our_f, GREEN)
d.text((x_last+T-10,y2+8), chr(0x2713), font=FB(46), fill=GREEN, anchor='ra')
d.text((x_last+T//2, y2+T+6), 'picks both correct containers', font=FB(15), fill=GREEN, anchor='ma')

# bank excerpt box
y3=HDR+2*ROWH+16+8
d.rounded_rectangle([LEFTW,y3,W-24,y3+BANKH-6], radius=8, outline=GREEN, width=2, fill=(246,252,247))
bank=[('the memory bank after the demo (written lines, verbatim):',FB(14),DGREY),
 ('[event] observed: containers were placed over the cubes '+chr(0x2014)+' the blue cube at <108, 88>, the green cube at <86, 122>, ...',FM(13),(30,30,30)),
 ('[event] observed: the blue cube container moved from <106, 88> to <93, 88>     (... '+chr(0x00d7)+'6 evidence-gated move lines)',FM(13),(30,30,30)),
 ('[event] observed: now the blue cube is under the container at <82, 122>, the green cube is under the container at <107, 88>, ...',FM(13),(30,30,30)),
 ('reasoner '+chr(0x2192)+' "pick up the container at <106, 87> that hides the green cube",  then  "... at <82, 122> that hides the blue cube"',FB(13),GREEN)]
yy=y3+8
for txt,fnt,col in bank:
    d.text((LEFTW+12,yy),txt,font=fnt,fill=col); yy+=21
d.text((10,y3+8),'why it succeeds:',font=FB(15),fill=DGREY)
d.text((10,y3+30),'the shuffle survives as',font=F(14),fill=GREY)
d.text((10,y3+50),'auditable text, not',font=F(14),fill=GREY)
d.text((10,y3+70),'dropped pixels',font=F(14),fill=GREY)

img.save(f'{SC}/teaser.png')
print('saved', f'{SC}/teaser.png', img.size)

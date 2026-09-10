import os
#!/usr/bin/env python
"""Patch Nano-Banana compare.png: crop title, fix row labels, replace the
invented bank lines and reasoner bubble with verbatim episode text."""
from PIL import Image, ImageDraw, ImageFont

SRC=os.path.expandvars('${GAMMA_WORK}/figures/compare.png')
OUT=os.path.expandvars('${GAMMA_WORK}/figures/teaser.png')
FD='/usr/share/fonts/truetype/dejavu'
FB=lambda s: ImageFont.truetype(f'{FD}/DejaVuSans-Bold.ttf',s)
F=lambda s: ImageFont.truetype(f'{FD}/DejaVuSans.ttf',s)
FR=lambda s: ImageFont.truetype(f'{FD}/DejaVuSans.ttf',s)
FM=lambda s: ImageFont.truetype(f'{FD}/DejaVuSansMono.ttf',s)

im=Image.open(SRC).convert('RGB')
d=ImageDraw.Draw(im)
GREEN=(46,125,50); DARK=(35,35,35); GREY=(100,100,100); BG=(236,244,235)

# --- 1. row label 1 (fits left of the "Memory:" heading at x~1010) ---
d.rectangle([60,240,1000,325], fill=(255,255,255))
d.text((66,250),'Single-VLM Keyframe Memory',font=FB(36),fill=DARK)
w=d.textlength('Single-VLM Keyframe Memory',font=FB(36))
d.text((66+w+18,258),'(MemER-style)',font=FR(28),fill=GREY)

# --- 2. row label 2: remove "Bottom:" ---
d.rectangle([60,810,640,890], fill=(255,255,255))
d.text((66,820),'GAMMA (Ours)',font=FB(36),fill=DARK)

# --- 3. bank card list area: verbatim lines ---
X0,Y0,X1,Y1=966,1046,1528,1266
d.rectangle([X0,Y0,X1,Y1], fill=BG)
lines=[(1,'[event] containers placed over the cubes',0),
       (1,'[event] blue cube container moved from',0),
       (0,'        <106, 88> to <93, 88>  (... x6)',0),
       (1,'[event] now the blue cube is under',1),
       (0,'        the container at <82, 122>',1)]
lh=42; yy=Y0+14
for i,(chk,t,hl) in enumerate(lines):
    y=yy+i*lh
    if hl: d.rectangle([X0+6,y-4,X1-8,y+lh-10], fill=(200,229,192))
for i,(chk,t,hl) in enumerate(lines):
    y=yy+i*lh
    if chk: d.text((X0+12,y+2),'✓',font=FB(24),fill=GREEN)
    d.text((X0+44,y),t,font=FM(20),fill=(30,30,30))

# --- 4. reasoner bubble: verbatim subgoal ---
BX0,BY0,BX1,BY1=2010,876,2364,1104
d.rectangle([BX0-6,BY0-6,BX1+6,BY1+6], fill=(255,255,255))
d.rounded_rectangle([BX0,BY0,BX1,BY1], radius=26, fill=BG, outline=(120,140,120), width=3)
for j,t in enumerate(['pick up the container','at <82, 122> that','hides the blue cube']):
    d.text(((BX0+BX1)//2,BY0+42+j*50),t,font=FB(27),fill=GREEN,anchor='ma')

# --- 4b. policy box: finetuned-once-then-fixed, not frozen ---
d.rectangle([2445,680,2665,885], fill=(213,213,213))
FPB=FB(46)
for j,t in enumerate(['Fixed','pi-0.5','policy']):
    d.text((2553,690+j*62),t,font=FPB,fill=(51,51,50),anchor='ma')

# --- 5. headline bars band (MemoryVLA-style) + footer on extended canvas ---
BARH=310
big=Image.new('RGB',(2752,1536+BARH),(255,255,255))
big.paste(im.crop((0,0,2752,1416)),(0,0))
d=ImageDraw.Draw(big)
d.line([70,1436,2682,1436],fill=(225,225,225),width=2)
bars=[('pi-0.5 e2e',17.9,(160,160,160),0),('MemER',42.4,(210,140,140),0),
      ('FrameSamp+Modul',44.5,(210,140,140),0),('GAMMA (ours)',66.0,(46,125,50),0),
      ('GroundSG oracle',84.1,(200,200,200),1)]
X0,XW,BW,BASE,SC=560,2100,210,1700,2.4   # baseline y, px per point
d.text((80,1500),'average success on the',font=FB(30),fill=(45,45,45))
d.text((80,1540),'16-task suite (%)',font=FB(30),fill=(45,45,45))
step=XW//len(bars)
for i,(name,v,col,priv) in enumerate(bars):
    cx=X0+i*step+step//2
    h=int(v*SC); y0=BASE-h
    if priv:
        d.rectangle([cx-BW//2,y0,cx+BW//2,BASE],fill=(238,238,238),outline=(150,150,150),width=3)
    else:
        d.rectangle([cx-BW//2,y0,cx+BW//2,BASE],fill=col)
    d.text((cx,y0-38),f'{v:.1f}',font=FB(30),fill=(45,45,45) if not priv else (120,120,120),anchor='ma')
    d.text((cx,BASE+10),name,font=F(24),fill=(70,70,70),anchor='ma')
    if priv:
        d.text((cx,BASE+42),'(privileged)',font=F(22),fill=(150,150,150),anchor='ma')
d.line([X0-40,BASE-int(84.1*SC),X0+XW,BASE-int(84.1*SC)],fill=(170,170,170),width=2)
d.line([X0-40,BASE,X0+XW-260,BASE],fill=(90,90,90),width=3)

FI=ImageFont.truetype(f'{FD}/DejaVuSans-Oblique.ttf',34)
fy=1536+BARH-100
d.text((2752//2,fy),
  'same frames, same fixed policy '+chr(0x2014)+' in between, two agents collaborate through a text bank,',
  font=FI,fill=(110,110,110),anchor='ma')
d.text((2752//2,fy+44),
  'and a propose'+chr(0x2013)+'verify harness audits every claim before it becomes memory',
  font=FI,fill=(110,110,110),anchor='ma')

# --- 6. crop the big title band ---
big=big.crop((0,190,2752,1536+BARH))
big.save(OUT)
print('saved',OUT,big.size)

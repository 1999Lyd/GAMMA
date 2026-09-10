import os
#!/usr/bin/env python
"""Right-hand panel of the harness-cases figure: how DETECTION evidence and a
deterministic reader correct the VLM's direction judgement (RouteStick,
benchmark test episode 0, stroke 3). Flow: VLM claim -> detector evidence
-> reader computation -> verdict. Data: demo frames from the QwenVL diag
lane; VLM lines from the 0.8B harness-off serve trace of the same episode;
reader quantities recomputed on exactly the frames shown."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

SP = os.path.expandvars("${GAMMA_WORK}")
frames = np.load(f"{SP}/rs_ep0_frames.npy")
strokes = json.load(open(f"{SP}/rs_ep0_strokes.json"))
s = strokes[2]                       # stroke 3: VLM wrong on side AND direction
A, B = np.array(s["A"]), np.array(s["B"])
pts = np.array(s["pts"]); ch = B - A
cross = ch[0]*(pts[:, 1]-A[1]) - ch[1]*(pts[:, 0]-A[0])
med = float(np.median(cross))

GREEN = "#1a8a1a"; RED = "#cc2222"; DARK = "#3a3028"; YEL = "#ffd400"
POS = "#2b6fd6"; NEG = "#e06020"
plt.rcParams.update({"font.family": "DejaVu Sans"})
fig = plt.figure(figsize=(8.05, 3.60), dpi=200)

def box(x, y, w, h, title, fc="#f7f4ef"):
    fig.patches.append(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.01",
                       fc=fc, ec="#c9c2b8", lw=0.8, transform=fig.transFigure, zorder=-5))
    fig.text(x + 0.008, y + h - 0.012, title, fontsize=8.6, fontweight="bold", va="top", color=DARK)

def arrow(x0, x1, y):
    fig.patches.append(FancyArrowPatch((x0, y), (x1, y), transform=fig.transFigure,
                       arrowstyle="-|>", mutation_scale=12, color=DARK, lw=1.2))

CROP = (18, 150, 55, 150)   # x0,x1,y0,y1 around the stroke
x0, x1, y0, y1 = CROP
fr = frames[s["last_frame"]][y0:y1, x0:x1]

# ---- (1) VLM claim -------------------------------------------------------
box(0.01, 0.30, 0.215, 0.66, "1  writer (VLM) claims")
ax1 = fig.add_axes([0.02, 0.50, 0.195, 0.36]); ax1.set_zorder(5); ax1.imshow(fr, extent=[x0, x1, y1, y0]); ax1.axis("off")
ax1.text(x0+3, y0+9, "stroke 3", fontsize=7.5, color="w", va="center", bbox=dict(facecolor=DARK, edgecolor="none", pad=1.8))
fig.text(0.02, 0.49, "“move to the nearest", fontsize=7.8, style="italic", va="top")
fig.text(0.02, 0.445, "right target, circling", fontsize=7.8, style="italic", va="top", color=RED, fontweight="bold")
fig.text(0.02, 0.40, "clockwise”", fontsize=7.8, style="italic", va="top", color=RED, fontweight="bold")
fig.text(0.02, 0.356, "a finetuned 9B VLM cannot read\nthe turning sense from frames", fontsize=6.4, color="0.35", va="top", linespacing=1.25)
arrow(0.23, 0.262, 0.62)

# ---- (2) detector evidence + reader ---------------------------------------
box(0.262, 0.30, 0.462, 0.66, "2  detector evidence  →  deterministic reader")
ax2 = fig.add_axes([0.272, 0.39, 0.215, 0.49]); ax2.set_zorder(5); ax2.imshow(fr, extent=[x0, x1, y1, y0]); ax2.axis("off")
ax2.set_xlim(x0, x1); ax2.set_ylim(y1, y0)
ax2.scatter(pts[cross > 0, 0], pts[cross > 0, 1], s=9, c=POS, edgecolors="none", zorder=3)
ax2.scatter(pts[cross <= 0, 0], pts[cross <= 0, 1], s=9, c=NEG, edgecolors="none", zorder=3)
ax2.add_patch(FancyArrowPatch(A, B, arrowstyle="-|>", color=YEL, lw=1.8, mutation_scale=10, zorder=4))
ax2.plot([A[0]], [A[1]], "o", ms=7, mfc="none", mec=YEL, mew=1.6, zorder=4)
ax2.plot([B[0]], [B[1]], "s", ms=6, mfc="none", mec=YEL, mew=1.4, zorder=4)
ax2.text(A[0]-4, A[1]+9, "A", color=YEL, fontsize=8, fontweight="bold", zorder=5)
ax2.text(B[0]+2, B[1]-4, "B", color=YEL, fontsize=8, fontweight="bold", zorder=5)
sgn = 1 if med > 0 else -1
ax2.add_patch(FancyArrowPatch(A, B, connectionstyle=f"arc3,rad={0.55*sgn}", arrowstyle="-|>",
                              color="w", lw=1.4, mutation_scale=11, zorder=5, alpha=0.95))
fig.text(0.50, 0.86, "detected in the demo frames:", fontsize=7.2, va="top", color="0.2")
fig.text(0.50, 0.815, "A  start pad (marker)\nB  next pad\n•  ink left by the stroke\n    (65 points)",
         fontsize=6.8, va="top", linespacing=1.3)
fig.text(0.50, 0.64, "reader asks two questions:", fontsize=7.2, va="top", color="0.2")
fig.text(0.50, 0.598, "which side of the line A→B\ndoes the ink bulge to?", fontsize=6.8, va="top", linespacing=1.3)
fig.text(0.50, 0.515, "61 of 65 points on the ccw side\n⇒ counterclockwise", fontsize=6.8, va="top",
         linespacing=1.3, color=GREEN, fontweight="bold")
fig.text(0.50, 0.43, "where is B relative to A?", fontsize=6.8, va="top")
fig.text(0.50, 0.39, "to the robot's left\n⇒ left target", fontsize=6.8, va="top", color=GREEN, fontweight="bold", linespacing=1.3)
fig.text(0.272, 0.375, "blue: ink on the ccw side of A→B\norange: cw side", fontsize=6.0, color="0.35", va="top", linespacing=1.2)
arrow(0.728, 0.752, 0.62)

# ---- (3) verdict ------------------------------------------------------------
box(0.757, 0.30, 0.233, 0.66, "3  verdict")
fig.text(0.767, 0.86, "REJECT", fontsize=13, fontweight="bold", color=RED, va="top")
fig.text(0.767, 0.77, "the writer's line: both\nwords contradict the ink", fontsize=7.0, va="top", linespacing=1.35)
fig.text(0.767, 0.62, "ADMIT", fontsize=13, fontweight="bold", color=GREEN, va="top")
fig.text(0.767, 0.53, "the reader's line:\n“move to the nearest\nleft target, circling\ncounterclockwise”",
         fontsize=7.0, va="top", linespacing=1.35, color=GREEN)
fig.text(0.767, 0.375, "reader correct on 33/33\nheld-out strokes", fontsize=6.2, color="0.35", va="top", linespacing=1.2)

# ---- footer: writer vs reader on every stroke of this episode -------------
W = [("left", "clockwise"), ("right", "counterclockwise"), ("right", "clockwise")]
AB = {"clockwise": "cw", "counterclockwise": "ccw"}
fig.text(0.01, 0.22, "writer vs. reader, all strokes of this episode:", fontsize=7.2, color="0.2", va="top")
for k, st in enumerate(strokes):
    wl, wd = W[k]; ok = (wl == st['lat'] and wd == st['dir']); xk = 0.30 + k * 0.235
    fig.text(xk, 0.22, f"stroke {k+1}", fontsize=7.0, va="top", color="0.2")
    fig.text(xk, 0.165, f"writer  {wl}/{AB[wd]}", fontsize=7.0, va="top", color=(GREEN if ok else RED))
    fig.text(xk, 0.11, f"reader  {st['lat']}/{AB[st['dir']]}   {'✓' if ok else '✗'}", fontsize=7.0, va="top")
fig.text(0.01, 0.04, "direction is a percept the VLM cannot judge: wrong on 2 of 3 strokes here; the reader's lines replace its words",
         fontsize=6.8, color="0.35", va="top")

fig.savefig(os.path.expandvars("${GAMMA_WORK}/figures/harness_rs.png"), dpi=300)
fig.savefig(os.path.expandvars("${GAMMA_WORK}/figures/harness_rs.pdf"))
fig.savefig(f"{SP}/harness_rs_draft.png", dpi=200)
print("saved; stroke3 median=%.1f n=%d pos=%d" % (med, len(cross), int((cross > 0).sum())))

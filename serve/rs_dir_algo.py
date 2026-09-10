#!/usr/bin/env python
"""v18 RouteStick demo-direction extraction (serve algorithm).

Eval demo prefix = K back-to-back 50-frame circling strokes.  The red
marker stands at the CURRENT stroke's start pad (and teleports to the next
pad when the stroke ends; after the last stroke it returns to the route
start for exec).  The stick draws a fast-fading white ink wisp along the
arc: its new-pixels, filtered to ink-on-wood, are samples of the path.
Orientation = sign of the median cross product of those samples w.r.t. the
directed chord (image coords, y down): cw <=> median < 0.
"""
import numpy as np
from scipy import ndimage

STROKE = 50


def marker_of(a):
    m = (a[..., 0] > 150) & (a[..., 1] < 70) & (a[..., 2] < 70)
    md = ndimage.binary_dilation(m, iterations=2)
    lab, n = ndimage.label(md)
    if n == 0:
        return None
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    k = int(np.argmax(sizes)) + 1
    if sizes[k - 1] < 6:
        return None
    ys, xs = np.nonzero((lab == k) & m)
    return float(xs.mean()), float(ys.mean())


def _wm(a):
    a = a.astype(np.int16)
    mx = a.max(axis=2)
    mn = a.min(axis=2)
    m = (mx > 185) & ((mx - mn) < 32)
    m[:40, :] = False
    return m


def ink_pixels(arrs, a, b, A, stride=2, maxcomp=150, dil=3, rad=55):
    """New white pixels that replaced WOOD (prev frame strongly r>b), in
    small components (arm slivers merge into the giant arm blob under
    dilation), within rad of the stroke's start pad."""
    pts, pm, prev = [], None, None
    for f in range(a, b, stride):
        arr = arrs[f]
        m = _wm(arr)
        if pm is not None:
            new = m & ~pm
            if new.any():
                md = ndimage.binary_dilation(m, iterations=dil)
                lab, n = ndimage.label(md)
                sizes = ndimage.sum(m, lab, range(1, n + 1))
                ys, xs = np.nonzero(new)
                for y, x in zip(ys, xs):
                    c = lab[y, x]
                    if (c and sizes[c - 1] <= maxcomp and y > 60
                            and int(prev[y, x, 0]) - int(prev[y, x, 2]) >= 40
                            and (x - A[0])**2 + (y - A[1])**2 <= rad**2):
                        pts.append((float(x), float(y)))
        pm, prev = m, arr
    return pts


def stroke_dirs(arrs):
    """arrs: demo frames (stride 1, RGB uint8 HxWx3).  Returns list of
    per-stroke dicts {A, B, dir, lat, med, n} (None where unmeasurable).
    K by rounding: the serve client drops the trailing exec-init frame, so
    demos arrive as exactly 50K frames ((n-1)//50 loses the last stroke)."""
    K = max(int(round(len(arrs) / STROKE)), 0)
    mk = []
    for k in range(K + 1):
        a = k * STROKE
        ps = [marker_of(arrs[f]) for f in range(a, min(a + 40, len(arrs)), 4)]
        ps = [p for p in ps if p]
        mk.append(np.median(np.array(ps), axis=0) if ps else None)
    steps = [mk[k + 1] - mk[k] for k in range(K - 1)
             if mk[k] is not None and mk[k + 1] is not None]
    line = None
    if steps:
        v = steps[0] / np.linalg.norm(steps[0])
        line = (v, float(np.median([np.linalg.norm(s) for s in steps])))
    out = []
    for k in range(K):
        A = mk[k]
        if A is None:
            out.append(None)
            continue
        # drop the trailing frames of the LAST stroke (exec reset motion)
        b = min((k + 1) * STROKE, len(arrs) - 1) - (4 if k == K - 1 else 0)
        pts = ink_pixels(arrs, k * STROKE, b, A)
        if len(pts) < 6:
            out.append(None)
            continue
        if k < K - 1 and mk[k + 1] is not None:
            B = mk[k + 1]
        elif line is not None:
            v, spacing = line
            proj = [(x - A[0]) * v[0] + (y - A[1]) * v[1] for x, y in pts]
            B = A + v * spacing * np.sign(np.median(proj))
        else:
            P = np.array(pts)
            dv = P.mean(axis=0) - A
            nv = np.linalg.norm(dv)
            if nv < 4:
                out.append(None)
                continue
            B = A + dv / nv * 35
        ch = B - A
        if np.linalg.norm(ch) < 8:
            out.append(None)
            continue
        crosses = [ch[0] * (y - A[1]) - ch[1] * (x - A[0]) for x, y in pts]
        med = float(np.median(crosses))
        out.append({"A": [round(float(A[0]), 1), round(float(A[1]), 1)],
                    "B": [round(float(B[0]), 1), round(float(B[1]), 1)],
                    "dir": "clockwise" if med < 0 else "counterclockwise",
                    "lat": "left" if ch[0] > 0 else "right",
                    "med": round(med, 1), "n": len(pts)})
    return out


# ---------------------------------------------------------------------------
# PatternLock (v18): the demo lights the route's dots in sequence -- the
# start dot is lit at frame 0, each NEXT dot lights up ~6-8 frames before
# the current stroke ends, and the previous dot extinguishes afterwards.
# The dot chain therefore gives every stroke's endpoints, INCLUDING the
# fore/aft component that no arm/ink channel could measure (10/10 GT
# episodes, all words exact; consecutive same-word strokes -- which fuse
# into one gso run -- are recovered separately).
# Calibration (image coords): left = +col, right = -col, forward = +row,
# backward = -row.

def _red_dots(a):
    m = (a[..., 0] > 150) & (a[..., 1] < 70) & (a[..., 2] < 70)
    md = ndimage.binary_dilation(m, iterations=2)
    lab, n = ndimage.label(md)
    out = []
    for c in range(1, n + 1):
        ys, xs = np.nonzero((lab == c) & m)
        if len(ys) >= 5:
            out.append((float(xs.mean()), float(ys.mean())))
    return out


def pl_route(arrs, stride=2):
    """Returns (strokes, appear): strokes[k] = {A, B, word}; appear[i] =
    first frame at which route dot i was lit (appear[k+1] ~ stroke k's
    end minus ~7 frames)."""
    route, appear = [], []
    for i in range(0, len(arrs), stride):
        for (x, y) in _red_dots(arrs[i]):
            if all((x - rx) ** 2 + (y - ry) ** 2 > 8 ** 2 for rx, ry in route):
                route.append((x, y))
                appear.append(i)
    strokes = []
    for k in range(len(route) - 1):
        (ax, ay), (bx, by) = route[k], route[k + 1]
        dc, dr = bx - ax, by - ay
        fore = "forward" if dr >= 9 else ("backward" if dr <= -9 else "")
        lat = "left" if dc >= 10 else ("right" if dc <= -10 else "")
        word = "-".join(w for w in (fore, lat) if w)
        if not word:
            strokes.append(None)
            continue
        strokes.append({"A": [round(ay), round(ax)], "B": [round(by), round(bx)],
                        "word": word})
    return strokes, appear

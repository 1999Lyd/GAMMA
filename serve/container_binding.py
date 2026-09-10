#!/usr/bin/env python
"""Per-frame bound container positions, for LABEL construction only.

Replays annot_pipeline COVER/MOVE_START/MOVE_END events and tracks each
bound container BETWEEN events by per-frame nearest-neighbour steps to real
detections.  Radii follow the measured geometry (4-13px container motion per
4-step frame gap vs >=27px minimum inter-container separation), so a 20px
step radius follows any move without reaching a neighbouring container.
Inputs to the model keep raw 'container' labels -- binding is the VLM's
judgment, expressed in its OUTPUT lines; this module only supplies the
grounded coordinates those target lines cite.
"""
import math

TRACK_R = 20.0   # per-frame-gap NN tracking radius (max real step 13px)


def replay(events, real, grid):
    """events: annot_pipeline build_events() output [(t, kind, payload)].
    real: {frame: [(label, x, y), ...]}   grid: sorted frame indices.
    Returns (bound, moves):
      bound: {frame: {color: (x, y)}} bound container position per frame,
             always the coordinates of an actual detection when one matched
      moves: [(t_end, color, (sx, sy), (dx, dy))] whole-move records
    """
    ev_by_t = {}
    for t, k, p in events:
        ev_by_t.setdefault(t, []).append((k, p))
    pos, moving = {}, {}
    bound, moves = {}, []
    for t in grid:
        for k, p in ev_by_t.get(t, []):
            if k == "COVER":
                for c, (xy, site) in p.items():
                    pos[c] = tuple(site)
            elif k == "MOVE_START":
                c = p[0]
                if c in pos:
                    moving.setdefault(c, pos[c])
            elif k == "MOVE_END":
                c, dest = p[0], tuple(p[1])
                src = moving.pop(c, pos.get(c))
                if src is not None:
                    moves.append((t, c, src, dest))
                pos[c] = dest
        # NN tracking step: follow each bound container to the nearest
        # container detection, greedily by distance, one detection per color
        cdets = [(x, y) for n, x, y in real[t] if "container" in n]
        cand = []
        for c, (px, py) in pos.items():
            for j, (x, y) in enumerate(cdets):
                d = math.hypot(px - x, py - y)
                if d <= TRACK_R:
                    cand.append((d, c, j))
        used_c, used_j = set(), set()
        for d, c, j in sorted(cand):
            if c in used_c or j in used_j:
                continue
            used_c.add(c); used_j.add(j)
            pos[c] = cdets[j]
        bound[t] = dict(pos)
    return bound, moves


import re as _re
_COVER = _re.compile(r"(\w+)_cube_container<\s*(\d+)\s*,\s*(\d+)\s*>")
_MOVED = _re.compile(r"the (\w+) cube container moved from <\s*\d+\s*,\s*\d+\s*> "
                     r"to <\s*(\d+)\s*,\s*(\d+)\s*>")


def fold_state(bank_lines):
    """Current container position per cube color, folded deterministically
    from the writer's own cover + moved-from lines (v16 formats)."""
    st = {}
    for ln in bank_lines:
        l = ln.lower()
        if "were placed over the cubes" in l:
            for c, x, y in _COVER.findall(ln):
                st[c] = (int(x), int(y))
        else:
            m = _MOVED.search(l)
            if m:
                st[m.group(1)] = (int(m.group(2)), int(m.group(3)))
    return st

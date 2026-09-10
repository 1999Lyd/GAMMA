"""Deterministic VideoPlaceButton demo reader (2026-09-10).

Evidence: the demo cube's detected track. The demo is pick -> drop -> press
-> pick -> drop -> pick -> drop(table); the instruction asks for the pad
"right before" or "right after" the press, i.e. the 1st or 2nd pad the cube
comes to rest on. A pad is identified by where the cube rests (its own
detection is occluded by the cube while it sits there), so pad anchors are
clustered over the whole demo.

vpb_target(track, targets, instr) -> (pad or None, drops)
  track   : list of (frame_idx, (r,c) or None) for the task-colour cube
  targets : list of (r,c) detections of class 'target' over all demo frames
"""
import re

def _d(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def cluster(pts, radius=10, min_n=2):
    cl = []
    for p in pts:
        for c in cl:
            if _d(c[0], p) <= radius:
                c[1] += 1; break
        else:
            cl.append([p, 1])
    return [c[0] for c in cl if c[1] >= min_n]

def rest_segments(track, still_px=6, min_frames=20, gap_frames=12):
    """Runs of frames where the cube is detected and does not move; a run
    ends when the cube moves > still_px or is undetected for > gap_frames."""
    segs = []; cur = None; last_seen = None
    for fi, p in track:
        if p is None:
            if cur and last_seen is not None and fi - last_seen > gap_frames:
                segs.append(cur); cur = None
            continue
        if cur and _d(cur["pos"], p) <= still_px:
            cur["end"] = fi
        else:
            if cur:
                segs.append(cur)
            cur = {"start": fi, "end": fi, "pos": p}
        last_seen = fi
    if cur:
        segs.append(cur)
    return [s for s in segs if s["end"] - s["start"] >= min_frames]

def vpb_target(track, targets, instr, pad_radius=20):
    pads = cluster(targets)
    segs = rest_segments(track)
    if not segs:
        return None, []
    # drop k = k-th rest after the initial rest; consecutive rests at the same
    # spot (detector flicker) are merged
    drops = []
    for s in segs[1:]:
        if drops and _d(drops[-1]["pos"], s["pos"]) <= 14:
            continue
        pad = min(pads, key=lambda t: _d(t, s["pos"])) if pads else None
        drops.append({"frame": s["start"], "pos": s["pos"],
                      "pad": pad if pad is not None and _d(pad, s["pos"]) <= pad_radius else None})
    il = (instr or "").lower()
    # VPB: "right before/after the button was pressed" = 1st/2nd pad landing;
    # VPO: "the first/second/third/fourth target it was previously placed on"
    # = k-th pad landing (corpus: 73/90 = 81% oracle-pad recovery).
    if "before the button" in il:
        k = 0
    elif "after the button" in il:
        k = 1
    else:
        words = {"first": 0, "second": 1, "third": 2, "fourth": 3, "fifth": 4}
        m = re.search(r"the (first|second|third|fourth|fifth) target", il)
        k = words[m.group(1)] if m else 0
    pad_drops = [d for d in drops if d["pad"] is not None]
    ans = pad_drops[k]["pad"] if len(pad_drops) > k else None
    return ans, drops

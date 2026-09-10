#!/usr/bin/env python
"""Annotated videos of every FAILED closed-loop episode, per task, for
by-eye diagnosis.

For each lane/task/episode with progress.json == False:
  * the recorded eval video (512x512, one frame per control step; video-demo
    tasks carry the demo prefix of n_demo frames first),
  * the serve trace of that episode (the occurrence whose length matches the
    video), giving per tick: window detections, writer line, bank, our
    subgoal, oracle subgoal.
Output: <out>/<task>/<task>_ep<N>_annotated.mp4 with, per frame,
  left  : the frame + last-window detections (red dots), our cited coordinate
          (cyan circle) and the oracle's (yellow circle), step/tick header;
  right : OURS (green = content matches oracle, red = differs), ORACLE,
          WRITER line, and the bank (last lines),
plus <out>/index.md: per task, each failed episode with tick count, first
divergence tick and the diverging pair, and the video path.

Usage: annotate_failed_eps.py --arm 0p8b_on20 --lanes W,X,Y,Z --out DIR [--tasks A,B]
Env WORKERS (default 16).
"""
import argparse, glob, json, os, re, textwrap
from multiprocessing import Pool

import cv2
import numpy as np

E = os.path.expandvars("${ROBOMME_ROOT}/examples/robomme/runs/evaluation")
TR = os.path.expandvars("${GAMMA_DATA}/traces")
H = 16
PANEL_W = 560
FONT = cv2.FONT_HERSHEY_SIMPLEX


def strip(s):
    return re.sub(r"\s*at <[^>]*>", "", s or "").strip().lower()


def coord(s):
    m = re.search(r"<\s*(\d+)\s*,\s*(\d+)\s*>", s or "")
    return (int(m.group(1)), int(m.group(2))) if m else None


def load_occurrences(trace_path):
    """(task, ep) -> list of {'reset', 'ticks'} in file order (lane files are appended across runs)."""
    occ = {}
    cur = None
    with open(trace_path) as f:
        for line in f:
            r = json.loads(line)
            k = r.get("kind")
            if k == "reset":
                cur = {"reset": r, "ticks": []}
                occ.setdefault((r["task"], r["ep"]), []).append(cur)
            elif k == "tick" and cur is not None:
                cur["ticks"].append(r)
            elif k == "end":
                cur = None
    return occ


def pick_occurrence(occs, n_frames):
    """the occurrence whose (demo + execution) length best matches the video."""
    def length(o):
        n_demo = o["reset"].get("n_demo") or 0
        last = o["ticks"][-1]["step"] if o["ticks"] else 0
        return n_demo + last + H
    return min(occs, key=lambda o: abs(length(o) - n_frames))


def put_wrapped(img, text, x, y, width_chars, color, scale=0.42, lh=15, max_lines=None):
    lines = []
    for para in (text or "").split("\n"):
        lines += textwrap.wrap(para, width_chars) or [""]
    if max_lines:
        lines = lines[:max_lines]
    for ln in lines:
        cv2.putText(img, ln, (x, y), FONT, scale, color, 1, cv2.LINE_AA)
        y += lh
    return y


def annotate(job):
    video, out_path, occ, task, ep = job
    cap = cv2.VideoCapture(video)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W0, H0 = int(cap.get(3)), int(cap.get(4))
    # eval video = composite: text panel (top half), agentview 256x256 at
    # bottom-left, wrist view bottom-right. We enlarge the agentview crop x2.
    sc = 2.0
    W, Hh = 512, max(512, H0)
    reset, ticks = occ["reset"], occ["ticks"]
    n_demo = reset.get("n_demo") or 0
    instr = reset.get("instruction", "")
    bank_demo = reset.get("bank_after_demo") or []
    steps = [t["step"] for t in ticks]
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W + W0 + PANEL_W, Hh))
    first_div = None
    for t in ticks:
        if strip(t["oracle"]) != strip(t["agent2_out"]):
            first_div = t
            break
    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        canvas = np.zeros((Hh, W + W0 + PANEL_W, 3), np.uint8)
        agent = frame[H0 - 256:H0, :256]                          # 256x256 agentview crop (bottom-left)
        canvas[:512, :W] = cv2.resize(agent, (W, 512), interpolation=cv2.INTER_NEAREST)
        canvas[:H0, W:W + W0] = frame                             # original composite (wrist + its text)
        panel = canvas[:, W + W0:]
        panel[:] = (28, 28, 28)
        step = fi - n_demo
        # current tick = last tick whose step <= current execution step
        ti = None
        if step >= 0 and steps:
            ti = max([i for i, s in enumerate(steps) if s <= step] or [None]) if any(s <= step for s in steps) else None
        y = 22
        y = put_wrapped(panel, f"{task} ep{ep}   frame {fi}/{n_frames}", 8, y, 60, (255, 255, 255), 0.5, 18)
        y = put_wrapped(panel, "instr: " + instr, 8, y, 78, (200, 200, 200), 0.38, 13, 3)
        if step < 0 or ti is None:
            y = put_wrapped(panel, "DEMO / PRE-TICK PHASE", 8, y + 4, 60, (0, 200, 255), 0.5, 18)
            y = put_wrapped(panel, "bank after demo:", 8, y, 60, (180, 180, 180), 0.42, 15)
            for ln in bank_demo[-10:]:
                y = put_wrapped(panel, "- " + ln.split("  [sam]")[0], 8, y, 78, (180, 180, 180), 0.36, 12, 2)
            cv2.putText(canvas, f"step {step}" if step >= 0 else f"demo {fi}/{n_demo}", (6, 18), FONT, 0.5, (0, 200, 255), 1, cv2.LINE_AA)
        else:
            t = ticks[ti]
            match = strip(t["oracle"]) == strip(t["agent2_out"])
            col_ours = (80, 220, 80) if match else (60, 60, 255)
            cv2.putText(canvas, f"step {step}  tick {t['tick']}", (6, 18), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            # detections of the last window frame
            dets = t.get("dets") or []
            last = dets[-1][1:] if dets else []
            for d in last:
                m = re.match(r"(\w+)<(\d+), (\d+)>", d)
                if not m:
                    continue
                r, c = int(m.group(2)), int(m.group(3))
                cv2.circle(canvas, (int(c * sc), int(r * sc)), 4, (0, 0, 255), -1)
                cv2.putText(canvas, m.group(1)[:8], (int(c * sc) + 5, int(r * sc) - 3), FONT, 0.33, (0, 0, 255), 1, cv2.LINE_AA)
            co = coord(t["agent2_out"])
            if co:
                cv2.circle(canvas, (int(co[1] * sc), int(co[0] * sc)), 12, (255, 255, 0), 2)
            cg = coord(t["oracle"])
            if cg:
                cv2.circle(canvas, (int(cg[1] * sc), int(cg[0] * sc)), 16, (0, 255, 255), 2)
            y += 4
            y = put_wrapped(panel, "OURS: " + (t["agent2_out"] or ""), 8, y, 66, col_ours, 0.45, 16, 3)
            y = put_wrapped(panel, "ORACLE: " + (t["oracle"] or ""), 8, y, 66, (0, 255, 255), 0.45, 16, 3)
            y = put_wrapped(panel, "WRITER: " + ((t["agent1_out"] or "").split("  [sam]")[0] or "(none)"), 8, y + 2, 78, (255, 200, 120), 0.38, 13, 4)
            y = put_wrapped(panel, f"BANK ({len(t['bank'])} lines, last 10):", 8, y + 4, 60, (180, 180, 180), 0.42, 15)
            for ln in t["bank"][-10:]:
                y = put_wrapped(panel, "- " + ln.split("  [sam]")[0], 8, y, 78, (180, 180, 180), 0.35, 12, 2)
                if y > Hh - 30:
                    break
            cv2.putText(panel, "cyan=our coord  yellow=oracle coord  red=detections", (8, Hh - 8), FONT, 0.36, (150, 150, 150), 1, cv2.LINE_AA)
        writer.write(canvas)
        fi += 1
    writer.release()
    cap.release()
    # OpenCV's mp4v (MPEG-4 part 2) does not play in VS Code / browsers: re-encode to H.264
    import subprocess
    tmp = out_path + ".h264.mp4"
    if subprocess.run([os.path.expandvars("ffmpeg"), "-v", "error", "-y", "-i", out_path, "-c:v", "libx264",
                       "-pix_fmt", "yuv420p", "-crf", "23", "-movflags", "+faststart", tmp]).returncode == 0:
        os.replace(tmp, out_path)
    return {"task": task, "ep": ep, "ticks": len(ticks), "frames": n_frames, "n_demo": n_demo,
            "video": out_path, "instr": instr,
            "first_div": (first_div["tick"], first_div["agent2_out"], first_div["oracle"]) if first_div else None,
            "final_bank": len(ticks[-1]["bank"]) if ticks else 0, "src": os.path.basename(video)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="0p8b_on20")
    ap.add_argument("--trace", default="wam_live_bank_0p8b_on19_lane{L}.jsonl")
    ap.add_argument("--lanes", default="W,X,Y,Z")
    ap.add_argument("--dir-pattern", default="wam_{arm}_lane{L}", help="result dir name pattern")
    ap.add_argument("--append", action="store_true", help="append to an existing index.md")
    ap.add_argument("--tasks", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    only = set(args.tasks.split(",")) if args.tasks else None
    jobs = []
    for L in args.lanes.split(","):
        root = f"{E}/{args.dir_pattern.format(arm=args.arm, L=L)}/ckpt79999/seed7/oracle"
        prog = json.load(open(f"{root}/progress.json"))
        occ = load_occurrences(f"{TR}/{args.trace.format(L=L)}")
        for task, eps in prog.items():
            if only and task not in only:
                continue
            for ep, v in eps.items():
                if v is not False:
                    continue
                ep = int(ep)
                vids = [f for f in glob.glob(f"{root}/videos/{task}_ep{ep}_*.mp4") if "ongoing" not in f]
                occs = occ.get((task, ep))
                if not vids or not occs:
                    print(f"skip {task} ep{ep}: vids={len(vids)} traces={len(occs) if occs else 0}")
                    continue
                video = max(vids, key=os.path.getmtime)
                n = int(cv2.VideoCapture(video).get(cv2.CAP_PROP_FRAME_COUNT))
                o = pick_occurrence(occs, n)
                d = f"{args.out}/{task}"
                os.makedirs(d, exist_ok=True)
                jobs.append((video, f"{d}/{task}_ep{ep}_annotated.mp4", o, task, ep))
    print(f"{len(jobs)} failed episodes to annotate")
    with Pool(int(os.environ.get("WORKERS", "16"))) as p:
        res = p.map(annotate, jobs)
    by = {}
    for r in res:
        by.setdefault(r["task"], []).append(r)
    with open(f"{args.out}/index.md", "a" if args.append else "w") as f:
        f.write(f"# Failed episodes, arm {args.arm} lanes {args.lanes} (seed 7), annotated videos\n\n")
        f.write("Panel: OURS green = subgoal text matches the oracle at that tick, red = differs. "
                "Circles: cyan = coordinate we cited, yellow = oracle's. Red dots = SAM-3 detections "
                "of the last window frame. Bank = last 10 admitted lines.\n\n")
        for task in sorted(by):
            rows = sorted(by[task], key=lambda r: r["ep"])
            f.write(f"## {task} ({len(rows)} failed)\n\n| ep | ticks | first divergence (tick: ours / oracle) | bank lines | video |\n|---|---|---|---|---|\n")
            for r in rows:
                fd = r["first_div"]
                fds = f"t{fd[0]}: `{fd[1]}` / `{fd[2]}`" if fd else "never (content matched throughout)"
                f.write(f"| {r['ep']} | {r['ticks']} | {fds} | {r['final_bank']} | `{r['video']}` |\n")
            f.write("\n")
    print("index:", f"{args.out}/index.md")


if __name__ == "__main__":
    main()

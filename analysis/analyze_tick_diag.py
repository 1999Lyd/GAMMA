#!/usr/bin/env python
"""Tick-diagnosis statistics + figure from WAM_DIAG_TRACE jsonls
(per tick: {count, pred, oracle}).

Metrics per task, aggregated to family:
  content    : % ticks where pred == oracle after coordinate stripping
               (both trained/prompted on the same oracle string recipe,
               so strict match is fair)
  gnd med/px : median |pred coord - oracle coord| over content-matched
               ticks where both cite a coordinate
  gnd >14px  : share of those beyond the corpus hit radius (wrong-instance
               proxy; SAM-3 prompt tuning scored hit@14px)
  adopted    : % oracle transitions whose new text the prediction ever
               adopts, and median adoption lag in ticks
  premature  : % predicted transitions to a text the oracle has NOT yet
               issued and does not hold now (schedule-driven switches)

Env: DIAG_DIR (dir of diag_*.jsonl), OUT_PREFIX (stats md/json), FIG (pdf path,
optional), TIMELINE_EP (task substring for panel b, default VideoUnmaskSwap).
"""
import collections, glob, json, math, os, re

C = re.compile(r"<\s*(-?\d+)\s*,\s*(-?\d+)\s*>")
DIAG = os.environ["DIAG_DIR"]
OUT = os.environ.get("OUT_PREFIX", os.path.join(DIAG, "tick_diag"))
FIG = os.environ.get("FIG", "")
THR = 14.0

FAMILY = {"ButtonUnmask": "press/count", "ButtonUnmaskSwap": "press/count",
          "StopCube": "press/count",
          "VideoUnmaskSwap": "binding", "VideoRepick": "binding",
          "VideoPlaceOrder": "binding", "VideoUnmask": "binding",
          "VideoPlaceButton": "binding", "PickHighlight": "binding",
          "PatternLock": "direction", "RouteStick": "direction"}


def strip(s):
    return re.sub(r"\s+", " ", C.sub("<>", (s or "").lower())).strip().rstrip(".")


def episode(rows):
    out = dict(ticks=0, content=0, gerr=[], adopted=[], premature=0, ptrans=0)
    for r in rows:
        p, o = r["pred"], r["oracle"]
        out["ticks"] += 1
        if strip(p) == strip(o):
            out["content"] += 1
            pc, oc = C.findall(p or ""), C.findall(o or "")
            if pc and oc:
                out["gerr"].append(math.dist(tuple(map(int, pc[0])),
                                             tuple(map(int, oc[0]))))
    osts = [strip(r["oracle"]) for r in rows]
    psts = [strip(r["pred"]) for r in rows]
    # oracle transitions: adoption + lag
    for i in range(1, len(rows)):
        if osts[i] != osts[i - 1]:
            lag = next((j - i for j in range(i, len(rows))
                        if psts[j] == osts[i]), None)
            out["adopted"].append(lag)
    # predicted transitions: premature if oracle hasn't issued the text yet
    for i in range(1, len(rows)):
        if psts[i] != psts[i - 1]:
            out["ptrans"] += 1
            if psts[i] not in osts[: i + 1]:
                out["premature"] += 1
    return out


tasks = collections.defaultdict(list)
for f in sorted(glob.glob(os.path.join(DIAG, "diag_*.jsonl"))):
    task = os.path.basename(f).split("diag_")[1].rsplit("_ep", 1)[0]
    tasks[task].append([json.loads(l) for l in open(f)])


def agg(eps):
    T = sum(e["ticks"] for e in eps)
    c = sum(e["content"] for e in eps)
    g = sorted(x for e in eps for x in e["gerr"])
    ad = [l for e in eps for l in e["adopted"]]
    got = [l for l in ad if l is not None]
    pt = sum(e["ptrans"] for e in eps)
    pm = sum(e["premature"] for e in eps)
    return dict(ticks=T, content=100 * c / max(T, 1),
                gmed=g[len(g) // 2] if g else None,
                gbad=100 * sum(x > THR for x in g) / max(len(g), 1) if g else None,
                ntrans=len(ad), adopted=100 * len(got) / max(len(ad), 1),
                lagmed=sorted(got)[len(got) // 2] if got else None,
                ntranspred=pt, premature=100 * pm / max(pt, 1))


proc = {t: [episode(r) for r in eps] for t, eps in tasks.items()}
rows_t = {t: agg(eps) for t, eps in sorted(proc.items())}
fam_eps = collections.defaultdict(list)
for t, eps in proc.items():
    fam_eps[FAMILY.get(t, "other")] += eps
rows_f = {f: agg(eps) for f, eps in sorted(fam_eps.items())}

def fmt(v, d=1):
    return "--" if v is None else f"{v:.{d}f}"

with open(OUT + ".md", "w") as w:
    for name, rows in (("task", rows_t), ("family", rows_f)):
        w.write(f"| {name} | ticks | content% | gnd med px | gnd>14px% "
                "| trans | adopted% | lag med | pred-trans | premature% |\n")
        w.write("|" + "---|" * 10 + "\n")
        for k, s in rows.items():
            w.write(f"| {k} | {s['ticks']} | {s['content']:.1f} "
                    f"| {fmt(s['gmed'])} | {fmt(s['gbad'])} | {s['ntrans']} "
                    f"| {s['adopted']:.1f} | {fmt(s['lagmed'], 0)} "
                    f"| {s['ntranspred']} | {s['premature']:.1f} |\n")
        w.write("\n")
json.dump({"task": rows_t, "family": rows_f}, open(OUT + ".json", "w"), indent=1)
print(open(OUT + ".md").read())

if FIG:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 2.9))
    # (a) grounding error per task (content-matched ticks)
    ax = axes[0]
    names = [t for t in rows_t if proc[t] and any(e["gerr"] for e in proc[t])]
    data = [[x for e in proc[t] for x in e["gerr"]] for t in names]
    ax.boxplot(data, tick_labels=[n.replace("Video", "V") for n in names],
               showfliers=False)
    for i, d in enumerate(data):
        ax.plot([i + 1 + 0.08] * len(d), d, ".", ms=2, alpha=0.25, color="tab:blue")
    ax.axhline(THR, color="tab:red", lw=0.8, ls="--")
    ax.text(0.98, THR + 1, "wrong-instance radius", ha="right", fontsize=7,
            color="tab:red", transform=ax.get_yaxis_transform())
    ax.set_ylabel("grounding error (px)")
    ax.set_title("(a) right words, wrong place", fontsize=9)
    ax.tick_params(axis="x", labelsize=7, rotation=30)
    # (b) one episode timeline
    ax = axes[1]
    tl_task = os.environ.get("TIMELINE_EP", "VideoUnmaskSwap")
    eps = tasks.get(tl_task, [[]])
    ep = max(eps, key=len)
    osts = [strip(r["oracle"]) for r in ep]
    psts = [strip(r["pred"]) for r in ep]
    vocab = []
    for s in osts + psts:
        if s not in vocab:
            vocab.append(s)
    x = [r["count"] for r in ep]
    ax.step(x, [vocab.index(s) for s in osts], where="post", lw=1.6,
            label="oracle", color="tab:gray")
    ax.step(x, [vocab.index(s) for s in psts], where="post", lw=1.0,
            label="single VLM", color="tab:red")
    for r in ep:
        pc, oc = C.findall(r["pred"] or ""), C.findall(r["oracle"] or "")
        if pc and oc and strip(r["pred"]) == strip(r["oracle"]):
            d = math.dist(tuple(map(int, pc[0])), tuple(map(int, oc[0])))
            if d > THR:
                ax.plot(r["count"], vocab.index(strip(r["pred"])), "x",
                        color="tab:red", ms=4)
    ax.set_xlabel("control step")
    ax.set_ylabel("subgoal index")
    ax.set_title(f"(b) {tl_task} episode ('x' = off-instance)", fontsize=9)
    ax.legend(fontsize=7, loc="lower right")
    # (c) family bars
    ax = axes[2]
    fams = list(rows_f)
    metrics = [("content", "content match %"),
               ("adopted", "transitions adopted %"),
               ("premature", "premature switches %")]
    wd = 0.25
    for m, (key, lab) in enumerate(metrics):
        vals = [rows_f[f][key] for f in fams]
        ax.bar([i + (m - 1) * wd for i in range(len(fams))], vals, wd, label=lab)
    ax.set_xticks(range(len(fams)))
    ax.set_xticklabels(fams, fontsize=8)
    ax.set_ylim(0, 105)
    ax.set_title("(c) failure mode by family", fontsize=9)
    ax.legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(FIG, bbox_inches="tight")
    print("figure ->", FIG)

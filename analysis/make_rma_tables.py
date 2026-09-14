"""Regenerate rma_section_draft.tex (main-text §5.5 + tab:rma, Appendix G + tab:rma-full) from the
eval outputs. Re-run whenever a lane finishes; then apply_rma_section.py applies it to the paper."""
import csv, glob, os
ST = os.path.expandvars('${GAMMA_ROOT}/rma/eval_stack/outputs')
OUT = os.path.expandvars('${GAMMA_WORK}/rma_section_draft.tex')
FAM = {'Sequence': [1, 2, 3, 22], 'Counting': [6, 7, 8, 9, 10, 15, 16],
       'Occlusion': [4, 5, 11, 12, 13, 14, 17, 20, 21, 23, 24], 'Transferring': [18, 19, 25, 26]}
def load(pats):
    rows = {}
    for pat in pats:
        for f in sorted(glob.glob(os.path.join(ST, pat))):
            for r in csv.DictReader(open(f), delimiter='\t'):
                rows.setdefault(int(r['task_id']), (float(r['TSR']), float(r['CSR'])))
    return rows
def fam(rows):
    out = {}
    for k, ts in FAM.items():
        out[k] = (sum(rows[t][0] for t in ts) / len(ts), sum(rows[t][1] for t in ts) / len(ts))
    out['Average'] = (sum(v[0] for v in rows.values()) / 26, sum(v[1] for v in rows.values()) / 26)
    return out
def complete(rows): return all(t in rows for t in range(1, 27))
rows = {}
rows['ungated50'] = load(['oracle_full50_rma_pi05_sgprompt_v1_79999_L*/task_summary.tsv'])
rows['gate2'] = load(['oracle_gate2_10_rma_pi05_sgprompt_v1_79999_t*/task_summary.tsv'])
h50 = load(['gamma_final50h_*/task_summary.tsv']); h10 = load(['gamma_final10_rma_pi05_sgprompt_v1_79999_t10_L*/task_summary.tsv'])
rows['harness'] = h50 if complete(h50) else h10; harness_trials = 50 if complete(h50) else 10
w = load(['gamma_writer10b*/task_summary.tsv', 'gamma_writer10_rma_pi05_sgprompt_v1_79999_t10_L*/task_summary.tsv'])
rows['writer'] = w if complete(w) else None
t50 = load(['theirs50u_*/task_summary.tsv']); t10 = load(['oracle_probe10_predimem_vla_alltask_L*/task_summary.tsv'])
rows['theirs'] = t50 if complete(t50) else t10; theirs_trials = 50 if complete(t50) else 10
rows['theirs_h'] = load(['gamma_theirs10_predimem_vla_alltask_theirs_t10_L*/task_summary.tsv'])
for k, v in rows.items():
    if v is not None: assert complete(v), (k, sorted(set(range(1, 27)) - set(v)))
F = {k: fam(v) for k, v in rows.items() if v is not None}
def f1(x): return f'{x:.1f}'
def pair(k, famname): a, b = F[k][famname]; return f'{f1(a)} / {f1(b)}'
def famrow_old(label, k):
    return label + ' & ' + ' & '.join(pair(k, n) for n in ('Sequence', 'Counting', 'Occlusion', 'Transferring', 'Average')) + r' \\'
theirs_tab2 = {  # RoboMemArena Table 2 (their protocol), (task, subtask) per family
    r"authors' reported $\pi_{0.5}$": {'Sequence': (60.0, 71.6), 'Counting': (14.3, 50.9), 'Occlusion': (12.7, 17.2), 'Transferring': (20.0, 42.8), 'Average': (21.5, 38.7)},
    r"authors' reported PrediMem (full system)": {'Sequence': (72.5, 89.5), 'Counting': (45.7, 69.3), 'Occlusion': (27.3, 38.4), 'Transferring': (22.5, 45.2), 'Average': (38.5, 55.2)},
    r"authors' reported ground-truth subtask feed": {'Sequence': (85.0, 92.3), 'Counting': (51.4, 75.6), 'Occlusion': (33.6, 49.8), 'Transferring': (32.5, 54.8), 'Average': (46.1, 64.8)},
}
def reprow2(label, d):
    return label + ' & ' + ' & '.join(f'{f1(d[n][0])} / {f1(d[n][1])}' for n in ('Sequence', 'Counting', 'Occlusion', 'Transferring', 'Average')) + r' \\'

COLS = ('Transferring', 'Occlusion', 'Counting', 'Sequence', 'Average')
theirs_tab2.update({
    r"MemoryVLA~\citep{memoryvla}": {'Sequence': (37.5, 65.2), 'Counting': (14.3, 55.1), 'Occlusion': (7.3, 13.1), 'Transferring': (15.0, 37.2), 'Average': (15.0, 35.3)},
    r"MemER~\citep{memer}": {'Sequence': (65.0, 79.1), 'Counting': (27.1, 65.1), 'Occlusion': (16.4, 33.2), 'Transferring': (20.0, 36.1), 'Average': (27.3, 49.1)},
})
def cell(a, b, bold_a=False, bold_b=False):
    fa, fb = f1(a), f1(b)
    return (f'\\textbf{{{fa}}}' if bold_a else fa) + ' / ' + (f'\\textbf{{{fb}}}' if bold_b else fb)
def table2_rows(spec):
    """spec: list of (label, dict family->(task,subtask), privileged:bool, rowcolor:str|None). Bold best non-privileged per column and metric."""
    best = {c: (max(d[c][0] for _, d, priv, _ in spec if not priv), max(d[c][1] for _, d, priv, _ in spec if not priv)) for c in COLS}
    out = []
    for label, d, priv, color in spec:
        cells = [cell(d[c][0], d[c][1], (not priv) and d[c][0] == best[c][0], (not priv) and d[c][1] == best[c][1]) for c in COLS]
        out.append((f'\\rowcolor{{{color}}} ' if color else '') + label + ' & ' + ' & '.join(cells) + r' \\')
    return out
def famrow2(label, k, color=None):
    d = F[k]; return (f'\\rowcolor{{{color}}} ' if color else '') + label + ' & ' + ' & '.join(f'{f1(d[c][0])} / {f1(d[c][1])}' for c in COLS) + r' \\'
def reprow2(label, d):
    return label + ' & ' + ' & '.join(f'{f1(d[c][0])} / {f1(d[c][1])}' for c in COLS) + r' \\'

U, H, T, G, TH = F['ungated50']['Average'], F['harness']['Average'], F['theirs']['Average'], F['gate2']['Average'], F['theirs_h']['Average']
mark = lambda n: '' if n == 50 else f' ({n} ep.)'
writer_main = writer_full = ''
if 'writer' in F:
    W = F['writer']['Average']
    writer_main = f"\\rowcolor{{oursrowlite}} \\sys{{}} with the writer (10 ep.) & {f1(W[0])} & {f1(W[1])} \\\\\n"
    writer_full = "\\rowcolor{oursrowlite} " + famrow2(r"\sys{} with the writer (10 ep.)", 'writer') + "\n"
gt_row = theirs_tab2[r"authors' reported ground-truth subtask feed"]
spec = [(r"$\pi_{0.5}$ (no subgoals)", theirs_tab2[r"authors' reported $\pi_{0.5}$"], False, None),
        (r"MemoryVLA~\citep{memoryvla}", theirs_tab2[r"MemoryVLA~\citep{memoryvla}"], False, None),
        (r"MemER~\citep{memer}", theirs_tab2[r"MemER~\citep{memer}"], False, None),
        (r"PrediMem~\citep{rma}", theirs_tab2[r"authors' reported PrediMem (full system)"], False, None),
        (r"\sys{} (ours)", F['harness'], False, 'oursrow')]
if 'writer' in F: spec.append((r"\sys{} with the writer (10 ep.)", F['writer'], True, 'oursrowlite'))  # not in the bold race
spec.append((r"ground-truth subtask feed (privileged)", gt_row, True, 'oraclerow'))
rows_tex = table2_rows(spec)
ours_idx = 4
main_rows = "\n".join(rows_tex[:4]) + "\n\\midrule\n" + "\n".join(rows_tex[4:-1]) + "\n\\midrule\n" + rows_tex[-1]
main = f"""%% ===== DRAFT: main-text §5.5 (replaces the current subsection + tab:rma) =====
\\subsection{{A second benchmark: RoboMemArena}}
\\label{{sec:rma}}

\\sys{{}} transfers to RoboMemArena~\\citep{{rma}}, 26 long-horizon kitchen
tasks in four families, with its contracts unchanged
(Appendix~\\ref{{app:rma}}). Behind the benchmark's own $\\pi_{{0.5}}$ recipe
it reaches {f1(H[0])} task and {f1(H[1])} subtask success on held-out
layouts, above the reported memory-VLA and keyframe-memory methods and
level with the benchmark's own memory system (Table~\\ref{{tab:rma}}).

\\begin{{table}}[h]
\\vspace{{-2pt}}
\\centering
\\scriptsize
\\setlength{{\\tabcolsep}}{{3pt}}
\\renewcommand{{\\arraystretch}}{{0.85}}
\\begin{{tabular}}{{l|cccc|c}}
\\toprule
RoboMemArena, task / subtask success (\\%) & Transferring & Occlusion & Counting & Sequence & Average \\\\
\\midrule
{main_rows}
\\bottomrule
\\end{{tabular}}
\\caption{{RoboMemArena by task family, in the layout of the benchmark's
Table~2: baseline and privileged rows as reported by the benchmark;
\\sys{{}} on held-out layouts, {harness_trials} episodes per task. Bold:
best non-privileged result per column and metric.}}
\\vspace{{-8pt}}
\\label{{tab:rma}}
\\end{{table}}

"""
app = f"""%% ===== DRAFT: Appendix G (replaces the current \\section{{Towards a second benchmark...}}) =====
\\section{{RoboMemArena details}}
\\label{{app:rma}}
\\textbf{{Transfer.}} Nothing in \\sys{{}} binds to RoboMME: the contracts
assume a detector, a tick grid, recorded episodes and an evidence stream
to verify claims against. The port changed two geometry constants (tick
$=10$ steps, $3$ frames per window), the detector prompts, and the
harness's evidence predicates, which here read the executor's
proprioceptive state: pick (fingers stalled inside the object's grasp
band, then the hand rises 3\\,cm), place (carry, then release), pour
(wrist rotated $\\geq 25^\\circ$ from the carry orientation and back), open
and close (handle grasp and stroke). The agent code, the corpus
generator and the harness logic are those of Section~\\ref{{sec:method}}.
In the reported configuration the writer is not run: with a fixed plan
and proprioceptive evidence, progress claims are raised by the harness's
own predicates and admitted once a predicate has held for two ticks. On the recorded episodes the predicates, with the
grasp-width priors used at deployment, fire inside their own segment in
879 of 888 cases. Because every RoboMemArena task
has a fixed subtask sequence, the reasoner is pinned to the recorded
plan and the memory stage reduces to verified progress: the harness
decides \\emph{{when}} the executor is handed the next subtask.
\\textbf{{Executor.}} The authors' bundled recipe: $\\pi_{{0.5}}$ initialised
from the base checkpoint, action horizon 10, batch 128, 40k updates,
warm-up 10k then constant $5\\times10^{{-5}}$, EMA 0.999, trained on the
2{{,}}600 released demonstrations (100 per task, segmented into subtasks)
with each segment's subtask text as the prompt. State normalisation
statistics match the released checkpoint's to three decimals and the
action quantiles differ by at most 0.04.
\\textbf{{Protocol.}} Task success requires every scoring stage of an
episode; subtask success is the fraction of stages reached. Episodes
run 2{{,}}500 steps with 10 actions per policy call. The authors'
evaluation scripts default to seed 100 with one episode per task; the
released demonstrations are seeds 100 and above, so that default replays
training layouts. We evaluate on seeds 50--99 (50 episodes per task, 10
for the marked rows). On tasks 1--4 the authors' released checkpoint
scores the same on training and held-out seeds (70/100/90/0 vs.\\
70/90/100/0 task success), so the choice of seeds is not what separates
the rows of Table~\\ref{{tab:rma-full}}.
\\textbf{{Hand-off timing.}} The released segments end at completion:
place segments end with the gripper fully open and the arm retracted
($100\\%$ of segments), pour segments after the wrist has returned
($|\\Delta\\text{{rot}}|=0.01$\\,rad), pick segments with the fingers closed
on the object. The benchmark's scoring predicates fire earlier (object
entering the receptacle while still grasped; tilt peak), so a plan
advanced on them hands the executor a subtask mid-motion; advancing on
the segment-completion conditions instead (object released, wrist
returned) removes part of the gap ({f1(G[0])} task success), and the
harness's verified transitions, which also wait two ticks after the
predicate holds, remove most of it ({f1(H[0])}).
\\textbf{{Two executors, complementary failures.}} The authors' released
$\\pi_{{0.5}}$ under our protocol reaches {f1(T[0])} / {f1(T[1])} with the
benchmark-predicate feed and {f1(TH[0])} / {f1(TH[1])} with the harness's
timing: it is stronger on the microwave and cabinet-transfer tasks and
weaker on pouring. Both executors, on held-out and on training layouts,
fail the drawer tasks (4, 5, 11, 14); on the three-drawer tasks 4 and 5
ours opens and closes the first drawer in every episode and then fails
the second, while theirs rarely opens the first.

\\begin{{table}}[tb]
\\centering
\\scriptsize
\\setlength{{\\tabcolsep}}{{3pt}}
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{l|cccc|c}}
\\toprule
Row (task / subtask success, \\%) & Transferring & Occlusion & Counting & Sequence & Average \\\\
\\midrule
\\multicolumn{{6}}{{l}}{{\\emph{{Our protocol: held-out layouts (seeds 50--99), 50 episodes per task unless marked}}}} \\\\
{famrow2(r"$\pi_{0.5}$ executor, plan fed on benchmark stage predicates (privileged)", 'ungated50')}
{famrow2(r"$\pi_{0.5}$ executor, plan fed on segment-completion conditions (privileged, 10 ep.)", 'gate2')}
\\rowcolor{{oursrow}} {famrow2(r"\sys{} (harness-verified transitions)" + mark(harness_trials), 'harness')}
{writer_full}{famrow2(r"authors' released $\pi_{0.5}$, plan fed on benchmark stage predicates" + mark(theirs_trials), 'theirs')}
{famrow2(r"authors' released $\pi_{0.5}$, \sys{} harness timing (10 ep.)", 'theirs_h')}
\\midrule
\\multicolumn{{6}}{{l}}{{\\emph{{Authors' protocol, as reported}}}} \\\\
{reprow2(r"authors' reported $\pi_{0.5}$", theirs_tab2[r"authors' reported $\pi_{0.5}$"])}
{reprow2(r"authors' reported PrediMem (full system)", theirs_tab2[r"authors' reported PrediMem (full system)"])}
{reprow2(r"authors' reported ground-truth subtask feed", theirs_tab2[r"authors' reported ground-truth subtask feed"])}
\\bottomrule
\\end{{tabular}}}}
\\caption{{RoboMemArena by task family (task / subtask success, \\%). Family
sizes: sequence 4 tasks, counting 7, occlusion 11, transferring 4. The
upper block is our protocol; the lower block reproduces the authors'
Table~2 under their own protocol for reference.}}
\\label{{tab:rma-full}}
\\end{{table}}

"""
open(OUT, 'w').write(main + app)
print('draft written:', OUT)
print(f"ungated50 {U}  harness({harness_trials}) {H}  gate2 {G}  theirs({theirs_trials}) {T}  theirs_h {TH}  writer {'complete' if 'writer' in F else 'incomplete -> row omitted'}")

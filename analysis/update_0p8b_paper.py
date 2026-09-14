#!/usr/bin/env python3
"""Regenerate every 0.8B-dependent number in the paper from the result files.

Idempotent: reads the three-seed 0.8B sweep (lanes W-Z) with the post-fix
PickXtimes reruns (laneP) overriding lane W's PickXtimes, computes the
three-seed per-task means, and rewrites
  * Table 2  (tab:main)      the '0.8B agents' row
  * Table 3  (tab:ablation)  the 0.8B 'full harness' row and the grey shading
                             of the 0.8B 'no harness' row (>= 4-point loss)
  * Sec. 5.3 'Across scale'  harness delta at 0.8B, 13-task delta, 'vs. 44.5'
  * Table 2 caption          both GAMMA rows are three-seed means
  * Fig. 1 bar               patch_compare.py value (+ regenerate teaser.png)
Usage: update_0p8b_paper.py --tex <file> [--allow-partial] [--teaser] [--dry-run]
"""
import argparse, json, os, re, sys, glob, subprocess
from decimal import Decimal, ROUND_HALF_UP

E = os.path.expandvars('${ROBOMME_ROOT}/examples/robomme/runs/evaluation')
R = os.path.expandvars('${GAMMA_DATA}/traces')
PATCH_COMPARE = os.path.expandvars('${GAMMA_WORK}/teaser/patch_compare.py')
MSPY = os.path.expandvars('${MSSWIFT_PY}')
SEEDS = (7, 8, 9)
LANES = {'laneW': ['BinFill', 'PickXtimes', 'ButtonUnmask', 'ButtonUnmaskSwap'],
         'laneX': ['VideoUnmask', 'VideoRepick', 'MoveCube', 'PatternLock'],
         'laneY': ['StopCube', 'SwingXtimes', 'PickHighlight', 'VideoUnmaskSwap'],
         'laneZ': ['VideoPlaceButton', 'VideoPlaceOrder', 'InsertPeg', 'RouteStick'],
         'laneP': ['PickXtimes']}          # post-fix rerun; overrides laneW's PickXtimes
COLS = ['PickXtimes', 'BinFill', 'SwingXtimes', 'StopCube', 'VideoUnmask', 'ButtonUnmask',
        'VideoUnmaskSwap', 'ButtonUnmaskSwap', 'PickHighlight', 'VideoRepick', 'VideoPlaceButton',
        'VideoPlaceOrder', 'MoveCube', 'InsertPeg', 'PatternLock', 'RouteStick']
READER_TASKS = {'VideoRepick', 'VideoPlaceButton', 'VideoPlaceOrder'}   # 0.8B-specific extractors -> excluded from the 13-task figure
NOHARNESS_0P8B = [70.0, 46.7, 56.7, 10.0, 73.3, 53.3, 50.0, 6.7, 50.0, 26.7, 13.3, 23.3, 30.0, 0.0, 3.3, 3.3]
NINEB_FULL, NINEB_OFF, BEST_PRIOR = 66.0, 51.2, 44.5

def r1(x):
    return float(Decimal(str(x)).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP))

def load_rates(allow_partial):
    """rates[task][seed] = success fraction over 30 episodes; source notes."""
    rates, notes = {}, []
    for lane in ('laneW', 'laneX', 'laneY', 'laneZ', 'laneP'):
        for s in SEEDS:
            p = f'{E}/wam_0p8b_on20_{lane}/ckpt79999/seed{s}/oracle/log.json'
            got = {}
            if os.path.exists(p):
                d = json.load(open(p))
                got = {t: v for t, v in d.get('success_rate', {}).items() if t in LANES[lane]}
            missing = [t for t in LANES[lane] if t not in got]
            if missing and allow_partial:
                # fall back to the live trace (partial lane): end records
                tr = f'{R}/wam_live_bank_0p8b_on19_{lane}_s{s}.jsonl'
                cnt = {}
                if os.path.exists(tr):
                    for l in open(tr):
                        try: rec = json.loads(l)
                        except Exception: continue
                        if rec.get('kind') == 'end' and rec.get('task') in missing:
                            c = cnt.setdefault(rec['task'], [0, 0]); c[1] += 1; c[0] += rec.get('success') == 'success'
                for t, (a, n) in cnt.items():
                    got[t] = a / n; notes.append(f'PARTIAL {lane} seed{s} {t}: {a}/{n} episodes (trace)')
            for t, v in got.items():
                if lane == 'laneP' or t != 'PickXtimes':
                    rates.setdefault(t, {})[s] = v
                if lane == 'laneW' and t == 'PickXtimes':
                    rates.setdefault('PickXtimes_prefix', {})[s] = v
    return rates, notes

def fmt(x): return f'{r1(x):.1f}'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tex', required=True); ap.add_argument('--allow-partial', action='store_true')
    ap.add_argument('--teaser', action='store_true'); ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    rates, notes = load_rates(a.allow_partial)
    missing = [(t, s) for t in COLS for s in SEEDS if s not in rates.get(t, {})]
    if missing and not a.allow_partial:
        print('INCOMPLETE:', missing); sys.exit(2)
    per_seed = {t: {s: rates[t][s] * 100 for s in SEEDS if s in rates.get(t, {})} for t in COLS}
    means = {t: r1(sum(v.values()) / len(v)) if v else float('nan') for t, v in per_seed.items()}
    row = [means[t] for t in COLS]
    avg = r1(sum(row) / len(row))
    off = dict(zip(COLS, NOHARNESS_0P8B)); off_avg = r1(sum(NOHARNESS_0P8B) / 16)
    thirteen = [t for t in COLS if t not in READER_TASKS]
    full13 = r1(sum(means[t] for t in thirteen) / 13); off13 = r1(sum(off[t] for t in thirteen) / 13)
    delta = r1(avg - off_avg); delta13 = r1(full13 - off13); delta9 = r1(NINEB_FULL - NINEB_OFF)
    shade = [t for t in COLS if means[t] - off[t] >= 4.0 - 1e-9]
    # ---------------- report ----------------
    print('per-seed (%):'); [print(f'  {t:17s}', {s: fmt(v) for s, v in per_seed[t].items()}, '-> mean', fmt(means[t])) for t in COLS]
    if 'PickXtimes_prefix' in rates: print('  (lane W pre-fix PickXtimes, replaced):', {s: fmt(v*100) for s, v in rates['PickXtimes_prefix'].items()})
    print(f'Avg {avg}  | no-harness Avg {off_avg} -> harness delta +{delta} (9B +{delta9}) | 13-task {off13} -> {full13} (+{delta13})')
    print('no-harness cells shaded (>=4 below full):', shade)
    for n in notes: print(' ', n)
    if a.dry_run: print('dry run: tex untouched'); return
    tex = open(a.tex).read(); orig = tex
    cells = ' & '.join(fmt(x) for x in row[:8]) + '\n& ' + ' & '.join(fmt(x) for x in row[8:]) + f' & {avg:.1f} \\\\'
    # Table 2 row
    pat = re.compile(r'(& \\sys\{\} \\emph\{0\.8B agents\}\n)& [^\n]*\n& [^\n]*\\\\')
    assert len(pat.findall(tex)) == 1, 'tab:main 0.8B row not unique'
    tex = pat.sub(lambda m: m.group(1) + '& ' + cells, tex)
    # Table 3 full-harness row (the one after the 0.8B multicolumn header)
    pat = re.compile(r'(Qwen3\.5-0\.8B\}\} \\\\\n\\sys\{\} \(full harness\)\n)& [^\n]*\n& [^\n]*\\\\')
    assert len(pat.findall(tex)) == 1, 'tab:ablation 0.8B full row not unique'
    tex = pat.sub(lambda m: m.group(1) + '& ' + cells, tex)
    # Table 3 no-harness row: values fixed, shading recomputed
    pat = re.compile(r'(Qwen3\.5-0\.8B\}\} \\\\\n\\sys\{\} \(full harness\)\n& [^\n]*\n& [^\n]*\\\\\n\\quad no harness\n)((?:& [^\n]*\n)*?& [^\n]*\\\\)')
    m = pat.search(tex); assert m, 'tab:ablation 0.8B no-harness row not found'
    def cell(t, v): return ('\\cellcolor{offcell}' if t in shade else '') + fmt(v)
    offcells = [cell(t, off[t]) for t in COLS]
    offrow = ('& ' + ' & '.join(offcells[0:5]) + '\n& ' + ' & '.join(offcells[5:8]) + '\n& ' + ' & '.join(offcells[8:12])
              + '\n& ' + ' & '.join(offcells[12:14]) + '\n& ' + ' & '.join(offcells[14:16]) + f' & {off_avg:.1f} \\\\')
    tex = tex[:m.start(2)] + offrow + tex[m.end(2):]
    # Sec 5.3 numbers
    subs = [
        (r'the harness is worth \$\+[0-9.]+\$ points at\n0\.8B \(\$[0-9.]+ \\to [0-9.]+\$\)',
         f'the harness is worth $+{delta}$ points at\n0.8B (${off_avg} \\to {avg}$)'),
        (r'the small-scale figure is \$\+[0-9.]+\$ \(\$[0-9.]+ \\to [0-9.]+\$\)',
         f'the small-scale figure is $+{delta13}$ (${off13} \\to {full13}$)'),
        (r'\(\$[0-9.]+\$ vs\.\\ \$44\.5\$\)', f'(${avg}$ vs.\\ $44.5$)'),
        (r'(?:\\sys\{\} \(ours\) is the mean|both \\sys\{\} rows are means) over three\nserving seeds', 'both \\sys{} rows are means over three\nserving seeds'),
    ]
    for p, rep in subs:
        n = len(re.findall(p, tex)); assert n == 1, f'pattern not unique ({n}): {p[:60]}'
        tex = re.sub(p, lambda m: rep, tex)
    # Appendix D: 0.8B per-seed table (marker-delimited, regenerated each run)
    seed_avg = {s: r1(sum(r1(per_seed[t][s]) for t in COLS if s in per_seed[t]) / max(1, sum(1 for t in COLS if s in per_seed[t]))) for s in SEEDS}
    def srow(s): return ' & '.join(fmt(per_seed[t][s]) if s in per_seed[t] else '--' for t in COLS[:8]) + '\n& ' + ' & '.join(fmt(per_seed[t][s]) if s in per_seed[t] else '--' for t in COLS[8:]) + f' & {seed_avg[s]:.1f} \\\\'
    spread = r1(max(seed_avg.values()) - min(seed_avg.values()))
    block = ('% BEGIN 0P8B-SEEDS (generated by update_0p8b_paper.py; do not edit by hand)\n'
             'Table~\\ref{tab:seeds-0p8b} gives the same breakdown for the 0.8B agents\n'
             '(Section~\\ref{sec:ablation}): suite averages are\n'
             f'${seed_avg[7]}/{seed_avg[8]}/{seed_avg[9]}$---a ${spread}$-point spread---so the\n'
             'scale comparison of Table~\\ref{tab:ablation} rests on three seeds at both sizes.\n\n'
             '\\begin{table}[tb]\n\\centering\n\\scriptsize\n\\setlength{\\tabcolsep}{3pt}\n\\resizebox{\\textwidth}{!}{%\n'
             '\\begin{tabular}{l|cccc|cccc|cccc|cccc|c}\n\\toprule\n'
             'Seed & PickX & BinF & SwingX & StopC & VU & BU & VUS & BUS\n& PH & VRP & VPB & VPO & MC & IP & PL & RS & Avg \\\\\n\\midrule\n'
             + '\n'.join(f'{s} & ' + srow(s) for s in SEEDS) + '\n\\midrule\nmean & ' + cells + '\n'
             '\\bottomrule\n\\end{tabular}}\n'
             '\\caption{\\sys{} with the 0.8B agents: per-task success (\\%) by serving seed,\n30 episodes/task each.}\n'
             '\\label{tab:seeds-0p8b}\n\\end{table}\n'
             '% END 0P8B-SEEDS')
    if '% BEGIN 0P8B-SEEDS' in tex:
        tex = re.sub(r'% BEGIN 0P8B-SEEDS.*?% END 0P8B-SEEDS', lambda m: block, tex, flags=re.S)
    else:
        anchor = '\\label{tab:seeds}\n\\end{table}\n'
        assert tex.count(anchor) == 1, 'tab:seeds anchor not unique'
        tex = tex.replace(anchor, anchor + '\n' + block + '\n', 1)
    print('0.8B per-seed suite averages:', seed_avg, 'spread', spread)
    if tex != orig:
        open(a.tex, 'w').write(tex); print('tex updated:', a.tex)
    else: print('tex unchanged')
    if a.teaser:
        s = open(PATCH_COMPARE).read()
        s2, n = re.subn(r"\('GAMMA 0\.8B agents',[0-9.]+,", f"('GAMMA 0.8B agents',{avg},", s); assert n == 1
        open(PATCH_COMPARE, 'w').write(s2)
        subprocess.run([MSPY, PATCH_COMPARE], check=True); print('teaser regenerated with bar', avg)

if __name__ == '__main__':
    main()

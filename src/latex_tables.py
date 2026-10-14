"""latex_tables.py -- writes the LaTeX tables of the revised manuscript from
results/benchmark/{tables.json, loadcases.json, cs6_mlc.json} into <out>/tables/.
Also prints the numbers quoted in the prose (placeholders)."""
import os, sys, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R2 = os.path.join(ROOT, 'results', 'benchmark')
OUTDIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'tables')
os.makedirs(OUTDIR, exist_ok=True)
import numpy as np

T = json.load(open(os.path.join(R2, 'tables.json')))
CASES = [('CS2', 'CS2 nexorade'), ('CS3', 'CS3 hex.\\ nexorade'), ('CS6', 'CS4 bridge'),
         ('Dome', 'CS5 dome'), ('Vault', 'CS6 vault')]
ARMS = [('ga', 'GA, pop.\\ 20', 'ga'), ('ga8_w5', 'GA, pop.\\ 8', 'ga8_w5'), ('random', 'Random', 'random'),
        ('ls_w5', 'Local search', 'ls_w5'), ('sa_qubo', 'Single-flip', 'sa_single_w1000f'),
        ('greedy_w5', 'Decode', 'greedy_w1000f'), ('sa_block_w5', 'Block', 'sa_block_w1000f')]


def fmt(a, key='best', mass=False):
    if a is None or np.isnan(a.get(f'{key}_mean', np.nan)):
        return '--'
    base = f"{a[f'{key}_mean']:.0f} $\\pm$ {a[f'{key}_std']:.0f}"
    if mass and 'best_mass_err_pct' in a:
        base += f" ({a['best_mass_err_pct']:+.0f})"
    return base


def pstr(p):
    if p is None or np.isnan(p):
        return '--'
    return f"{p:.3f}" if p >= 0.001 else f"{p:.1e}".replace('e-0', 'e-')


def best_mass_err(arm_runs):
    return None


lines = [r'\begin{table}[!htbp]', r'\centering', r'\footnotesize',
         r'\caption{Cross-section sizing on five geometries at 50 FE evaluations. Stiffness-led rule: lowest compliance visited and the mass error of that design (linearised arms at $w_{\text{m}}=5$). Matched-mass rule: lowest compliance among designs within 5 percent of the mass target (linearised arms at $w_{\text{m}}=1000$; the number of seeds that reached the band is given). $p$ is the exact Mann--Whitney test against the block annealer under the matched-mass rule, Holm-adjusted within the table; $\delta$ is Cliff\textquotesingle s delta, negative when the block annealer has lower compliance. Seed counts are those of Table~\ref{tab:protocol}; the in-band column gives the seeds of the matched-mass arm.}',
         r'\label{tab:sizing}', r'\scriptsize', r'\setlength{\tabcolsep}{3pt}', r'\begin{tabular}{llccccc}', r'\toprule',
         r'Case & Method & Lowest $C$ [J] & Mass err.\ [\%] & Matched $C$ [J] & In band & $p$ (Holm), $\delta$ \\', r'\midrule']
mm_adv = {}; sl_adv = {}; sl_mass = []; w5e = []; w20e = []; w100e = []; w100cost = []; w100real = []
for ck, clabel in CASES:
    t = T[ck]; arms = t['arms']
    tests = {f['arm']: f for f in t['tests'] if f['ref'] == 'sa_block_w1000f'}
    ref = arms.get('sa_block_w1000f')
    first = True
    for led_key, label, mm_key in ARMS:
        a = arms.get(led_key); b = arms.get(mm_key)
        n = a['n'] if a else (b['n'] if b else 0)
        led = fmt(a, 'best'); lederr = f"{a['best_mass_err_pct']:+.1f}" if a and 'best_mass_err_pct' in a else '--'
        mm = fmt(b, 'mm') if b else '--'; band = f"{b['n_feasible']}/{b['n']}" if b else '--'
        if mm_key == 'sa_block_w1000f' or b is None:
            pc = '--'
        else:
            f = tests.get(mm_key); pc = f"{pstr(f['p_mm_holm'])}, ${f['cliff_mm']:+.2f}$" if f else '--'
        lines.append(f"{clabel if first else ''} & {label} & {led} & {lederr} & {mm} & {band} & {pc} \\\\")
        first = False
    lines.append(r'\midrule')
    # prose numbers
    if ref and 'mm_mean' in ref:
        pop = [arms[k]['mm_mean'] for k in ('ga', 'ga8_w5', 'random', 'ls_w5') if k in arms and not np.isnan(arms[k].get('mm_mean', np.nan))]
        if pop:
            mm_adv[ck] = 100 * (1 - ref['mm_mean'] / min(pop))
    if 'sa_qubo' in arms and 'ga' in arms:
        sl_adv[ck] = 100 * (1 - arms['sa_qubo']['best_mean'] / arms['ga']['best_mean']); sl_mass.append(arms['sa_qubo'].get('best_mass_err_pct', np.nan))
    for k, lst in (('sa_block_w5', w5e), ('sa_block_w20', w20e), ('sa_block_w1000f', w100e)):
        if k in arms: lst.append(arms[k].get('best_mass_err_pct', np.nan))
    if 'sa_block_w100' in arms: w100real.append(arms['sa_block_w100'].get('final_mass_err_pct', np.nan))
    if 'sa_block_w5' in arms and 'sa_block_w1000f' in arms:
        w100cost.append(100 * (arms['sa_block_w1000f']['best_mean'] / arms['sa_block_w5']['best_mean'] - 1))
lines[-1] = r'\bottomrule'
lines += [r'\end{tabular}', r'\normalsize', r'\end{table}']
open(os.path.join(OUTDIR, 'tab_sizing.tex'), 'w').write('\n'.join(lines) + '\n')

# ---- annealer table ----
L = [r'\begin{table}[!htbp]', r'\centering', r'\footnotesize',
     r'\caption{What the annealer adds: lowest compliance [J] from the same random initial design per seed for the greedy coordinate-descent decode, the single-flip annealer (25 reads, 90 sweeps) and the block annealer, at a soft ($w_{\text{m}}=5$) and a tight ($w_{\text{m}}=1000$, matched mass) mass penalty; the $w_{\text{m}}=20$ arms are in the data package. The mass error [\%] of each reported design is given in parentheses. Differences are relative to the decode (negative is better) with the exact paired Wilcoxon $p$ in parentheses; its floor is 0.002 at ten seeds, 0.0625 at five and 0.25 at three (the CS3 decode and single-flip arms at $w_{\\text{m}}=5$). On the bridge at $w_{\text{m}}=5$ the single-flip annealer at 100 reads and 400 sweeps attains STRONGCS6.}',
     r'\label{tab:annealer}', r'\scriptsize', r'\setlength{\tabcolsep}{4pt}', r'\begin{tabular}{llccc}', r'\toprule',
     r'Case & $w_{\text{m}}$ & Decode & Single-flip & Block \\', r'\midrule']
blk_vs_g = []; blk_vs_g1000 = []
for ck, clabel in CASES:
    t = T[ck]; arms = t['arms']; P = t['paired']
    for w in (5, 20, 1000):
        sfx = 'f' if w == 1000 else ''
        g = arms.get(f'greedy_w{w}{sfx}'); s = arms.get(f'sa_single_w1000f' if w == 1000 else f'sa_paired_w{w}'); b = arms.get(f'sa_block_w{w}{sfx}')
        if not (g and s and b): continue
        skey = 'sa_single_w1000f' if w == 1000 else f'sa_paired_w{w}'
        ps = P.get(f'greedy_w{w}{sfx} vs {skey}', {}); pb = P.get(f'greedy_w{w}{sfx} vs sa_block_w{w}{sfx}', {})
        ds = 100 * (s['best_mean'] / g['best_mean'] - 1); db = 100 * (b['best_mean'] / g['best_mean'] - 1)
        if w == 20: continue
        L.append(f"{clabel} & {w} & {fmt(g, mass=True)} & {fmt(s, mass=True)} & {fmt(b, mass=True)} \\\\")
        L.append(f" & & vs decode: & ${ds:+.1f}$\\% ($p={pstr(ps.get('p_wilcoxon_best'))}$) & ${db:+.1f}$\\% ($p={pstr(pb.get('p_wilcoxon_best'))}$) \\\\[2pt]")
        if w == 1000: blk_vs_g1000.append(db)
        else: blk_vs_g.append(db)
L += [r'\bottomrule', r'\end{tabular}', r'\normalsize', r'\end{table}']
txt = '\n'.join(L) + '\n'
if 'CS6' in T and 'sa_strong_w5' in T['CS6']['arms']:
    st = T['CS6']['arms']['sa_strong_w5']; txt = txt.replace('STRONGCS6', f"{st['best_mean']:.0f} $\\pm$ {st['best_std']:.0f}\\,J")
open(os.path.join(OUTDIR, 'tab_annealer.tex'), 'w').write(txt)

# ---- load cases table ----
LC = json.load(open(os.path.join(R2, 'loadcases.json')))
L = [r'\begin{table}[!htbp]', r'\centering', r'\footnotesize',
     r'\caption{Re-analysis of every best design under self-weight (SW) and live load (LL): ratio of total self-weight to total live load, peak deflection relative to span/250 and elastic utilisation (S355) under the nominal live load, SW with full LL and SW with half-span LL. The last four rows are means over ten seeds of the matched-mass bridge designs sized with one and with two load cases.}',
     r'\label{tab:loadcases}', r'\scriptsize', r'\setlength{\tabcolsep}{2.5pt}', r'\begin{tabular}{p{2.9cm}cccccccc}', r'\toprule',
     r'Design & Mass [kg] & SW/LL & \multicolumn{3}{c}{$d_{\max}/(\text{span}/250)$} & \multicolumn{3}{c}{Utilisation} \\',
     r' & & & nom. & SW+LL & SW+$\tfrac12$LL & nom. & SW+LL & SW+$\tfrac12$LL \\', r'\midrule']
short = {'CS2 nexorade (SA-QUBO best, seed 6)': 'CS2 nexorade', 'CS3 hexagonal nexorade (SA-QUBO best, seed 0)': 'CS3 hex.\\ nexorade',
         'CS6 da Vinci bridge (SA-QUBO best, seed 8)': 'CS4 bridge, one case, $w_{\\text{m}}=5$', 'Dome geometry-best (FMQA, seed 1, uniform D=244.5)': 'CS5 dome, geometry',
         'Dome sizing-best (SA-QUBO, seed 2)': 'CS5 dome, sized', 'Vault geometry-best (FMQA, seed 2, uniform D=244.5)': 'CS6 vault, geometry',
         'Vault sizing-best (SA-QUBO, seed 3)': 'CS6 vault, sized'}
for name, r in LC.items():
    c = lambda k, f: r[k][f]
    L.append(f"{short.get(name, name)} & {r['mass_kg']:.0f} & {r['selfweight_over_live']:.2f} & {c('nominal','d_over_span250'):.2f} & {c('sw_plus_full_live','d_over_span250'):.2f} & {c('sw_plus_half_live','d_over_span250'):.2f} & {c('nominal','max_util'):.2f} & {c('sw_plus_full_live','max_util'):.2f} & {c('sw_plus_half_live','max_util'):.2f} \\\\")
mlc_path = os.path.join(R2, 'cs6_mlc.json'); MLC_TEXT = '{{MLC_TEXT}}'
if os.path.exists(mlc_path):
    M = json.load(open(mlc_path)); L.append(r'\midrule')
    def agg(runs, getp):
        vals = {k: [] for k in ('mass', 'nom_d', 'full_d', 'half_d', 'nom_u', 'full_u', 'half_u', 'nomC')}
        for r in runs:
            p = getp(r); vals['mass'].append(p['mass']); vals['nom_d'].append(p['nominal']['d_over_L250']); vals['full_d'].append(p['sw_full']['d_over_L250']); vals['half_d'].append(p['sw_half']['d_over_L250'])
            vals['nom_u'].append(p['nominal']['util']); vals['full_u'].append(p['sw_full']['util']); vals['half_u'].append(p['sw_half']['util']); vals['nomC'].append(p['nominal']['C'])
        return {k: (np.mean(v), np.std(v)) for k, v in vals.items()}
    rows = [('CS4 bridge, block, one case, matched', 'sa_block_w1000f_posthoc', lambda r: r),
            ('CS4 bridge, decode, one case, matched', 'greedy_w1000f_posthoc', lambda r: r),
            ('CS4 bridge, block, two cases, matched', 'mlc_block_w1000f', lambda r: r['posthoc']),
            ('CS4 bridge, decode, two cases, matched', 'mlc_greedy_w1000f', lambda r: r['posthoc'])]
    agg_all = {}
    for label, key, getp in rows:
        if key not in M['methods'] or not M['methods'][key]: continue
        a = agg(M['methods'][key], getp); agg_all[key] = a
        L.append(f"{label} & {a['mass'][0]:.0f} & -- & {a['nom_d'][0]:.2f} & {a['full_d'][0]:.2f} & {a['half_d'][0]:.2f}$\\pm${a['half_d'][1]:.2f} & {a['nom_u'][0]:.2f} & {a['full_u'][0]:.2f} & {a['half_u'][0]:.2f} \\\\")
    if 'mlc_block_w1000f' in agg_all and 'sa_block_w1000f_posthoc' in agg_all:
        s1 = agg_all['sa_block_w1000f_posthoc']; s2 = agg_all['mlc_block_w1000f']
        MLC_TEXT = (f"Over ten seeds the matched-mass single-case designs of the block annealer deflect by {s1['half_d'][0]:.2f} $\\pm$ {s1['half_d'][1]:.2f} times span/250 under self-weight and half-span live load, at a utilisation of {s1['half_u'][0]:.2f}, "
                    f"and the two-load-case designs of the same mass ({s2['mass'][0]:.0f} against {s1['mass'][0]:.0f}\\,kg) deflect by {s2['half_d'][0]:.2f} $\\pm$ {s2['half_d'][1]:.2f} times span/250 at a utilisation of {s2['half_u'][0]:.2f}, "
                    f"with {100*(s2['nomC'][0]/s1['nomC'][0]-1):+.0f} percent compliance under the symmetric case; the greedy decode behaves alike (Table~\\ref{{tab:loadcases}}).")
L += [r'\bottomrule', r'\end{tabular}', r'\normalsize', r'\end{table}']
open(os.path.join(OUTDIR, 'tab_loadcases.tex'), 'w').write('\n'.join(L) + '\n')

# ---- prose placeholders ----
ph = dict(SL_RANGE=f"{min(sl_adv.values()):.0f} to {max(sl_adv.values()):.0f}" if sl_adv else '??',
          SL_MASS=f"{np.nanmin(sl_mass):.0f} to {np.nanmax(sl_mass):.0f}" if sl_mass else '??',
          MM_RANGE=f"{min(mm_adv.values()):.0f} to {max(mm_adv.values()):.0f}" if mm_adv else '??',
          W5_ERR=f"{np.nanmean(w5e):.0f}" if w5e else '??', W100_REAL=f"{np.nanmin(w100real):.0f} to {np.nanmax(w100real):.0f}" if w100real else '??', W20_ERR=f"{np.nanmean(w20e):.0f}" if w20e else '??', W100_ERR=f"{np.nanmean(w100e):.0f}" if w100e else '??',
          W100_COST=f"{np.nanmin(w100cost):.0f} to {np.nanmax(w100cost):.0f}" if w100cost else '??',
          BLOCK_VS_GREEDY=(f"between {min(blk_vs_g):+.1f} and {max(blk_vs_g):+.1f} percent" if blk_vs_g else '??'), BLOCK_VS_GREEDY_1000=(f"{-max(blk_vs_g1000):.0f} to {-min(blk_vs_g1000):.0f}" if blk_vs_g1000 else '??'),
          MLC_TEXT=MLC_TEXT, MM_ADV_BY_CASE=json.dumps({k: round(v, 1) for k, v in mm_adv.items()}), SL_ADV_BY_CASE=json.dumps({k: round(v, 1) for k, v in sl_adv.items()}))
if 'CS6' in T and 'sa_strong_w5' in T['CS6']['arms']:
    ph['STRONG_CS6'] = f"{T['CS6']['arms']['sa_strong_w5']['best_mean']:.0f}"
json.dump(ph, open(os.path.join(OUTDIR, 'placeholders.json'), 'w'), indent=1)
for k, v in ph.items():
    print(k, '=', v)

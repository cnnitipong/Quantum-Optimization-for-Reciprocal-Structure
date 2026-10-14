"""figures_geometry_search.py -- figures and summary table of the engagement-ratio search comparison.

Reads results/complex/{cs4_complex,vault_hex_geo}.json (FMQA, GA, random search),
results/geometry_search/geo_methods.json (iterative linearisation, kernel surrogates, hybrid)
and results/geometry_search/dome_enum_all.json (exhaustive dome landscape).
Writes one PDF per panel to figures/panels (override with FIG_OUT) and
results/geometry_search/geo_table.json.  Run from the repository root.
"""
import os, sys, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from exact_mw import exact_mannwhitney
from stats_utils import cliffs_delta

OUT = os.environ.get('FIG_OUT', os.path.join(ROOT, 'figures', 'panels')); os.makedirs(OUT, exist_ok=True)
R4 = os.path.join(ROOT, 'results', 'geometry_search')
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.labelsize': 9, 'xtick.labelsize': 8,
                     'ytick.labelsize': 8, 'legend.fontsize': 8, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.linewidth': 0.7, 'pdf.fonttype': 42, 'mathtext.default': 'regular'})
LAM = np.linspace(0.05, 0.5, 8)
ORDER = ['random', 'ga', 'fmqa', 'qubo2', 'hobo3', 'qml', 'il', 'hybrid']
LAB = {'random': 'Random', 'ga': 'GA', 'fmqa': 'FMQA', 'qubo2': 'QUBO kernel', 'hobo3': 'HOBO kernel',
       'qml': 'Quantum kernel', 'il': 'Iter. linearisation', 'hybrid': 'Hybrid'}
STY = {'random': (':', 'x', '0.55', 'black'), 'ga': ('--', 's', '0.45', 'white'), 'fmqa': ('-', 'o', '0.45', 'white'),
       'qubo2': ('--', None, 'black', None), 'hobo3': ('-.', '^', 'black', 'white'), 'qml': (':', 'D', 'black', 'white'),
       'il': ('-', None, 'black', None), 'hybrid': ('-', 'v', 'black', 'black')}
STORED = {'dome': 'results/complex/cs4_complex.json', 'vault': 'results/complex/vault_hex_geo.json'}
TOL = 0.02


def save(fig, name, extra=()):
    fig.savefig(os.path.join(OUT, name + '.pdf'), bbox_inches='tight', pad_inches=0.02, bbox_extra_artists=list(extra))
    plt.close(fig); print('OK', name)


def load():
    res = json.load(open(os.path.join(R4, 'geo_methods.json'))); data = {}
    for p in ('dome', 'vault'):
        d = json.load(open(os.path.join(ROOT, STORED[p])))['methods']; data[p] = {}
        for k in ('fmqa', 'ga', 'random'):
            rr = d[k]; rr = list(rr.values()) if isinstance(rr, dict) else rr
            H = np.array([np.minimum.accumulate(r['best_history'])[:80] for r in rr])
            X = [r['idx_data'][int(np.argmin(r['y_data']))] if 'idx_data' in r else r['best_x'] for r in rr]
            data[p][k] = (H, [list(map(int, x)) for x in X])
        for m in ('il', 'qubo2', 'hobo3', 'qml', 'hybrid'):
            rr = [res[f'{p}|{m}|{s}'] for s in range(10)]
            data[p][m] = (np.array([r['bsf'][:80] for r in rr]), [r['best_x'] for r in rr])
    return res, data


def main():
    res, data = load()
    enum = json.load(open(os.path.join(R4, 'dome_enum_all.json')))
    Cenum = np.array(enum['C']); opt = {'dome': float(Cenum.min())}
    opt['vault'] = float(min(v[0][:, -1].min() for v in data['vault'].values()))     # best design seen by any method
    table = {}
    for p in ('dome', 'vault'):
        fin = {k: data[p][k][0][:, -1] for k in ORDER}
        rows, ps = {}, []
        for k in ORDER:
            H = data[p][k][0]; v = fin[k]; tgt = opt[p] * (1 + TOL)
            reach = [int(np.argmax(h <= tgt)) + 1 if (h <= tgt).any() else None for h in H]
            rows[k] = dict(mean=float(v.mean()), std=float(v.std()), median=float(np.median(v)), best=float(v.min()),
                           worst=float(v.max()), within2=int(sum(x is not None for x in reach)),
                           hits=int(np.sum(np.abs(v - opt[p]) / opt[p] < 1e-4)),
                           evals_to_2pct_median=float(np.median([x for x in reach if x])) if any(reach) else None,
                           gap_pct_mean=float(np.mean((v - opt[p]) / opt[p] * 100)))
            if k != 'ga':
                rows[k]['p_ga'] = float(exact_mannwhitney(v, fin['ga'])['p_exact_two_sided'])
                rows[k]['delta_ga'] = float(cliffs_delta(v, fin['ga'])); ps.append((k, rows[k]['p_ga']))
            if p == 'dome':
                rows[k]['pct_rank_mean'] = float(np.mean([(Cenum < x - 1e-9).mean() * 100 for x in v]))
        o = sorted(ps, key=lambda t: t[1]); run = 0
        for r, (k, pv) in enumerate(o):
            run = max(run, min(1.0, (len(o) - r) * pv)); rows[k]['p_ga_holm'] = run
        table[p] = dict(reference=opt[p], rows=rows)
        # convergence
        fig, ax = plt.subplots(figsize=(3.1, 2.5))
        for k in ORDER:
            ls, mk, c, fc = STY[k]; mn = data[p][k][0].mean(0)
            ax.plot(np.arange(1, 81), mn, ls=ls, marker=mk, markevery=(4, 12), ms=3.5, color=c, mfc=fc, lw=1.0, label=LAB[k])
        ax.axhline(opt[p], color='0.3', lw=0.6, ls=(0, (1, 2)))
        ax.set_yscale('log'); ax.set_xlim(1, 80); ax.set_ylim(opt[p] * 0.95, opt[p] * 4)
        ax.set_xlabel('Real FE evaluations'); ax.set_ylabel('Mean best compliance [J]')
        save(fig, f'fig_alt_conv_{"a" if p == "dome" else "b"}')
        # box plots of the final best
        fig, ax = plt.subplots(figsize=(3.1, 2.5))
        V = [fin[k] for k in ORDER]
        bp = ax.boxplot(V, widths=0.55, patch_artist=True, showfliers=False, medianprops=dict(color='black', lw=1.2))
        for b, k in zip(bp['boxes'], ORDER):
            b.set_facecolor('white' if k in ('random', 'ga', 'fmqa') else '0.8'); b.set_edgecolor('black'); b.set_linewidth(0.7)
        rng = np.random.default_rng(0)
        for i, v in enumerate(V):
            ax.scatter(i + 1 + rng.uniform(-0.15, 0.15, len(v)), v, s=6, c='black', zorder=3, linewidths=0)
        ax.axhline(opt[p], color='0.3', lw=0.6, ls=(0, (1, 2)))
        ax.set_yscale('log'); ax.set_xticks(range(1, 9)); ax.set_xticklabels([LAB[k] for k in ORDER], rotation=55, ha='right')
        ax.set_ylabel('Best compliance at 80 evaluations [J]')
        save(fig, f'fig_alt_box_{"a" if p == "dome" else "b"}')
        # attainment: share of seeds within 2 percent of the reference
        fig, ax = plt.subplots(figsize=(3.1, 2.5))
        for k in ORDER:
            ls, mk, c, fc = STY[k]; H = data[p][k][0]
            ax.plot(np.arange(1, 81), 100 * (H <= opt[p] * (1 + TOL)).mean(0), ls=ls, marker=mk, markevery=(4, 12), ms=3.5,
                    color=c, mfc=fc, lw=1.0, label=LAB[k])
        ax.set_xlim(1, 80); ax.set_ylim(-3, 103); ax.set_xlabel('Real FE evaluations'); ax.set_ylabel('Seeds within 2% of reference [%]')
        save(fig, f'fig_alt_att_{"a" if p == "dome" else "b"}')
        # lambda profile of the best designs
        nv = 4 if p == 'dome' else 6
        fig, ax = plt.subplots(figsize=(3.1, 2.5))
        for k in ('fmqa', 'ga', 'il', 'hybrid'):
            ls, mk, c, fc = STY[k]
            L = np.array([[LAM[x[i]] for i in range(nv)] for x in data[p][k][1]])
            ax.errorbar(np.arange(1, nv + 1) + {'fmqa': -0.15, 'ga': -0.05, 'il': 0.05, 'hybrid': 0.15}[k], L.mean(0), yerr=L.std(0),
                        ls=ls, marker=mk or 's', ms=3.5, color=c, mfc=fc or 'black', lw=0.9, capsize=2, label=LAB[k])
        if p == 'dome':
            xb = enum['x'][int(np.argmin(Cenum))]
            ax.plot(np.arange(1, 5), LAM[xb], ls='none', marker='*', ms=9, color='black', mfc='white', label='Exact optimum')
        ax.set_xticks(range(1, nv + 1)); ax.set_xlabel('Ring family' if p == 'dome' else 'Transverse band')
        ax.set_ylabel(r'Engagement ratio $\lambda$ of best design'); ax.set_ylim(0, 0.55)
        save(fig, f'fig_alt_lam_{"a" if p == "dome" else "b"}')
    # shared legend panel
    fig = plt.figure(figsize=(6.4, 0.45)); hs = []
    for k in ORDER:
        ls, mk, c, fc = STY[k]
        hs.append(plt.Line2D([], [], ls=ls, marker=mk, color=c, mfc=fc, ms=4, lw=1.0, label=LAB[k]))
    hs.append(plt.Line2D([], [], ls=(0, (1, 2)), color='0.3', lw=0.6, label='Reference optimum'))
    fig.legend(handles=hs, loc='center', ncol=5, frameon=False, fontsize=8); save(fig, 'fig_alt_legend')
    # exhaustive dome landscape with the final designs of each method
    fig, (a1, ax) = plt.subplots(2, 1, figsize=(6.4, 3.4), sharex=True, gridspec_kw=dict(height_ratios=[1.25, 1], hspace=0.08))
    hi = max(np.quantile(Cenum, 0.995), max(data['dome'][k][0][:, -1].max() for k in ORDER)) * 1.02
    bins = np.logspace(np.log10(Cenum.min() * 0.98), np.log10(hi), 70)
    ax.hist(Cenum, bins=bins, color='0.8', edgecolor='0.4', lw=0.4)
    ax.set_xscale('log'); ax.set_xlabel('Compliance [J]'); ax.set_ylabel('Designs (of 4096)')
    for i, k in enumerate(ORDER):
        v = data['dome'][k][0][:, -1]; ls, mk, c, fc = STY[k]
        a1.scatter(v, np.full(len(v), i), marker=mk or 'o', s=16, facecolors=fc if fc else 'black',
                   edgecolors='black', linewidths=0.6) if mk not in ('x',) else a1.scatter(v, np.full(len(v), i), marker='x', s=16, c='black', linewidths=0.7)
    a1.set_yticks(range(len(ORDER))); a1.set_yticklabels([LAB[k] for k in ORDER]); a1.set_ylim(-0.6, len(ORDER) - 0.4)
    a1.spines['bottom'].set_visible(False); a1.tick_params(axis='x', bottom=False)
    for a in (a1, ax):
        a.axvline(Cenum.min(), color='black', lw=0.7, ls=(0, (1, 2)))
    ax.set_xlim(bins[0], bins[-1])
    save(fig, 'fig_alt_land')
    # geometry of the FMQA-best and of the best found designs (plan)
    _argv = sys.argv; sys.argv = ['x']
    import cs4_dome2 as DOME, cs_vault_hex2_geo as VAULT
    sys.argv = _argv
    best_f = {p: min(range(10), key=lambda s: data[p]['fmqa'][0][s, -1]) for p in ('dome', 'vault')}
    picks = [('a', DOME, data['dome']['fmqa'][1][best_f['dome']]), ('b', DOME, enum['x'][int(np.argmin(Cenum))] + [7]),
             ('c', VAULT, data['vault']['fmqa'][1][best_f['vault']]),
             ('d', VAULT, data['vault']['hybrid'][1][int(np.argmin(data['vault']['hybrid'][0][:, -1]))])]
    for tag, mod, x in picks:
        m = mod._build(x); P = np.array(m.nodes)
        bi = [e for e in m.elements if e['kind'] == 'beam']
        fig, ax = plt.subplots(figsize=(3.0, 2.6 if mod is DOME else 2.4))
        ax.add_collection(LineCollection([[P[e['i'], :2], P[e['j'], :2]] for e in bi], colors='black', linewidths=0.7))
        s = sorted(m.support_idx); ax.scatter(P[s, 0], P[s, 1], s=6, marker='s', c='black')
        ax.set_aspect('equal'); ax.autoscale_view(); ax.set_axis_off()
        save(fig, f'fig_alt_geo_{tag}')
        table.setdefault('geometries', {})[tag] = [int(v) for v in x]
    json.dump(table, open(os.path.join(R4, 'geo_table.json'), 'w'), indent=1)
    for p in ('dome', 'vault'):
        print('==', p, 'reference', round(table[p]['reference'], 1))
        for k in ORDER:
            r = table[p]['rows'][k]
            print(f"  {LAB[k]:20s} {r['mean']:7.1f}±{r['std']:6.1f} med {r['median']:7.1f} best {r['best']:7.1f} gap {r['gap_pct_mean']:5.1f}% hits {r['hits']} w2 {r['within2']} e2 {r['evals_to_2pct_median']} pH {r.get('p_ga_holm', float('nan')):.2g} d {r.get('delta_ga', float('nan')):.2f} rank {r.get('pct_rank_mean', float('nan')):.2f}")


if __name__ == '__main__':
    main()

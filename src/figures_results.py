"""Generate the figure panels of the revised manuscript (one PDF per panel).

Run from the repository root:  python3 -I src/figures_results.py
Every panel is wrapped: a missing file or series gives a printed warning and
the panel (or the series) is skipped.  Re-run when more result files exist.
"""
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

import json
import traceback
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

OUT = os.environ.get('FIG_OUT', os.path.join(ROOT, 'figures', 'panels'))
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 9,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.edgecolor': 'black',
    'axes.linewidth': 0.7,
    'xtick.color': 'black',
    'ytick.color': 'black',
    'text.color': 'black',
    'axes.labelcolor': 'black',
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'pdf.fonttype': 42,
    'mathtext.default': 'regular',
})

HALF, TYP, FULL = (3.1, 2.4), (2.1, 2.1), (6.4, 2.6)
STATUS = {}


class Skip(Exception):
    pass


def panel(name):
    def deco(fn):
        def wrapped(*a, **k):
            plt.close('all')
            try:
                fn(*a, **k)
                STATUS[name] = 'produced'
                print('OK    %s' % name)
            except Skip as e:
                STATUS[name] = 'skipped: %s' % e
                print('WARN  %s skipped: %s' % (name, e))
            except Exception as e:  # robust against unexpected layouts
                STATUS[name] = 'skipped: %s: %s' % (type(e).__name__, e)
                print('WARN  %s skipped: %s: %s' % (name, type(e).__name__, e))
                traceback.print_exc(limit=2)
            finally:
                plt.close('all')
        return wrapped
    return deco


def save(fig, fname, extra=()):
    path = os.path.join(OUT, fname)
    arts = list(extra)
    for ax in fig.axes:
        arts += [ax.xaxis.label, ax.yaxis.label] + list(ax.texts)
        lg = ax.get_legend()
        if lg is not None:
            arts.append(lg)
    fig.savefig(path, bbox_inches='tight', pad_inches=0.02, bbox_extra_artists=arts)
    plt.close(fig)


# ----------------------------------------------------------------- data utils
_cache = {}


def load(rel):
    p = os.path.join(ROOT, rel)
    if p in _cache:
        return _cache[p]
    if not os.path.exists(p):
        raise Skip('missing file %s' % rel)
    with open(p) as f:
        d = json.load(f)
    _cache[p] = d
    return d


def runs_of(entry):
    """list of run dicts from a list or a dict keyed by seed string."""
    if isinstance(entry, dict):
        keys = sorted(entry.keys(), key=lambda k: (not str(k).lstrip('-').isdigit(), int(k) if str(k).lstrip('-').isdigit() else 0, str(k)))
        return [entry[k] for k in keys]
    return list(entry)


def get_runs(rel, method):
    d = load(rel)
    m = d.get('methods', d)
    if method not in m:
        raise Skip('series %s not in %s' % (method, rel))
    r = runs_of(m[method])
    if not r:
        raise Skip('series %s empty in %s' % (method, rel))
    return r


def seed0(runs):
    for r in runs:
        if isinstance(r, dict) and r.get('seed') == 0:
            return r
    return runs[0]


def bsf_stack(runs, n, key='compliance_history'):
    arrs = []
    for r in runs:
        h = np.asarray(r[key], float)[:n]
        arrs.append(np.minimum.accumulate(h))
    L = min(len(a) for a in arrs)
    return np.vstack([a[:L] for a in arrs])


def legend_below(fig, ax, ncol=2, dy=-0.27, **kw):
    h, l = ax.get_legend_handles_labels()
    return fig.legend(h, l, loc='upper center', bbox_to_anchor=(0.5, dy),
                      bbox_transform=ax.transAxes, ncol=ncol, frameon=False, **kw)


def style_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# ------------------------------------------------------------------ 1 typology
def draw_model(model, kind, elev=28, azim=-55, figsize=TYP, lw_beam=0.9):
    nodes = np.array([[float(c) for c in n] for n in model.nodes])
    beams = [(e['i'], e['j']) for e in model.elements if e['kind'] == 'beam']
    links = [(e['i'], e['j']) for e in model.elements if e['kind'] == 'link']
    sup = sorted(model.support_idx)
    ld = list(model.load_idx)
    fig = plt.figure(figsize=figsize)
    if kind == 'plan':
        ax = fig.add_subplot(111)
        seg = lambda pairs: [[nodes[i, :2], nodes[j, :2]] for i, j in pairs]
        ax.add_collection(LineCollection(seg(beams), colors='black', linewidths=lw_beam))
        ax.add_collection(LineCollection(seg(links), colors='0.6', linewidths=0.6))
        ax.scatter(nodes[ld, 0], nodes[ld, 1], s=2, c='0.6', zorder=3, linewidths=0)
        ax.scatter(nodes[sup, 0], nodes[sup, 1], s=6, c='black', marker='s', zorder=4, linewidths=0)
        ax.set_aspect('equal')
        ax.autoscale_view()
        ax.set_xlim(nodes[:, 0].min() - 0.05, nodes[:, 0].max() + 0.05)
        ax.set_ylim(nodes[:, 1].min() - 0.05, nodes[:, 1].max() + 0.05)
        ax.set_axis_off()
    else:
        ax = fig.add_subplot(111, projection='3d')
        seg = lambda pairs: [[nodes[i], nodes[j]] for i, j in pairs]
        ax.add_collection3d(Line3DCollection(seg(beams), colors='black', linewidths=lw_beam))
        ax.add_collection3d(Line3DCollection(seg(links), colors='0.6', linewidths=0.6))
        ax.scatter(nodes[ld, 0], nodes[ld, 1], nodes[ld, 2], s=2, c='0.6', linewidths=0, depthshade=False)
        ax.scatter(nodes[sup, 0], nodes[sup, 1], nodes[sup, 2], s=6, c='black', marker='s', linewidths=0, depthshade=False)
        rng = np.maximum(nodes.max(0) - nodes.min(0), 1e-6)
        ax.set_xlim(nodes[:, 0].min(), nodes[:, 0].max())
        ax.set_ylim(nodes[:, 1].min(), nodes[:, 1].max())
        ax.set_zlim(nodes[:, 2].min(), nodes[:, 2].max())
        ax.set_box_aspect(tuple(rng))
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
    return fig, ax, nodes


def build_models():
    from reciprocal_geom_v2 import (fan_unit, square_nexorade, hex_nexorade,
                                    rf_dome2, rf_vault_hex2, davinci_bridge2)
    return {
        'a': ('plan', lambda: fan_unit(4, R=3.0, lam=0.25), 28, -55),
        'b': ('plan', lambda: square_nexorade(6, 6, 2.5, 0.25), 28, -55),
        'c': ('plan', lambda: hex_nexorade(3, 2.2, 0.25), 28, -55),
        'd': ('3d', lambda: rf_dome2(n_sect=7, n_ring=4, span=18.5, rise=3.15, lam=0.25, max_group_rings=4), 28, -55),
        'e': ('3d', lambda: rf_vault_hex2(lam_bands=(0.25,) * 6), 28, -55),
        'f': ('3d', lambda: davinci_bridge2(n_bays=24, span=20.0, width=6.0, lam=0.25), 22, -60),
    }


def make_typology():
    try:
        specs = build_models()
    except Exception as e:
        for k in 'abcdef':
            print('WARN  fig_typ_%s skipped: cannot import geometry (%s)' % (k, e))
            STATUS['fig_typ_%s' % k] = 'skipped: import error'
        return
    for k, (kind, builder, elev, azim) in specs.items():
        @panel('fig_typ_%s' % k)
        def _p(k=k, kind=kind, builder=builder, elev=elev, azim=azim):
            model = builder()
            fig, ax, _ = draw_model(model, kind, elev, azim)
            save(fig, 'fig_typ_%s.pdf' % k)
        _p()


# --------------------------------------------------------------- curve helpers
def plot_curve(ax, x, y, st, label):
    ax.plot(x, y, linestyle=st['ls'], color=st['c'], lw=st.get('lw', 1.1),
            marker=st.get('m', None), markersize=st.get('ms', 4.0),
            markerfacecolor='white' if st.get('hollow') else st['c'],
            markeredgecolor=st['c'], markeredgewidth=0.8,
            markevery=st.get('me', None), label=label)


def finish_logy(ax, lo, hi, xlabel='Real FE evaluations', ylabel='Compliance [J]'):
    ax.set_yscale('log')
    ax.set_ylim(lo / 1.12, hi * 1.12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    style_axes(ax)


# ----------------------------------------------------------------- 2 geometry
GEOM_STYLES = {
    'fmqa': dict(ls='-', c='black', m='o', hollow=True, me=(0, 8)),
    'ga': dict(ls='--', c='0.35', m='s', hollow=True, me=(3, 8)),
    'random': dict(ls=':', c='0.6', m='x', me=(5, 8)),
}
GEOM_LABELS = {'fmqa': 'FMQA', 'ga': 'GA, population 20', 'random': 'Random search'}


def geom_panel(fname, rel):
    @panel(fname.replace('.pdf', ''))
    def _p():
        fig, ax = plt.subplots(figsize=HALF)
        lo, hi, n_ok = np.inf, 0, 0
        for m in ('fmqa', 'ga', 'random'):
            try:
                runs = get_runs(rel, m)
                key = 'best_history' if 'best_history' in runs[0] else 'compliance_history'
                S = bsf_stack(runs, 80, key)
            except Skip as e:
                print('WARN  %s: %s' % (fname, e))
                continue
            mean = S.mean(0)
            x = np.arange(1, len(mean) + 1)
            plot_curve(ax, x, mean, GEOM_STYLES[m], GEOM_LABELS[m])
            lo, hi, n_ok = min(lo, mean.min()), max(hi, mean.max()), n_ok + 1
        if n_ok == 0:
            raise Skip('no series')
        finish_logy(ax, lo, hi)
        ax.set_xlim(1, 80)
        leg = ax.legend(loc='best', frameon=False)
        save(fig, fname, [leg])
    _p()


# ------------------------------------------------------------- 3 convergence
ARM_STYLES = [
    ('Single-flip annealer', dict(ls='-', c='black')),
    ('Block annealer', dict(ls='-', c='black', m='o', hollow=True, me=(0, 5))),
    ('Greedy decode', dict(ls='--', c='black')),
    ('FE local search', dict(ls='-.', c='0.35', m='^', hollow=True, me=(1, 5))),
    ('GA, population 20', dict(ls=':', c='0.35', m='s', hollow=True, me=(2, 5))),
    ('Random search', dict(ls=':', c='0.6', m='x', me=(3, 5))),
]
CONV_SRC = {
    'a': {'sa_qubo': ('results/cs2_v2.json', 'sa_qubo'),
          'greedy': ('results/r3_ablation_v2.json', 'greedy_w5'),
          'ga': ('results/cs2_v2.json', 'ga'),
          'random': ('results/cs2_v2.json', 'random'),
          'block': ('results/benchmark/cs2.json', 'sa_block_w5'),
          'ls': ('results/benchmark/cs2.json', 'ls_w5')},
    'b': {'sa_qubo': ('results/complex/cs6_complex.json', 'sa_qubo'),
          'greedy': ('results/complex/cs6_complex.json', 'greedy_w5'),
          'ga': ('results/complex/cs6_complex.json', 'ga'),
          'random': ('results/complex/cs6_complex.json', 'random'),
          'block': ('results/benchmark/cs6.json', 'sa_block_w5'),
          'ls': ('results/benchmark/cs6.json', 'ls_w5')},
    'c': {'sa_qubo': ('results/complex/dome_sizing.json', 'sa_qubo'),
          'greedy': ('results/complex/dome_sizing.json', 'greedy_w5'),
          'ga': ('results/complex/dome_sizing.json', 'ga'),
          'random': ('results/complex/dome_sizing.json', 'random'),
          'block': ('results/benchmark/dome.json', 'sa_block_w5'),
          'ls': ('results/benchmark/dome.json', 'ls_w5')},
    'd': {'sa_qubo': ('results/complex/vault_hex_sizing.json', 'sa_qubo'),
          'greedy': ('results/complex/vault_hex_sizing.json', 'greedy_w5'),
          'ga': ('results/complex/vault_hex_sizing.json', 'ga'),
          'random': ('results/complex/vault_hex_sizing.json', 'random'),
          'block': ('results/benchmark/vault.json', 'sa_block_w5'),
          'ls': ('results/benchmark/vault.json', 'ls_w5')},
}
ARM_ORDER = ['sa_qubo', 'block', 'greedy', 'ls', 'ga', 'random']


def conv_panel(letter, nmax=50):
    fname = 'fig_conv_%s' % letter

    @panel(fname)
    def _p():
        fig, ax = plt.subplots(figsize=HALF)
        lo, hi, n_ok = np.inf, 0, 0
        first = True
        for arm, (label, st) in zip(ARM_ORDER, ARM_STYLES):
            rel, key = CONV_SRC[letter][arm]
            try:
                S = bsf_stack(get_runs(rel, key), nmax)
            except Skip as e:
                print('WARN  %s: %s' % (fname, e))
                continue
            mean = S.mean(0)
            x = np.arange(1, len(mean) + 1)
            plot_curve(ax, x, mean, st, label)
            cur_lo, cur_hi = mean.min(), mean.max()
            if first:
                sd = S.std(0)
                ax.fill_between(x, np.maximum(mean - sd, 1e-12), mean + sd, color='0.6',
                                alpha=0.15, linewidth=0)
                cur_lo, cur_hi = min(cur_lo, (mean - sd).min()), max(cur_hi, (mean + sd).max())
                first = False
            lo, hi, n_ok = min(lo, cur_lo), max(hi, cur_hi), n_ok + 1
        if n_ok == 0:
            raise Skip('no series available')
        finish_logy(ax, max(lo, 1e-12), hi)
        ax.set_xlim(1, nmax)
        extra = []
        if letter == 'a':
            extra.append(legend_below(fig, ax, 2))
        save(fig, fname + '.pdf', extra)
    _p()


# --------------------------------------------------------------------- 4 design
def pick_design(rel, prefer='sa_block_w1000f', fallback='sa_block_w5'):
    d = load(rel)
    m = d['methods']
    key = prefer if prefer in m else fallback
    if key not in m:
        raise Skip('neither %s nor %s in %s' % (prefer, fallback, rel))
    if key != prefer:
        print('NOTE  %s: %s absent, using %s' % (rel, prefer, key))
    target = d['mass_target']
    best = None
    for run in runs_of(m[key]):
        C = np.asarray(run['compliance_history'], float)
        M = np.asarray(run['mass_history'], float)
        ok = np.abs(M - target) <= 0.05 * target
        for it in range(len(C)):
            if ok[it] and (best is None or C[it] < best[0]):
                best = (C[it], run['idx_history'][it], run.get('seed'), it, True)
    if best is None:
        print('NOTE  %s: no iteration within 5%% of mass target, using lowest compliance' % rel)
        for run in runs_of(m[key]):
            C = np.asarray(run['compliance_history'], float)
            it = int(np.argmin(C))
            if best is None or C[it] < best[0]:
                best = (C[it], run['idx_history'][it], run.get('seed'), it, False)
    return best, key


def design_panel(fname, rel, builder, kind, elev, azim):
    @panel(fname.replace('.pdf', ''))
    def _p():
        from sections import CHS_CATALOG
        (C, idx, seed, it, matched), key = pick_design(rel)
        model = builder()
        nodes = np.array([[float(c) for c in n] for n in model.nodes])
        D = [CHS_CATALOG[i]['D_mm'] for i in range(len(CHS_CATALOG))]
        lwf = lambda dmm: 0.25 + 2.2 * dmm / 244.5
        beams = [(e['i'], e['j'], lwf(D[int(idx[e['group']])])) for e in model.elements
                 if e['kind'] == 'beam' and e.get('group') is not None]
        links = [(e['i'], e['j']) for e in model.elements if e['kind'] == 'link']
        sup = sorted(model.support_idx)
        fig = plt.figure(figsize=HALF)
        if kind == 'plan':
            ax = fig.add_subplot(111)
            ax.add_collection(LineCollection([[nodes[i, :2], nodes[j, :2]] for i, j in links],
                                             colors='0.6', linewidths=0.5))
            ax.add_collection(LineCollection([[nodes[i, :2], nodes[j, :2]] for i, j, _ in beams],
                                             colors='black', linewidths=[w for _, _, w in beams],
                                             capstyle='round'))
            ax.scatter(nodes[sup, 0], nodes[sup, 1], s=8, c='black', marker='s', linewidths=0, zorder=4)
            ax.set_xlim(nodes[:, 0].min() - 0.1, nodes[:, 0].max() + 0.1)
            ax.set_ylim(nodes[:, 1].min() - 0.1, nodes[:, 1].max() + 0.1)
            ax.set_aspect('equal', adjustable='datalim')
            ax.set_axis_off()
        else:
            ax = fig.add_subplot(111, projection='3d')
            ax.add_collection3d(Line3DCollection([[nodes[i], nodes[j]] for i, j in links],
                                                 colors='0.6', linewidths=0.5))
            ax.add_collection3d(Line3DCollection([[nodes[i], nodes[j]] for i, j, _ in beams],
                                                 colors='black', linewidths=[w for _, _, w in beams]))
            ax.scatter(nodes[sup, 0], nodes[sup, 1], nodes[sup, 2], s=8, c='black', marker='s',
                       linewidths=0, depthshade=False)
            rng = np.maximum(nodes.max(0) - nodes.min(0), 1e-6)
            ax.set_xlim(nodes[:, 0].min(), nodes[:, 0].max())
            ax.set_ylim(nodes[:, 1].min(), nodes[:, 1].max())
            ax.set_zlim(nodes[:, 2].min(), nodes[:, 2].max())
            ax.set_box_aspect(tuple(rng))
            ax.view_init(elev=elev, azim=azim)
            ax.set_axis_off()
        handles = [Line2D([0], [0], color='black', lw=lwf(d)) for d in (60.3, 139.7, 244.5)]
        labels = ['60.3 mm', '139.7 mm', '244.5 mm']
        # placed below the model so that it covers no member
        leg = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.02),
                         ncol=3, frameon=False, title='Outer diameter', handlelength=1.6,
                         columnspacing=1.0, handletextpad=0.5)
        leg.get_title().set_fontsize(8)
        print('NOTE  %s: method %s seed %s iteration %s compliance %.1f' % (fname, key, seed, it, C))
        save(fig, fname, [leg])
    _p()


def make_design_spec():
    """fig_design_a: CS2 nexorade (plan view); fig_design_b: bridge (3-D view, elev 22, azim -60)."""
    try:
        from reciprocal_geom_v2 import square_nexorade, davinci_bridge2
    except Exception as e:
        print('WARN  design panels skipped: %s' % e)
        for n in ('fig_design_a', 'fig_design_b'):
            STATUS[n] = 'skipped: import error'
        return
    sq = lambda: square_nexorade(6, 6, 2.5, 0.25)
    br = lambda: davinci_bridge2(n_bays=24, span=20.0, width=6.0, lam=0.25)
    design_panel('fig_design_a.pdf', 'results/benchmark/cs2.json', sq, 'plan', 90, -90)
    design_panel('fig_design_b.pdf', 'results/benchmark/cs6.json', br, '3d', 22, -60)


# --------------------------------------------------------------------- 5 barrier
def barrier_a():
    @panel('fig_barrier_a')
    def _p():
        d = load('results/benchmark/cs6_mlc.json')
        if 'barrier_diag_seed0_w5' not in d:
            raise Skip('key barrier_diag_seed0_w5 absent')
        rows = d['barrier_diag_seed0_w5']
        it = np.array([r['it'] for r in rows], float)
        g = np.array([r['E_greedy'] for r in rows], float)
        fig, ax = plt.subplots(figsize=HALF)
        series = [('E_sa', 'Single-flip, 25 x 90', dict(c='black', m='o', hollow=False)),
                  ('E_sa16', 'Single-flip, 100 x 400', dict(c='0.5', m='s', hollow=True)),
                  ('E_block', 'Block annealer', dict(c='black', m='x', hollow=False))]
        allv = [0.0]
        n_ok = 0
        for k, lab, st in series:
            if k not in rows[0]:
                print('WARN  fig_barrier_a: series %s absent' % k)
                continue
            y = np.array([r[k] for r in rows], float) - g
            ax.plot(it, y, ls='none', marker=st['m'], markersize=3, color=st['c'],
                    markerfacecolor='white' if st['hollow'] else st['c'],
                    markeredgecolor=st['c'], markeredgewidth=0.7, label=lab)
            allv += list(y)
            n_ok += 1
        if n_ok == 0:
            raise Skip('no energy series')
        ax.axhline(0, color='0.6', lw=0.8, ls=':', label='Decode')
        pad = 0.08 * (max(allv) - min(allv) + 1e-12)
        ax.set_ylim(min(allv) - pad, max(allv) + pad)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Energy gap to decode')
        style_axes(ax)
        leg = legend_below(fig, ax, 2)
        save(fig, 'fig_barrier_a.pdf', [leg])
    _p()


def barrier_b():
    @panel('fig_barrier_b')
    def _p():
        arms = [('Greedy decode', 'results/complex/cs6_complex.json', 'greedy_w5', dict(ls='--', c='black')),
                ('Single-flip, 25 x 90', 'results/complex/cs6_complex.json', 'sa_paired_w5',
                 dict(ls='-', c='black')),
                ('Single-flip, 100 x 400', 'results/benchmark/cs6.json', 'sa_strong_w5',
                 dict(ls='-.', c='0.35', m='s', hollow=True, me=(0, 4))),
                ('Block annealer', 'results/benchmark/cs6.json', 'sa_block_w5',
                 dict(ls='-', c='black', m='o', hollow=True, me=(2, 4)))]
        fig, ax = plt.subplots(figsize=HALF)
        lo, hi, n_ok = np.inf, 0, 0
        for lab, rel, key, st in arms:
            try:
                run = seed0(get_runs(rel, key))
            except Skip as e:
                print('WARN  fig_barrier_b: %s' % e)
                continue
            y = np.asarray(run['compliance_history'], float)
            plot_curve(ax, np.arange(1, len(y) + 1), y, st, lab)
            lo, hi, n_ok = min(lo, y.min()), max(hi, y.max()), n_ok + 1
        if n_ok == 0:
            raise Skip('no series')
        finish_logy(ax, lo, hi, xlabel='Iteration')
        leg = legend_below(fig, ax, 2)
        save(fig, 'fig_barrier_b.pdf', [leg])
    _p()


# --------------------------------------------------------------------- 6 Pareto
def nondominated(P):
    """P: (n,2) minimise both. returns the non-dominated subset sorted by x."""
    P = np.unique(P, axis=0)
    P = P[np.lexsort((P[:, 1], P[:, 0]))]
    keep, best = [], np.inf
    for p in P:
        if p[1] < best:
            keep.append(p)
            best = p[1]
    return np.array(keep)


def pareto_a():
    @panel('fig_pareto_a')
    def _p():
        d = load('results/cs5_v2.json')
        a = load('results/cs5_analysis_v2.json')
        (cmin, cmax), (mmin, mmax) = a['compliance_range'], a['mass_range']

        def pts(key):
            P = []
            for run in runs_of(d[key]):
                for q in run:
                    P.append(((q['mass'] - mmin) / (mmax - mmin), (q['compliance'] - cmin) / (cmax - cmin)))
            return np.array(P)
        fig, ax = plt.subplots(figsize=(3.4, 2.6))
        fq = nondominated(pts('sa_points'))
        fn = nondominated(pts('nsga_points'))
        ax.plot(fn[:, 0], fn[:, 1], ls='none', marker='x', color='0.5', markersize=4.5,
                markeredgewidth=0.8, label='NSGA-II')
        ax.plot(fq[:, 0], fq[:, 1], ls='none', marker='o', markerfacecolor='white',
                markeredgecolor='black', markersize=4.0, markeredgewidth=0.8, label='Weighted-sum QUBO')
        ax.set_xlabel('Normalised mass')
        ax.set_ylabel('Normalised compliance')
        ax.set_xlim(-0.03, 1.05)
        ax.set_ylim(-0.03, 1.05)
        style_axes(ax)
        leg = ax.legend(loc='upper right', frameon=False)
        save(fig, 'fig_pareto_a.pdf', [leg])
    _p()


def pareto_b():
    @panel('fig_pareto_b')
    def _p():
        a = load('results/cs5_analysis_v2.json')
        ps = a['per_seed']
        hq = np.array([v['sa_hv'] for k, v in sorted(ps.items())])
        hn = np.array([v['nsga_hv'] for k, v in sorted(ps.items())])
        fig, ax = plt.subplots(figsize=(2.6, 2.6))
        rs = np.random.RandomState(1)
        for x0, h, mk, c, mf in ((0, hq, 'o', 'black', 'white'), (1, hn, 'x', '0.5', '0.5')):
            ax.plot(x0 + rs.uniform(-0.12, 0.12, len(h)), h, ls='none', marker=mk, color=c,
                    markerfacecolor=mf, markeredgecolor=c, markersize=4.5, markeredgewidth=0.8)
            ax.errorbar([x0 + 0.3], [h.mean()], yerr=[h.std(ddof=1) if len(h) > 1 else 0.0],
                        fmt='s', color='black', markersize=4, capsize=3, lw=0.9)
        ax.set_xticks([0.15, 1.15])
        ax.set_xticklabels(['QUBO', 'NSGA-II'])
        ax.set_xlim(-0.5, 1.7)
        allv = np.concatenate([hq, hn])
        pad = 0.15 * (allv.max() - allv.min() + 1e-12)
        ax.set_ylim(allv.min() - pad - hq.std(), allv.max() + pad + hq.std())
        ax.set_ylabel('Hypervolume')
        style_axes(ax)
        save(fig, 'fig_pareto_b.pdf')
    _p()


# ------------------------------------------------------------------------ main
def main():
    make_typology()
    geom_panel('fig_geom_a.pdf', 'results/complex/cs4_complex.json')
    geom_panel('fig_geom_b.pdf', 'results/complex/vault_hex_geo.json')
    for L in 'abcd':
        conv_panel(L)
    make_design_spec()
    barrier_a()
    barrier_b()
    pareto_a()
    pareto_b()
    print('\nSUMMARY')
    for k in sorted(STATUS):
        print('  %-14s %s' % (k, STATUS[k]))


if __name__ == '__main__':
    main()

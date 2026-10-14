"""figures_structures.py -- extended figure set (form, shape, analysis, member sizes).

Run from the package root:  python3 -I src/figures_structures.py [group ...]
Groups: bearing groups elev lam def sed link size sechist mc bridge cs1 pen util
Member-wise maps (sizes, groups, utilisation, displacement, energy, connector
force) are in colour; line charts stay greyscale.  Every number quoted from these
panels is written to results/structures/figdata.json.
"""
import os, sys, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, Normalize, LogNorm, ListedColormap, TwoSlopeNorm
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from reciprocal_geom import build_frame, model_mass, LINK_SECTION
from reciprocal_geom_v2 import fan_unit, square_nexorade, hex_nexorade, davinci_bridge2
from sections import CHS_CATALOG
from sizing_benchmark import make_case

OUT = os.environ.get('FIG_OUT', os.path.join(ROOT, 'figures', 'panels'))
DATA = os.path.join(ROOT, 'results', 'structures'); os.makedirs(DATA, exist_ok=True); os.makedirs(OUT, exist_ok=True)
FD_PATH = os.path.join(DATA, 'figdata.json')
FD = json.load(open(FD_PATH)) if os.path.exists(FD_PATH) else {}
G = 9.81; FY = 355e6
D_MM = [s['D_mm'] for s in CHS_CATALOG]

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.labelsize': 9, 'xtick.labelsize': 8,
                     'ytick.labelsize': 8, 'legend.fontsize': 8, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.linewidth': 0.7, 'pdf.fonttype': 42, 'mathtext.default': 'regular',
                     'savefig.facecolor': 'white'})
GREY = ListedColormap(plt.get_cmap('Greys')(np.linspace(0.22, 1.0, 256)))
SIZE_CMAP = ListedColormap(plt.get_cmap('Greys')(np.linspace(0.22, 1.0, 8)))
SIZE_NORM = BoundaryNorm(np.arange(-0.5, 8.5, 1), 8)
CASE_LABEL = {'cs2': 'CS2', 'cs3': 'CS3', 'cs6': 'CS4', 'dome': 'CS5', 'vault': 'CS6'}
VIEW = {'dome_ref': ('plan', 0, 0), 'vault_ref': ('plan', 0, 0), 'fan': ('plan', 0, 0), 'cs2': ('plan', 0, 0), 'cs3': ('plan', 0, 0),
        'dome': ('plan', 0, 0), 'vault': ('plan', 0, 0), 'cs6': ('plan', 0, 0)}
VIEW3D = {'dome_ref': ('3d', 30, -55), 'vault_ref': ('3d', 28, -50), 'fan': ('3d', 30, -55), 'cs2': ('3d', 30, -55), 'cs3': ('3d', 30, -55),
          'dome': ('3d', 30, -55), 'vault': ('3d', 28, -50), 'cs6': ('3d', 20, -65)}


def save(fig, name, extra=()):
    arts = list(extra)
    for ax in fig.axes:
        arts += [ax.xaxis.label, ax.yaxis.label] + list(ax.texts)
        if ax.get_legend() is not None:
            arts.append(ax.get_legend())
    fig.savefig(os.path.join(OUT, name + '.pdf'), bbox_inches='tight', pad_inches=0.02, bbox_extra_artists=arts)
    plt.close(fig); print('OK', name, flush=True)


# ------------------------------------------------------------------ models
_models = {}


def model_of(case):
    if case not in _models:
        if case == 'fan':
            m = fan_unit(4, R=3.0, lam=0.25); _models[case] = (m, m.meta.get('load_per_node', -2200.0))
        elif case in ('dome_ref', 'vault_ref'):
            from reciprocal_geom_v2 import rf_dome2, rf_vault_hex2
            m = (rf_dome2(n_sect=7, n_ring=4, span=18.5, rise=3.15, lam=0.25, max_group_rings=4) if case == 'dome_ref'
                 else rf_vault_hex2(lam_bands=(0.25,) * 6))
            _models[case] = (m, m.meta['load_per_node'])
        else:
            _models[case] = make_case(case)
    return _models[case]


def gs_of(model, idx):
    return {k: CHS_CATALOG[int(idx[k])] for k in range(model.n_groups)}


def analyse(model, idx, load, sw=False, live='full', link=None):
    gs = gs_of(model, idx)
    frm = build_frame(model, gs, link_section=link, load_per_node=0.0)
    if sw:
        for e in model.elements:
            ni, nj = np.array(model.nodes[e['i']]), np.array(model.nodes[e['j']])
            sp = (link or LINK_SECTION) if e['kind'] == 'link' else gs[e['group']]
            w = -sp['mass_per_len'] * np.linalg.norm(nj - ni) * G / 2.0
            frm.add_load(e['i'], fz=w); frm.add_load(e['j'], fz=w)
    xs = np.array(model.nodes)[:, 0]; mid = 0.5 * (xs.min() + xs.max())
    if live != 'none':
        for ni in model.load_idx:
            if live == 'full' or model.nodes[ni][0] < mid:
                frm.add_load(ni, fz=load)
    frm.solve(check_cond=False)
    U = np.array([frm.node_disp(i)[:3] for i in range(len(model.nodes))])
    util = np.full(len(model.elements), np.nan); Nax = np.full(len(model.elements), np.nan)
    sed = np.array([frm.element_strain_energy(ei) for ei in range(len(model.elements))])
    for ei, e in enumerate(model.elements):
        f = frm.element_end_forces_local(ei); Nax[ei] = f[0]
        if e['kind'] == 'link':
            continue
        sp = gs[e['group']]; r = sp['D'] / 2.0
        util[ei] = max((abs(f[k]) / sp['A'] + np.hypot(f[k + 4], f[k + 5]) * r / sp['Iy']) / FY for k in (0, 6))
    return dict(frm=frm, U=U, util=util, N=Nax, sed=sed, C=frm.compliance(), dmax=float(np.linalg.norm(U, axis=1).max()),
                dz=float(-U[:, 2].min()))


def draw(ax, model, kind, val=None, cmap=None, norm=None, lw=None, nodes=None, show_links=True, link_col='0.55',
         ghost=None, sup=True, elev=28, azim=-55, beam_col='black'):
    P = np.array(model.nodes, float) if nodes is None else nodes
    bi = [k for k, e in enumerate(model.elements) if e['kind'] == 'beam']
    li = [k for k, e in enumerate(model.elements) if e['kind'] == 'link']
    dim = slice(0, 2) if kind == 'plan' else slice(0, 3)
    seg = lambda ids, Q: [[Q[model.elements[k]['i'], dim], Q[model.elements[k]['j'], dim]] for k in ids]
    LC = LineCollection if kind == 'plan' else Line3DCollection
    add = ax.add_collection if kind == 'plan' else ax.add_collection3d
    if ghost is not None:
        add(LC(seg(bi, ghost), colors='0.85', linewidths=0.5))
    if show_links and li:
        add(LC(seg(li, P), colors=link_col, linewidths=0.6))
    if val is not None:
        cols = cmap(norm(np.asarray([val[k] for k in bi])))
    else:
        cols = beam_col
    widths = [lw[k] for k in bi] if lw is not None else 0.9
    add(LC(seg(bi, P), colors=cols, linewidths=widths, capstyle='round'))
    s = sorted(model.support_idx)
    if kind == 'plan':
        if sup:
            ax.scatter(P[s, 0], P[s, 1], s=9, c='black', marker='s', zorder=5, linewidths=0)
        lo, hi = P[:, :2].min(0), P[:, :2].max(0); pad = 0.03 * (hi - lo).max()
        ax.set_xlim(lo[0] - pad, hi[0] + pad); ax.set_ylim(lo[1] - pad, hi[1] + pad)
        ax.set_aspect('equal'); ax.set_axis_off()
    else:
        if sup:
            ax.scatter(P[s, 0], P[s, 1], P[s, 2], s=7, c='black', marker='s', linewidths=0, depthshade=False)
        Q = P if ghost is None else np.vstack([P, ghost])
        lo, hi = Q.min(0), Q.max(0)
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
        ax.set_box_aspect(tuple(np.maximum(hi - lo, 0.04 * (hi - lo).max())), zoom=1.3)
        ax.view_init(elev=elev, azim=azim); ax.set_axis_off()


def newax(case, figsize=(2.3, 2.3), three_d=False):
    kind, el, az = (VIEW3D if three_d else VIEW)[case]
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111) if kind == 'plan' else fig.add_subplot(111, projection='3d')
    if kind == '3d':
        ax.set_position([0, 0.12, 1, 0.88])
    return fig, ax, kind, el, az


def hcbar(fig, mappable, label, ticks=None, ticklabels=None, rect=(0.12, 0.08, 0.76, 0.035)):
    cax = fig.add_axes(list(rect))
    cb = fig.colorbar(mappable, cax=cax, orientation='horizontal', ticks=ticks)
    if ticklabels is not None:
        cb.ax.set_xticklabels(ticklabels)
    cb.set_label(label, fontsize=8); cb.ax.tick_params(labelsize=8, length=2)
    return cb


MID = [4] * 16  # D = 139.7 mm, the mid-catalogue reference


# ----------------------------------------------------------- design picker
def runs(rel, key):
    d = json.load(open(os.path.join(ROOT, rel)))
    m = d.get('methods', d)
    if key not in m:
        return None, d
    v = m[key]
    return (list(v.values()) if isinstance(v, dict) else list(v)), d


REV = {'cs2': 'results/benchmark/cs2.json', 'cs3': 'results/benchmark/cs3.json',
       'cs6': 'results/benchmark/cs6.json', 'dome': 'results/benchmark/dome.json',
       'vault': 'results/benchmark/vault.json'}


def best_design(case, key, matched=True):
    rr, d = runs(REV[case], key)
    mt = d['mass_target']; best = None
    for r in rr:
        C = np.asarray(r['compliance_history'], float); M = np.asarray(r['mass_history'], float)
        ok = np.abs(M - mt) <= 0.05 * mt if matched else np.ones_like(C, bool)
        if not ok.any():
            continue
        i = int(np.argmin(np.where(ok, C, np.inf)))
        if best is None or C[i] < best['C']:
            best = dict(C=float(C[i]), M=float(M[i]), idx=[int(x) for x in r['idx_history'][i]], seed=r.get('seed'), it=i)
    if best is None and matched:
        b = best_design(case, key, matched=False); b['matched'] = False; return b
    if best is not None:
        best['matched'] = matched
    return best


# ================================================================== groups
def fig_bearing():
    m, load = model_of('fan')
    P = np.array(m.nodes); links = [e for e in m.elements if e['kind'] == 'link']
    e_len = float(np.mean([np.linalg.norm(P[e['j']] - P[e['i']]) for e in links]))
    FD['fan_ecc_m'] = e_len
    # (a) 3D view of the fan with the vertical coordinate exaggerated
    ex = 8.0; Q = P.copy(); Q[:, 2] *= ex
    fig = plt.figure(figsize=(3.1, 2.5)); ax = fig.add_subplot(111, projection='3d'); ax.set_position([0, 0, 1, 1])
    for e in m.elements:
        a, b = Q[e['i']], Q[e['j']]
        if e['kind'] == 'link':
            ax.plot(*zip(a, b), color='0.45', lw=2.0)
        else:
            ax.plot(*zip(a, b), color='black', lw=1.2)
    s = sorted(m.support_idx); ax.scatter(Q[s, 0], Q[s, 1], Q[s, 2], s=12, marker='s', c='black', depthshade=False)
    lo, hi = Q.min(0), Q.max(0); ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect((hi - lo)[:2].tolist() + [0.35 * (hi - lo)[:2].max()], zoom=1.25); ax.view_init(elev=24, azim=-58); ax.set_axis_off()
    save(fig, 'fig_bearing_a')
    # (b) true-scale section through one bearing: lower beam along the view plane, upper beam cut normal to it
    D = CHS_CATALOG[4]["D_mm"] / 1e3; t = CHS_CATALOG[4].get('t_mm', 8.0) / 1e3
    fig, ax = plt.subplots(figsize=(3.1, 2.5))
    zl, zu = -e_len / 2, e_len / 2
    ax.plot([-0.45, 0.45], [zl, zl], color='black', lw=0.8, ls=(0, (6, 2, 1, 2)))
    for dz in (-D / 2, D / 2):
        ax.plot([-0.45, 0.45], [zl + dz, zl + dz], color='black', lw=1.1)
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(D / 2 * np.cos(th), zu + D / 2 * np.sin(th), color='black', lw=1.1)
    ax.plot((D / 2 - t) * np.cos(th), zu + (D / 2 - t) * np.sin(th), color='black', lw=0.6)
    ax.plot([0], [zu], marker='+', color='black', ms=7)
    ax.plot([0, 0], [zl, zu], color='0.45', lw=3.0, solid_capstyle='butt')
    ax.plot([0], [zl], 'o', color='black', ms=4); ax.plot([0], [zu], 'o', mfc='white', mec='black', ms=4)
    ax.annotate('', xy=(0.28, zu), xytext=(0.28, zl), arrowprops=dict(arrowstyle='<->', lw=0.7))
    ax.plot([0.08, 0.32], [zu, zu], color='0.5', lw=0.5); ax.plot([0.25, 0.32], [zl, zl], color='0.5', lw=0.5)
    ax.text(0.30, 0.0, ' e = %.0f mm' % (e_len * 1e3), va='center', fontsize=8)
    ax.text(-0.44, zl - D / 2 - 0.035, 'lower beam, D = %.1f mm' % (D * 1e3), fontsize=8, va='top')
    ax.text(-0.44, zu + D / 2 + 0.02, 'upper beam, section', fontsize=8, va='bottom')
    ax.text(-0.03, zl - 0.012, 'lower node', fontsize=8, color='0.25', ha='right', va='top')
    ax.text(-0.09, zu, 'upper node', fontsize=8, color='0.25', ha='right', va='center')
    ax.set_aspect('equal'); ax.set_xlim(-0.46, 0.62); ax.set_ylim(zl - D / 2 - 0.09, zu + D / 2 + 0.08)
    ax.set_xlabel('Distance along lower beam [m]'); ax.set_ylabel('z [m]')
    save(fig, 'fig_bearing_b')


def fig_groups():
    GL = [(0, 0, 0, 1), (0.45, 0.45, 0.45, 1), (0.7, 0.7, 0.7, 1), (0.25, 0.25, 0.25, 1), (0.85, 0.85, 0.85, 1)]
    for tag, case in CASES6:
        m, _ = model_of(case)
        fig, ax, kind, el, az = newax(case)
        g = {k: e.get('group', 0) or 0 for k, e in enumerate(m.elements)}
        draw(ax, m, kind, val=g, cmap=lambda x: np.array([GL[int(v) % 5] for v in np.asarray(x)]), norm=lambda v: v, elev=el, azim=az,
             lw={k: 1.3 for k in g}, link_col='0.75')
        FD.setdefault('n_groups', {})[case] = int(m.n_groups)
        save(fig, 'fig_groups_%s' % tag)


def fig_elev():
    specs = [('a', 'dome', 0, 2, 'x [m]'), ('b', 'vault', 1, 2, 'y [m]'), ('c', 'cs6', 0, 2, 'x [m]')]
    for tag, case, h, v, xl in specs:
        m, _ = model_of(case); P = np.array(m.nodes)
        fig, ax = plt.subplots(figsize=(6.4, 1.7) if case == 'cs6' else (3.1, 1.8))
        for e in m.elements:
            a, b = P[e['i']], P[e['j']]
            ax.plot([a[h], b[h]], [a[v], b[v]], color='0.6' if e['kind'] == 'link' else 'black',
                    lw=0.5 if e['kind'] == 'link' else 0.6, alpha=0.8)
        s = sorted(m.support_idx); ax.scatter(P[s, h], P[s, v], s=10, marker='s', c='black', zorder=4)
        ax.set_aspect('equal'); ax.set_xlabel(xl); ax.set_ylabel('z [m]')
        span = P[:, h].max() - P[:, h].min(); rise = P[:, v].max() - P[s, v].min()
        FD.setdefault('elev', {})[case] = dict(extent=float(span), rise=float(rise))
        save(fig, 'fig_elev_%s' % tag)


def fig_lam():
    lams = [0.15, 0.25, 0.40]
    for tag, lam in zip('abc', lams):
        m = square_nexorade(6, 6, 2.5, lam)
        fig, ax = plt.subplots(figsize=(2.1, 2.1)); draw(ax, m, 'plan', link_col='0.6')
        save(fig, 'fig_lam_%s' % tag)
    # compliance against lambda, uniform mid-catalogue section, nominal load
    from reciprocal_geom_v2 import rf_dome2, rf_vault_hex2
    grid = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
    builders = {
        'Fan unit': lambda l: (fan_unit(4, R=3.0, lam=l), -2200.0),
        'CS2 nexorade': lambda l: (square_nexorade(6, 6, 2.5, l), -450.0),
        'CS3 hexagonal': lambda l: (hex_nexorade(3, 2.2, l), -1400.0),
        'Dome': lambda l: (lambda mm: (mm, mm.meta['load_per_node']))(rf_dome2(n_sect=7, n_ring=4, span=18.5, rise=3.15, lam=l, max_group_rings=4)),
        'Vault': lambda l: (lambda mm: (mm, mm.meta['load_per_node']))(rf_vault_hex2(lam_bands=(l,) * 6)),
        'Bridge': lambda l: (lambda mm: (mm, mm.meta['load_per_node']))(davinci_bridge2(n_bays=24, span=20.0, width=6.0, lam=l)),
    }
    res = {}
    for name, b in builders.items():
        cs = []
        for l in grid:
            try:
                m, load = b(l); a = analyse(m, [4] * m.n_groups, load); cs.append(a['C'])
            except Exception as ex:
                print('lam fail', name, l, ex); cs.append(np.nan)
        res[name] = cs
    FD['lam_sweep'] = dict(grid=grid, C=res)
    fig, ax = plt.subplots(figsize=(3.1, 2.5))
    sty = [('-', 'o'), ('--', 's'), (':', '^'), ('-.', 'v'), ('-', 'D'), ('--', 'x')]
    for (name, cs), (ls, mk) in zip(res.items(), sty):
        cs = np.array(cs); i0 = grid.index(0.25)
        ax.plot(grid, cs / cs[i0], ls=ls, marker=mk, ms=3.5, color='black', lw=0.9, mfc='white', label=name)
    ax.set_yscale('log'); ax.set_xlabel(r'Engagement ratio $\lambda$'); ax.set_ylabel(r'$C(\lambda)\,/\,C(0.25)$')
    leg = ax.legend(frameon=False, ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.25))
    save(fig, 'fig_lam_d', [leg])


CASES6 = [('a', 'fan'), ('b', 'cs2'), ('c', 'cs3'), ('d', 'dome_ref'), ('e', 'vault_ref'), ('f', 'cs6')]


def fig_def():
    out = {}
    for tag, case in CASES6:
        m, load = model_of(case); a = analyse(m, MID, load)
        P = np.array(m.nodes); ext = (P.max(0) - P.min(0)).max()
        sc = 0.08 * ext / max(a['dmax'], 1e-12)
        un = np.linalg.norm(a['U'], axis=1) * 1e3
        ev = {k: 0.5 * (un[e['i']] + un[e['j']]) for k, e in enumerate(m.elements)}
        fig, ax, kind, el, az = newax(case, (2.3, 2.3), three_d=True)
        ax.set_position([0, 0.16, 1, 0.84])
        norm = Normalize(0, un.max()); cmap = GREY
        draw(ax, m, kind, val=ev, cmap=cmap, norm=norm, nodes=P + sc * a['U'], ghost=P, show_links=False,
             elev=el, azim=az, lw={k: 1.1 for k in ev})
        hcbar(fig, plt.cm.ScalarMappable(norm=norm, cmap=cmap), '|u| [mm], shape x%.0f' % sc, rect=(0.12, 0.1, 0.76, 0.035))
        out[case] = dict(C=a['C'], dmax_mm=a['dmax'] * 1e3, scale=sc)
        save(fig, 'fig_def_%s' % tag)
    FD['deformed_mid'] = out


def fig_sed():
    """Strain energy per unit member length (normalised): the field that drives the F1 linearisation."""
    out = {}
    for tag, case in CASES6:
        m, load = model_of(case); a = analyse(m, MID, load); P = np.array(m.nodes)
        L = np.array([np.linalg.norm(P[e['j']] - P[e['i']]) for e in m.elements])
        bi = [k for k, e in enumerate(m.elements) if e['kind'] == 'beam']
        dens = a['sed'] / np.maximum(L, 1e-9); ref = dens[bi].max()
        val = {k: max(dens[k] / ref, 1e-4) for k in bi}
        fig, ax, kind, el, az = newax(case, (2.1, 2.1))
        norm = LogNorm(1e-3, 1.0); cmap = GREY
        draw(ax, m, kind, val=val, cmap=cmap, norm=norm, elev=el, azim=az, lw={k: 1.3 for k in val}, link_col='0.7')
        share = a['sed'] / a['sed'].sum(); li = [k for k, e in enumerate(m.elements) if e['kind'] == 'link']
        srt = np.sort(share[bi])[::-1]
        out[case] = dict(link_share=float(share[li].sum()), top10pct_beam_share=float(srt[:max(1, len(bi) // 10)].sum()))
        save(fig, 'fig_sed_%s' % tag)
    fig = plt.figure(figsize=(6.4, 0.55))
    hcbar(fig, plt.cm.ScalarMappable(norm=LogNorm(1e-3, 1.0), cmap=GREY),
          'strain energy per unit length, normalised by the maximum beam element', rect=(0.2, 0.55, 0.6, 0.3))
    fig.savefig(os.path.join(OUT, 'fig_sed_cbar.pdf'), bbox_inches='tight', pad_inches=0.02); plt.close(fig)
    FD['sed_mid'] = out


def fig_link():
    """Connector axial force on the matched-mass block-annealer designs; tension fraction bar chart."""
    out = {}
    for tag, case in [('a', 'cs2'), ('b', 'dome'), ('c', 'cs6')]:
        m, load = model_of(case); b = best_design(case, 'sa_block_w1000f'); a = analyse(m, b['idx'], load)
        li = [k for k, e in enumerate(m.elements) if e['kind'] == 'link']
        P = np.array(m.nodes)
        # local axis i->j of a link points from lower to upper node; N>0 at end i = compression
        Nk = np.array([a['N'][k] for k in li]) / 1e3
        fig, ax, kind, el, az = newax(case, (2.3, 2.55) if case != 'cs6' else (6.4, 2.6))
        if kind == 'plan':
            ax.set_position([0, 0.2, 1, 0.8])
        draw(ax, m, kind, show_links=False, beam_col='0.75', elev=el, azim=az, lw={k: 0.6 for k in range(len(m.elements))})
        lim = np.percentile(np.abs(Nk), 98); sz = 3 + 30 * np.minimum(np.abs(Nk) / lim, 1)
        Q = np.array([0.5 * (P[m.elements[k]['i']] + P[m.elements[k]['j']]) for k in li]); cmp_ = Nk > 0
        for sel, fc, lab in ((cmp_, 'black', 'compression'), (~cmp_, 'white', 'tension')):
            if kind == 'plan':
                ax.scatter(Q[sel, 0], Q[sel, 1], s=sz[sel], facecolors=fc, edgecolors='black', linewidths=0.5, zorder=6, label=lab)
            else:
                ax.scatter(Q[sel, 0], Q[sel, 1], Q[sel, 2], s=sz[sel], facecolors=fc, edgecolors='black', linewidths=0.5, depthshade=False, label=lab)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=2, frameon=False, fontsize=8, markerscale=1.0)
        out[case] = dict(tension_frac=float(np.mean(Nk < 0)), seed=b['seed'], Nmax_kN=float(np.abs(Nk).max()))
        save(fig, 'fig_link_%s' % tag)
    # tension fraction on all five matched-mass designs
    fr = {}
    for case in ['cs2', 'cs3', 'cs6', 'dome', 'vault']:
        m, load = model_of(case); b = best_design(case, 'sa_block_w1000f'); a = analyse(m, b['idx'], load)
        li = [k for k, e in enumerate(m.elements) if e['kind'] == 'link']
        fr[case] = float(np.mean(a['N'][li] < 0))
    out['tension_fraction_matched'] = fr
    fig, ax = plt.subplots(figsize=(2.3, 2.2))
    keys = ['cs2', 'cs3', 'cs6', 'dome', 'vault']
    ax.bar(range(5), [100 * fr[k] for k in keys], color='0.55', edgecolor='black', width=0.6, lw=0.6)
    ax.set_xticks(range(5)); ax.set_xticklabels([CASE_LABEL[k] for k in keys])
    ax.set_ylabel('Connectors in tension [%]'); ax.set_ylim(0, 60)
    save(fig, 'fig_link_d')
    FD['links'] = out


ARMS_MAP = [('sa_block_w1000f', 'Block annealer'), ('greedy_w1000f', 'Greedy decode'), ('sa_single_w1000f', 'Single-flip annealer')]


def size_panel(case, key, name, figsize):
    m, load = model_of(case); b = best_design(case, key)
    a = analyse(m, b['idx'], load)
    val = {k: b['idx'][e['group']] for k, e in enumerate(m.elements) if e['kind'] == 'beam'}
    lw = {k: 0.5 + 2.4 * D_MM[v] / 244.5 for k, v in val.items()}
    fig, ax, kind, el, az = newax(case, figsize)
    if kind == '3d':
        ax.set_position([0, 0, 1, 1])
    draw(ax, m, kind, val=val, cmap=SIZE_CMAP, norm=SIZE_NORM, lw=lw, elev=el, azim=az, link_col='0.7')
    FD.setdefault('size_maps', {}).setdefault(case, {})[key] = dict(C=b['C'], M=b['M'], seed=b['seed'], it=b['it'],
                                                                     matched=b['matched'], idx=b['idx'], C_check=a['C'])
    save(fig, name)


def fig_size():
    for case in ['cs2', 'cs3', 'dome', 'vault', 'cs6']:
        for j, (key, _) in enumerate(ARMS_MAP):
            fs = (6.4, 2.05) if case == 'cs6' else (2.1, 2.1)
            size_panel(case, key, 'fig_size_%s_%s' % (case, 'abc'[j]), fs)
    fig = plt.figure(figsize=(6.4, 0.55))
    hcbar(fig, plt.cm.ScalarMappable(norm=SIZE_NORM, cmap=SIZE_CMAP), 'CHS outer diameter D [mm] (EN 10210)',
          ticks=range(8), ticklabels=['%.1f' % d for d in D_MM], rect=(0.08, 0.55, 0.84, 0.3))
    fig.savefig(os.path.join(OUT, 'fig_size_cbar.pdf'), bbox_inches='tight', pad_inches=0.02); plt.close(fig)


def fig_sechist():
    keys = ['cs2', 'cs3', 'cs6', 'dome', 'vault']
    out = {}
    for tag, case in zip('abcde', keys):
        m, _ = model_of(case)
        fig, ax = plt.subplots(figsize=(2.1, 1.9))
        for j, (key, lab) in enumerate(ARMS_MAP):
            b = best_design(case, key)
            ms = np.zeros(8)
            for e in m.elements:
                if e['kind'] != 'beam':
                    continue
                L = np.linalg.norm(np.subtract(m.nodes[e['j']], m.nodes[e['i']]))
                c = b['idx'][e['group']]; ms[c] += CHS_CATALOG[c]['mass_per_len'] * L
            ms = 100 * ms / ms.sum()
            ax.bar(np.arange(8) + (j - 1) * 0.27, ms, width=0.27, color=['0.15', '0.55', 'white'][j],
                   edgecolor='black', lw=0.5, hatch=['', '', '////'][j], label=lab)
            out.setdefault(case, {})[key] = [float(x) for x in ms]
        ax.set_xticks(range(8)); ax.set_xticklabels([str(i + 1) for i in range(8)])
        ax.set_xlabel('Catalogue section'); ax.set_ylabel('Member mass [%]')
        extra = []
        if tag == 'a':
            extra = [fig.legend(*ax.get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(0.5, -0.3), bbox_transform=ax.transAxes, ncol=2, frameon=False)]
        save(fig, 'fig_sechist_%s' % tag, extra)
    FD['section_mass_share'] = out


MC_ARMS = [('sa_block_w5', 'Block annealer, $w_m$=5', 'o', '0.0', False), ('sa_block_w1000f', 'Block annealer, $w_m$=1000', 'o', 'white', True),
           ('greedy_w1000f', 'Greedy decode, $w_m$=1000', 's', 'white', True), ('sa_single_w1000f', 'Single-flip, $w_m$=1000', '^', 'white', True),
           ('ls_w5', 'FE local search', 'D', '0.6', False), ('ga8_w5', 'GA, population 8', 'v', '0.6', False)]


def fig_mc():
    out = {}
    for tag, case in zip('abcde', ['cs2', 'cs3', 'cs6', 'dome', 'vault']):
        fig, ax = plt.subplots(figsize=(3.1, 2.3))
        mt = json.load(open(os.path.join(ROOT, REV[case])))['mass_target']
        ax.axvspan(0.95, 1.05, color='0.9', lw=0)
        for key, lab, mk, fc, matched in MC_ARMS:
            rr, _ = runs(REV[case], key)
            if not rr:
                continue
            pts = []
            for r in rr:
                C = np.asarray(r['compliance_history']); M = np.asarray(r['mass_history']) / mt
                ok = np.abs(M - 1) <= 0.05 if matched else np.ones_like(C, bool)
                if not ok.any():
                    ok = np.ones_like(C, bool)
                i = int(np.argmin(np.where(ok, C, np.inf))); pts.append((M[i], C[i]))
            pts = np.array(pts)
            ax.scatter(pts[:, 0], pts[:, 1], marker=mk, s=16, facecolors=fc, edgecolors='black', linewidths=0.6, label=lab, zorder=3)
            out.setdefault(case, {})[key] = dict(mass_ratio_mean=float(pts[:, 0].mean()), C_mean=float(pts[:, 1].mean()))
        ax.set_yscale('log'); ax.set_xlabel('Mass / target'); ax.set_ylabel('Compliance [J]')
        extra = []
        if tag == 'a':
            extra = [fig.legend(*ax.get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(0.5, -0.24), bbox_transform=ax.transAxes, ncol=2, frameon=False)]
        save(fig, 'fig_mc_%s' % tag, extra)
    FD['mass_compliance'] = out


def scaled_link(f):
    return dict(A=LINK_SECTION['A'] * f, Iy=LINK_SECTION['Iy'] * f, Iz=LINK_SECTION['Iz'] * f, J=LINK_SECTION['J'] * f,
                mass_per_len=LINK_SECTION['mass_per_len'])


def fig_bridge():
    m, load = model_of('cs6'); P = np.array(m.nodes); span = 20.0; lim = span / 250 * 1e3
    b = best_design('cs6', 'sa_block_w1000f'); n_g = m.n_groups
    half = analyse(m, b['idx'], load, sw=True, live='half'); full = analyse(m, b['idx'], load, sw=True, live='full')
    # (a) deformed elevation under SW + half-span live load
    un = np.linalg.norm(half['U'], axis=1) * 1e3; sc = 0.04 * span / half['dmax']
    D2 = P + sc * half['U']
    fig, ax = plt.subplots(figsize=(6.4, 2.0))
    norm = Normalize(0, un.max()); cmap = GREY
    bi = [k for k, e in enumerate(m.elements) if e['kind'] == 'beam']
    ax.add_collection(LineCollection([[P[m.elements[k]['i'], [0, 2]], P[m.elements[k]['j'], [0, 2]]] for k in bi], colors='0.8', linewidths=0.6))
    ax.add_collection(LineCollection([[D2[m.elements[k]['i'], [0, 2]], D2[m.elements[k]['j'], [0, 2]]] for k in bi],
                                     colors=cmap(norm([0.5 * (un[m.elements[k]['i']] + un[m.elements[k]['j']]) for k in bi])), linewidths=1.2))
    s_ = sorted(m.support_idx); ax.scatter(P[s_, 0], P[s_, 2], s=12, marker='s', c='black', zorder=4)
    ax.set_xlim(-0.5, 20.5); ax.set_ylim(min(P[:, 2].min(), D2[:, 2].min()) - 0.2, max(P[:, 2].max(), D2[:, 2].max()) + 0.2)
    ax.set_aspect('equal'); ax.set_xlabel('x [m]'); ax.set_ylabel('z [m]')
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation='vertical', fraction=0.03, pad=0.02)
    cb.set_label('|u| [mm]', fontsize=8); cb.ax.tick_params(labelsize=8)
    ax.text(0.2, ax.get_ylim()[1] - 0.15, 'shape x%.0f' % sc, fontsize=8, va='top')
    save(fig, 'fig_bridge_a')
    # (b) utilisation map in plan
    fig, ax = plt.subplots(figsize=(6.4, 2.3))
    umax = float(np.nanmax(half['util'])); norm = Normalize(0, max(1.0, umax)); cmap = GREY
    uv = {k: half['util'][k] for k, e in enumerate(m.elements) if e['kind'] == 'beam'}
    draw(ax, m, 'plan', val=uv, cmap=cmap, norm=norm, lw={k: 1.5 for k in uv}, link_col='0.75')
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation='vertical', fraction=0.03, pad=0.02)
    cb.set_label('utilisation', fontsize=8); cb.ax.tick_params(labelsize=8)
    save(fig, 'fig_bridge_b')
    # (c) deflection profiles along the centre rail
    yc = 0.5 * (P[:, 1].max() + P[:, 1].min())
    rail = [i for i in range(len(P)) if abs(P[i, 1] - yc) < 0.2]
    rail = sorted(set(rail), key=lambda i: P[i, 0])
    cases = [('Block design, SW + full live', b['idx'], 'full', None, '-', None),
             ('Block design, SW + half live', b['idx'], 'half', None, '--', None),
             ('All sections D = 244.5 mm, SW + half', [7] * n_g, 'half', None, ':', None),
             ('Block design, links x10, SW + half', b['idx'], 'half', scaled_link(10), '-.', None)]
    fig, ax = plt.subplots(figsize=(3.1, 2.5)); prof = {}
    for lab, idx, live, lk, ls, _ in cases:
        a = analyse(m, idx, load, sw=True, live=live, link=lk)
        x = P[rail, 0]; uz = a['U'][rail, 2] * 1e3
        ax.plot(x, uz, ls=ls, color='black', lw=1.0, label=lab)
        prof[lab] = dict(rail_min_mm=float(uz.min()), rail_min_x=float(x[uz.argmin()]), rail_max_mm=float(uz.max()), rail_max_x=float(x[uz.argmax()]), dz_mm=a['dz'] * 1e3, ratio=a['dz'] * 1e3 / lim, dmax_ratio=a['dmax'] * 1e3 / lim,
                         maxutil=float(np.nanmax(a['util'])), mass=float(model_mass(m, gs_of(m, idx), lk)))
    ax.axhline(-lim, color='0.5', lw=0.8); ax.text(20.0, -lim - 8, 'span/250', fontsize=8, color='0.35', ha='right', va='top')
    ax.axhline(0, color='0.7', lw=0.5)
    ax.set_xlabel('x [m]'); ax.set_ylabel('Vertical displacement, centre rail [mm]')
    leg = ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.24), ncol=1)
    save(fig, 'fig_bridge_c', [leg])
    # (d) peak displacement / (span/250) versus link stiffness multiplier
    mult = [0.1, 0.3, 1, 3, 10, 30, 100]
    fig, ax = plt.subplots(figsize=(3.1, 2.5)); sweep = {}
    for lab, idx, ls, mk in [('Block design', b['idx'], '-', 'o'), ('All sections D = 244.5 mm', [7] * n_g, '--', 's')]:
        r = [analyse(m, idx, load, sw=True, live='half', link=scaled_link(f))['dmax'] * 1e3 / lim for f in mult]
        ax.plot(mult, r, ls=ls, marker=mk, color='black', mfc='white', ms=4, lw=1.0, label=lab); sweep[lab] = r
    ax.axhline(1, color='0.5', lw=0.8); ax.set_xscale('log'); ax.set_xlabel('Connector stiffness multiplier')
    ax.set_ylabel('Peak |u| / (span/250)'); leg = ax.legend(frameon=False)
    save(fig, 'fig_bridge_d', [leg])
    FD['bridge'] = dict(design=b, profiles=prof, link_mult=mult, sweep=sweep, mass_all_max=float(model_mass(m, gs_of(m, [7] * n_g))))


def fig_cs1():
    d = json.load(open(os.path.join(ROOT, 'results/benchmark/cs1_block.json')))
    t = json.load(open(os.path.join(ROOT, 'results/cs1_fans_456.json')))
    from math import sqrt

    def wilson(k, n, z=1.96):
        p = k / n; den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den; h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
        return c - h, c + h
    fig, ax = plt.subplots(figsize=(3.1, 2.3))
    for j, (key, lab, col) in enumerate([('single_success', 'Single-flip annealer', '0.6'), ('block_success', 'Block annealer', '0.15')]):
        ks = [d[f][key] for f in ('fan4', 'fan5', 'fan6')]
        lo_hi = [wilson(k, 10) for k in ks]
        x = np.arange(3) + (j - 0.5) * 0.36
        ax.bar(x, ks, width=0.36, color=col, edgecolor='black', lw=0.6, label=lab)
        ax.errorbar(x, ks, yerr=[[k - 10 * l for k, (l, h) in zip(ks, lo_hi)], [10 * h - k for k, (l, h) in zip(ks, lo_hi)]],
                    fmt='none', ecolor='black', capsize=2, lw=0.7)
    ax.set_xticks(range(3)); ax.set_xticklabels(['4-beam', '5-beam', '6-beam']); ax.set_ylim(0, 11)
    ax.set_ylabel('Seeds reaching the optimum (of 10)')
    leg = ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    save(fig, 'fig_cs1_a', [leg])
    fig, ax = plt.subplots(figsize=(3.1, 2.3))
    for f, ls in zip(('fan4', 'fan5', 'fan6'), ('-', '--', ':')):
        C = np.sort(np.asarray(t[f]['compliance_table'], float).ravel())
        ax.plot(np.arange(1, len(C) + 1), C, ls=ls, color='black', lw=1.0, label=f.replace('fan', '') + '-beam fan')
    ax.set_yscale('log'); ax.set_xlabel('Design rank (of 64)'); ax.set_ylabel('Compliance [J]'); leg = ax.legend(frameon=False)
    save(fig, 'fig_cs1_b', [leg])
    FD['cs1'] = {f: dict(min=float(np.min(t[f]['compliance_table'])), max=float(np.max(t[f]['compliance_table']))) for f in ('fan4', 'fan5', 'fan6')}


def fig_pen():
    p = json.load(open(os.path.join(ROOT, 'results/penalty_sensitivity_v2.json')))
    W = p['w_mass_levels']
    me = [np.mean([abs(r['mass_error_pct']) for r in p['runs'] if r['w_mass'] == w]) for w in W]
    cc = [np.mean([r['final_compliance'] for r in p['runs'] if r['w_mass'] == w]) for w in W]
    fig, ax = plt.subplots(figsize=(3.1, 2.3)); ax2 = ax.twinx(); ax2.spines['right'].set_visible(True)
    ax.plot(W, me, '-o', color='black', mfc='white', ms=4, lw=1, label='Mass error')
    ax2.plot(W, cc, '--s', color='0.4', mfc='0.4', ms=3.5, lw=1, label='Compliance')
    ax.set_xscale('log'); ax.set_xlabel('$w_m$'); ax.set_ylabel('Mean |mass error| [%]'); ax2.set_ylabel('Final compliance [J]')
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    leg = ax.legend(h1 + h2, l1 + l2, frameon=False, loc='upper center')
    save(fig, 'fig_pen_a', [leg])
    # block annealer: matched-mass trade-off over w_m on the five geometries (stiffness-led best)
    fig, ax = plt.subplots(figsize=(3.1, 2.3)); out = {}
    for case, mk in zip(['cs2', 'cs3', 'cs6', 'dome', 'vault'], 'os^Dv'):
        xs, ys = [], []
        d = json.load(open(os.path.join(ROOT, REV[case]))); mt = d['mass_target']
        for key, w in [('sa_block_w5', 5), ('sa_block_w20', 20), ('sa_block_w100', 100), ('sa_block_w1000f', 1000)]:
            rr = d['methods'].get(key)
            if not rr:
                continue
            em = []
            for r in rr:
                C = np.asarray(r['compliance_history']); i = int(np.argmin(C)); em.append(100 * (r['mass_history'][i] - mt) / mt)
            xs.append(w); ys.append(np.mean(em))
        ax.plot(xs, ys, '-', marker=mk, color='black', mfc='white', ms=4, lw=0.9, label=CASE_LABEL[case]); out[case] = dict(w=xs, mass_err=ys)
    ax.axhspan(-5, 5, color='0.9', lw=0); ax.axhline(0, color='0.6', lw=0.5)
    ax.set_xscale('log'); ax.set_xlabel('$w_m$'); ax.set_ylabel('Mass error of best design [%]'); leg = ax.legend(frameon=False, ncol=2)
    save(fig, 'fig_pen_b', [leg])
    FD['penalty'] = dict(single_flip=dict(w=W, mass_err=me, C=cc), block=out)


def fig_util():
    out = {}
    for tag, case in zip('abcde', ['cs2', 'cs3', 'dome', 'vault', 'cs6']):
        m, load = model_of(case); b = best_design(case, 'sa_block_w1000f'); a = analyse(m, b['idx'], load, sw=True, live='full')
        uv = {k: a['util'][k] for k, e in enumerate(m.elements) if e['kind'] == 'beam'}
        fig, ax, kind, el, az = newax(case, (6.4, 2.05) if case == 'cs6' else (2.1, 2.1))
        if kind == 'plan':
            ax.set_position([0, 0.2, 1, 0.8])
        norm = Normalize(0, 0.6); cmap = GREY
        draw(ax, m, kind, val=uv, cmap=cmap, norm=norm, elev=el, azim=az, lw={k: 1.3 for k in uv}, link_col='0.75')

        out[case] = dict(max_util=float(np.nanmax(a['util'])), mean_util=float(np.nanmean(a['util'])), dz_mm=a['dz'] * 1e3)
        save(fig, 'fig_util_%s' % tag)
    fig = plt.figure(figsize=(6.4, 0.55))
    hcbar(fig, plt.cm.ScalarMappable(norm=Normalize(0, 0.6), cmap=GREY),
          'elastic utilisation under self-weight + full live load', rect=(0.2, 0.55, 0.6, 0.3))
    fig.savefig(os.path.join(OUT, 'fig_util_cbar.pdf'), bbox_inches='tight', pad_inches=0.02); plt.close(fig)
    FD['util_full'] = out


ALL = dict(bearing=fig_bearing, groups=fig_groups, elev=fig_elev, lam=fig_lam, deform=fig_def, sed=fig_sed, link=fig_link,
           size=fig_size, sechist=fig_sechist, mc=fig_mc, bridge=fig_bridge, cs1=fig_cs1, pen=fig_pen, util=fig_util)

if __name__ == '__main__':
    todo = sys.argv[1:] or list(ALL)
    for t in todo:
        ALL[t]()
        json.dump(FD, open(FD_PATH, 'w'), indent=1, default=float)

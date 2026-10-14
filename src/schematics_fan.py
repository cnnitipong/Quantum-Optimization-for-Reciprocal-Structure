import os
import os as _os_bootstrap
ROOT = _os_bootstrap.path.dirname(_os_bootstrap.path.dirname(_os_bootstrap.path.abspath(__file__)))
"""
schematics_fan.py -- publication figures (PDF+PNG, 300dpi, serif fonts) built
from the real results/*.json produced by the case-study scripts, and (for
fig_fan_geometry / fig_encoding_qubo / fig_connection_detail) from the real
geometry/QUBO generator code (reciprocal_geom.py, sizing.py, qubo.py,
sections.py). Sizes target 90mm (single-column) / 190mm (double-column)
widths typical of the Structures (Elsevier) journal template.

REVISION (author feedback) policy:
  - fig_typologies (Fig. 1) and fig_cs2_design (Fig. 5) stay in COLOR, but
    the eccentric beam-to-beam "link" elements are drawn as a single small
    filled dot at the connection midpoint instead of as a separate line
    segment, and the z-axis exaggeration used for the flat typologies is
    greatly reduced (fixed, modest box_aspect) so the mutually-supporting
    beams read as *crossing* near the dots rather than as tall zig-zag
    spikes.
  - All other matplotlib figures (fig_cs1_validation, fig_cs2_convergence,
    fig_cs3_convergence, fig_cs4_fmqa, fig_cs5_pareto, fig_sensitivity,
    fig_ablation) are converted to black/gray, distinguished by linestyle
    (solid/dashed/dotted/dash-dot), marker shape (o/s/^/x), gray-shaded
    +/-std bands, and hatching for bar charts, so they remain unambiguous
    in pure grayscale print.
  - Three new explanatory figures are added (fig_fan_geometry,
    fig_encoding_qubo, fig_connection_detail), all black/gray, built from
    the real generator / sizing / QUBO code (no synthetic placeholder
    data).
"""
import json
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D  # noqa

RESULTS = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
CODE = os.path.join(ROOT, "src")
sys.path.insert(0, CODE)

MM = 1 / 25.4
COLORS = {  # colorblind-safe (Okabe-Ito) -- used ONLY by the two figures
    # that keep color per author instruction (fig_typologies, fig_cs2_design)
    'sa': '#0072B2', 'ga': '#E69F00', 'random': '#009E73',
    'fmqa': '#0072B2', 'nsga': '#E69F00', 'black': '#000000',
    'accent': '#D55E00', 'grey': '#999999',
}

# Grayscale line styles shared by every B/W figure: (color, linestyle, marker)
_BW_SERIES = [
    dict(color='#000000', ls='-', marker='o'),
    dict(color='#555555', ls='--', marker='s'),
    dict(color='#999999', ls=':', marker='^'),
    dict(color='#777777', ls='-.', marker='x'),
]

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 9,
    'legend.fontsize': 7,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'axes.linewidth': 0.6,
    'lines.linewidth': 1.2,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})


def savefig(fig, name):
    fig.savefig(f"{FIG}/{name}.pdf", bbox_inches='tight')
    fig.savefig(f"{FIG}/{name}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("saved", name)


def _markevery(n, k=9):
    return max(1, n // k)


def _link_midpoints(model, nodes):
    """(N,3) array of beam-to-beam connection points: midpoint of every
    'link' element (the short eccentric-bearing connector at each
    engagement point)."""
    mids = [(nodes[e['i']] + nodes[e['j']]) / 2.0
            for e in model.elements if e['kind'] == 'link']
    return np.array(mids) if mids else np.zeros((0, 3))


# --------------------------------------------------------------------------
# fig_typologies (Fig. 1, COLOR) -- beam-to-beam connections drawn as dots,
# not long orange lines; z-axis exaggeration reduced to a modest fixed
# value so beams read as crossing near the dots.
# --------------------------------------------------------------------------
def fig_typologies():
    from reciprocal_geom import (fan_unit, square_grillage, hex_grillage, dome,
                                  physical_beams, true_node_positions)

    fig = plt.figure(figsize=(190 * MM, 175 * MM))
    gs = fig.add_gridspec(2, 6)
    # (model, title, elev, azim, subplot-cell, z box-aspect)
    specs = [
        (fan_unit(3, R=3.0, lam=0.25), "(a) 3-beam fan unit", 22, -60, gs[0, 0:2], 0.16),
        (fan_unit(4, R=3.0, lam=0.25), "(b) 4-beam fan unit", 22, -60, gs[0, 2:4], 0.16),
        (square_grillage(4, 4, cell=3.0, lam=0.25), "(c) Rotated-square grillage", 32, -60, gs[0, 4:6], 0.16),
        (hex_grillage(2, edge=3.0, lam=0.25), "(d) Hexagonal grillage", 32, -60, gs[1, 1:3], 0.16),
        (dome(2, span=16.0, rise=3.0, lam_per_ring=[0.25, 0.25, 0.25]), "(e) Reciprocal dome", 22, -60, gs[1, 3:5], 0.32),
    ]
    for model, title, elev, azim, cell, zaspect in specs:
        ax = fig.add_subplot(cell, projection='3d')
        nodes = np.array(model.nodes)
        tpos = true_node_positions(model)
        # each PHYSICAL beam drawn as ONE straight rod (true, non-eccentric
        # support->tip axis) -- not the FE support->interior->tip polyline,
        # which kinks at the interior node's +-ecc/2 offset.
        for bm in physical_beams(model):
            p1, p2 = tpos[bm['support_idx']], tpos[bm['tip_idx']]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                    color=COLORS['sa'], lw=1.4, zorder=3)
        mids = _link_midpoints(model, nodes)
        if len(mids):
            ax.scatter(mids[:, 0], mids[:, 1], mids[:, 2], color=COLORS['accent'],
                       marker='o', s=12, zorder=6, depthshade=False, linewidth=0)
        sup = nodes[list(model.support_idx)]
        ax.scatter(sup[:, 0], sup[:, 1], sup[:, 2], color='k', marker='^',
                   s=18, zorder=10, depthshade=False)
        ax.set_title(title, pad=-4)
        ax.set_box_aspect([1, 1, zaspect])
        ax.view_init(elev=elev, azim=azim)
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
        ax.grid(False)
        for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
            pane.pane.set_alpha(0.0)
    fig.suptitle("Reciprocal-frame typologies (blue: beam members; dots: beam-to-beam "
                  "connection points; triangles: pinned supports)\n"
                  "(vertical axis compressed for clarity)",
                  y=0.03, fontsize=7)
    fig.subplots_adjust(wspace=0.15, hspace=0.15)
    savefig(fig, "fig_typologies")


# --------------------------------------------------------------------------
# fig_cs1_validation -- GRAYSCALE
# --------------------------------------------------------------------------
def fig_cs1_validation():
    d = json.load(open(f"{RESULTS}/cs1_validation.json"))
    fig, axes = plt.subplots(2, 2, figsize=(140 * MM, 100 * MM))

    for col, key in enumerate(("fan3", "fan4")):
        r = d[key]
        # success prob bar (hatched, gray fill)
        ax = axes[0, col]
        ax.bar([0], [r['sa_success_probability']], color='0.78', edgecolor='k',
               hatch='///', width=0.5, linewidth=0.9)
        ax.set_ylim(0, 1.05)
        ax.set_xticks([0])
        ax.set_xticklabels([f"{'3' if col==0 else '4'}-beam fan"])
        ax.set_ylabel("SA-QUBO success prob.")
        ax.set_title(f"{'3-beam' if col==0 else '4-beam'} fan (10 seeds x 100 reads)")
        ax.text(0, r['sa_success_probability'] + 0.03,
                f"{r['sa_success_probability']*100:.0f}%", ha='center', fontsize=7)

        # sorted energy landscape with SA hits
        ax2 = axes[1, col]
        T = np.array(r['compliance_table'])
        energies = np.sort(T.flatten())
        ax2.plot(energies, color='0.45', lw=1.1, ls='-', label='all 64 combos (sorted)')
        opt = r['global_opt_compliance']
        ax2.axhline(opt, color='k', lw=1.0, ls='--', label='global optimum')
        ax2.set_xlabel("rank (sorted compliance)")
        ax2.set_ylabel("compliance [J]")
        ax2.set_yscale('log')
        ax2.legend(loc='upper left', fontsize=6)

    fig.tight_layout()
    savefig(fig, "fig_cs1_validation")


# --------------------------------------------------------------------------
# fig_cs2_convergence / fig_cs3_convergence -- GRAYSCALE
# --------------------------------------------------------------------------
def _mean_std_bands(histories):
    n = min(len(h) for h in histories)
    arr = np.array([h[:n] for h in histories])
    return arr.mean(axis=0), arr.std(axis=0), n


def fig_convergence(json_name, title, outname, ylabel="compliance"):
    d = json.load(open(f"{RESULTS}/{json_name}"))
    fig, ax = plt.subplots(figsize=(90 * MM, 70 * MM))
    labels = {'sa_qubo': 'SA-QUBO (F1)', 'ga': 'GA', 'random': 'Random search'}
    for i, method in enumerate(('sa_qubo', 'ga', 'random')):
        st = _BW_SERIES[i]
        runs = d['methods'][method]
        hist = [r['compliance_best_so_far'] for r in runs]
        mean, std, n = _mean_std_bands(hist)
        x = np.arange(1, n + 1)
        ax.plot(x, mean, color=st['color'], ls=st['ls'], marker=st['marker'],
                markevery=_markevery(n), ms=3.5, mfc='white', mew=0.7,
                label=labels[method])
        ax.fill_between(x, mean - std, mean + std, color=st['color'], alpha=0.16, linewidth=0)
    ax.set_xlabel("real FE evaluations")
    ax.set_ylabel(ylabel + " (best-so-far)")
    ax.set_yscale('log')
    ax.set_title(title)
    ax.legend(loc='upper right')
    fig.tight_layout()
    savefig(fig, outname)


# --------------------------------------------------------------------------
# fig_cs2_design (Fig. 5, COLOR) -- keep before/after section-size coloring
# of beams; add the same beam-to-beam connection dot treatment (no link
# lines) used in fig_typologies.
# --------------------------------------------------------------------------
def fig_cs2_design():
    from reciprocal_geom import square_grillage, physical_beams, true_node_positions
    from sections import CHS_CATALOG, N_SECTIONS

    d = json.load(open(f"{RESULTS}/cs2_grillage.json"))
    sa_runs = d['methods']['sa_qubo']
    best_run = min(sa_runs, key=lambda r: r['best_compliance'])
    final_idx = best_run['final_idx']

    model = square_grillage(4, 4, cell=3.0, lam=0.25, n_rings=3)
    nodes = np.array(model.nodes)
    tpos = true_node_positions(model)
    beams = physical_beams(model)
    mids = _link_midpoints(model, nodes)

    fig, axes = plt.subplots(1, 2, figsize=(180 * MM, 95 * MM), subplot_kw={'projection': '3d'})
    cmap = plt.get_cmap('viridis')

    for ax, idx_vec, title in ((axes[0], [3] * model.n_groups, "(a) Uniform initial design (mid-catalog)"),
                                (axes[1], final_idx, "(b) SA-QUBO optimized (mass-target 60%)")):
        # each PHYSICAL beam drawn as ONE straight rod (true, non-eccentric
        # support->tip axis), coloured/thickened by its own group's section
        # -- not the FE support->interior->tip polyline, which kinks at the
        # interior node's +-ecc/2 offset.
        for bm in beams:
            p1, p2 = tpos[bm['support_idx']], tpos[bm['tip_idx']]
            sec_idx = idx_vec[bm['group']]
            color = cmap(sec_idx / (N_SECTIONS - 1))
            D = CHS_CATALOG[sec_idx]['D_mm']
            lw = 0.6 + 2.2 * (D / CHS_CATALOG[-1]['D_mm'])
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=color, lw=lw)
        if len(mids):
            ax.scatter(mids[:, 0], mids[:, 1], mids[:, 2], color=COLORS['accent'],
                       marker='o', s=6, zorder=6, depthshade=False, linewidth=0)
        ax.set_title(title, fontsize=8)
        ax.set_box_aspect([1, 1, 0.15])
        ax.view_init(elev=42, azim=-60)
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
        ax.grid(False)
        for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
            pane.pane.set_alpha(0.0)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(
        vmin=CHS_CATALOG[0]['D_mm'], vmax=CHS_CATALOG[-1]['D_mm']))
    cbar = fig.colorbar(sm, ax=axes, orientation='horizontal', fraction=0.05, pad=0.02, shrink=0.5)
    cbar.set_label("CHS outside diameter D [mm]  --  dots: beam-to-beam connection points")
    savefig(fig, "fig_cs2_design")


# --------------------------------------------------------------------------
# fig_cs1_design (CS1 fan-unit brute-force global optima, COLOR) -- same
# straight-beam / dot / viridis-by-D convention as fig_cs2_design. Both
# panels use the SAME (single, uniform) section, verified against the
# brute-force global optimum stored in results/cs1_validation.json
# (global_opt_lambda_idx / global_opt_section_idx, identical for the 3-beam
# and 4-beam fan: lambda=0.500, D=244.5mm -- the largest catalog section at
# the loosest engagement ratio, i.e. compliance is monotonically improved by
# stiffness in this small validation instance).
# --------------------------------------------------------------------------
def fig_cs1_design():
    from reciprocal_geom import fan_unit, physical_beams, true_node_positions
    from sections import CHS_CATALOG, N_SECTIONS

    d = json.load(open(f"{RESULTS}/cs1_validation.json"))
    cmap = plt.get_cmap('viridis')

    fig, axes = plt.subplots(1, 2, figsize=(170 * MM, 82 * MM), subplot_kw={'projection': '3d'})
    for ax, n_beams, key, title in ((axes[0], 3, 'fan3', "(a) 3-beam fan (global optimum)"),
                                     (axes[1], 4, 'fan4', "(b) 4-beam fan (global optimum)")):
        r = d[key]
        lam_idx = r['global_opt_lambda_idx']
        sec_idx = r['global_opt_section_idx']
        lam = r['lambda_levels'][lam_idx]
        print(f"[fig_cs1_design] {key}: decoded optimum lambda_idx={lam_idx} "
              f"(lambda={lam:.3f}), section_idx={sec_idx} (D={CHS_CATALOG[sec_idx]['D_mm']}mm), "
              f"compliance={r['global_opt_compliance']:.6e}")
        assert abs(lam - 0.5) < 1e-9 and CHS_CATALOG[sec_idx]['D_mm'] == 244.5, \
            "CS1 global optimum no longer matches expected lambda=0.500, D=244.5mm"

        model = fan_unit(n_beams, R=3.0, lam=lam, single_group=True)
        nodes = np.array(model.nodes)
        tpos = true_node_positions(model)
        beams = physical_beams(model)
        mids = _link_midpoints(model, nodes)

        color = cmap(sec_idx / (N_SECTIONS - 1))
        D = CHS_CATALOG[sec_idx]['D_mm']
        lw = 0.6 + 2.2 * (D / CHS_CATALOG[-1]['D_mm'])
        for bm in beams:
            p1, p2 = tpos[bm['support_idx']], tpos[bm['tip_idx']]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=color, lw=lw)
        if len(mids):
            ax.scatter(mids[:, 0], mids[:, 1], mids[:, 2], color=COLORS['accent'],
                       marker='o', s=10, zorder=6, depthshade=False, linewidth=0)
        sup = nodes[list(model.support_idx)]
        ax.scatter(sup[:, 0], sup[:, 1], sup[:, 2], color='k', marker='^',
                   s=16, zorder=10, depthshade=False)
        ax.set_title(title, fontsize=8)
        ax.set_box_aspect([1, 1, 0.16])
        ax.view_init(elev=24, azim=-60)
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
        ax.grid(False)
        for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
            pane.pane.set_alpha(0.0)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(
        vmin=CHS_CATALOG[0]['D_mm'], vmax=CHS_CATALOG[-1]['D_mm']))
    cbar = fig.colorbar(sm, ax=axes, orientation='horizontal', fraction=0.05, pad=0.02, shrink=0.5)
    cbar.set_label("CHS outside diameter D [mm]  --  dots: beam-to-beam connection points\n"
                    "(brute-force global optimum for both fans: $\\lambda$=0.500, D=244.5mm)")
    savefig(fig, "fig_cs1_design")


# --------------------------------------------------------------------------
# fig_cs3_design (CS3 hex grillage, COLOR) -- same convention as
# fig_cs2_design: (a) uniform mid-catalog initial design, (b) SA-QUBO
# optimized best design. The best design is NOT simply the stored
# `final_idx` in results/cs3_hex.json (that is the LAST sizing iteration's
# assignment, not the best-so-far one -- see r6_uls.py docstring); the true
# per-group section vector that achieves the recorded best_compliance was
# already recovered deterministically (re-running iterative_qubo_sizing
# with the identical seed/hyperparameters and matching the recomputed
# best-so-far compliance to the stored value) in
# results/study/r6_uls.json. That verified vector is reused here rather
# than re-deriving it.
# --------------------------------------------------------------------------
def fig_cs3_design():
    from reciprocal_geom import hex_grillage, physical_beams, true_node_positions
    from sections import CHS_CATALOG, N_SECTIONS

    d = json.load(open(f"{RESULTS}/cs3_hex.json"))
    sa_runs = d['methods']['sa_qubo']
    best_run = min(sa_runs, key=lambda r: r['best_compliance'])

    r6 = json.load(open(f"{RESULTS}/study/r6_uls.json"))
    rec = r6['cs3']
    assert rec['seed'] == best_run['seed'], "r6_uls.json cs3 seed does not match cs3_hex.json best seed"
    assert rec['reproducibility_match'], "r6_uls.json cs3 recovery was not verified reproducible"
    best_idx = rec['section_idx_per_group']
    print(f"[fig_cs3_design] best seed={rec['seed']} best_compliance="
          f"{best_run['best_compliance']:.4f} (r6-recovered compliance={rec['compliance']:.4f}) "
          f"section_idx_per_group={best_idx}")

    model = hex_grillage(rings=2, edge=3.0, lam=0.25)
    assert model.n_groups == len(best_idx), "n_groups mismatch vs recovered design vector"
    nodes = np.array(model.nodes)
    tpos = true_node_positions(model)
    beams = physical_beams(model)
    mids = _link_midpoints(model, nodes)

    fig, axes = plt.subplots(1, 2, figsize=(180 * MM, 95 * MM), subplot_kw={'projection': '3d'})
    cmap = plt.get_cmap('viridis')

    for ax, idx_vec, title in ((axes[0], [3] * model.n_groups, "(a) Uniform initial design (mid-catalog)"),
                                (axes[1], best_idx, "(b) SA-QUBO optimized (mass-target 60%)")):
        for bm in beams:
            p1, p2 = tpos[bm['support_idx']], tpos[bm['tip_idx']]
            sec_idx = idx_vec[bm['group']]
            color = cmap(sec_idx / (N_SECTIONS - 1))
            D = CHS_CATALOG[sec_idx]['D_mm']
            lw = 0.6 + 2.2 * (D / CHS_CATALOG[-1]['D_mm'])
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=color, lw=lw)
        if len(mids):
            ax.scatter(mids[:, 0], mids[:, 1], mids[:, 2], color=COLORS['accent'],
                       marker='o', s=6, zorder=6, depthshade=False, linewidth=0)
        sup = nodes[list(model.support_idx)]
        ax.scatter(sup[:, 0], sup[:, 1], sup[:, 2], color='k', marker='^',
                   s=14, zorder=10, depthshade=False)
        ax.set_title(title, fontsize=8)
        ax.set_box_aspect([1, 1, 0.15])
        ax.view_init(elev=42, azim=-60)
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
        ax.grid(False)
        for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
            pane.pane.set_alpha(0.0)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(
        vmin=CHS_CATALOG[0]['D_mm'], vmax=CHS_CATALOG[-1]['D_mm']))
    cbar = fig.colorbar(sm, ax=axes, orientation='horizontal', fraction=0.05, pad=0.02, shrink=0.5)
    cbar.set_label("CHS outside diameter D [mm]  --  dots: beam-to-beam connection points")
    savefig(fig, "fig_cs3_design")


# --------------------------------------------------------------------------
# fig_cs4_design (CS4 reciprocal dome, COLOR) -- same convention as
# fig_cs2_design/fig_cs3_design. Unlike CS2/CS3 (fixed geometry, section
# sizing only), CS4's black-box design vector also controls the per-ring
# engagement ratio lambda, which changes the DOME GEOMETRY itself (via a
# fresh dome() regeneration -- see cs4_dome.py / fmqa.py), so panel (a) and
# (b) are genuinely different geometries, not just different colors on the
# same frame. The FMQA-optimum per-ring lambda + global section were
# recovered deterministically (identical re-run of fmqa_optimize) and
# verified against the stored final_best in results/study/r6_uls.json.
# --------------------------------------------------------------------------
def fig_cs4_design():
    from reciprocal_geom import dome, physical_beams, true_node_positions
    from sections import CHS_CATALOG, N_SECTIONS

    d = json.load(open(f"{RESULTS}/cs4_dome.json"))
    fmqa_runs = d['methods']['fmqa']
    best_run = min(fmqa_runs, key=lambda r: r['final_best'])

    r6 = json.load(open(f"{RESULTS}/study/r6_uls.json"))
    rec = r6['cs4']
    assert rec['seed'] == best_run['seed'], "r6_uls.json cs4 seed does not match cs4_dome.json best seed"
    assert rec['reproducibility_match'], "r6_uls.json cs4 recovery was not verified reproducible"
    lam_per_ring_opt = rec['lam_per_ring']
    sec_idx_opt = rec['section_idx_global']
    print(f"[fig_cs4_design] best seed={rec['seed']} final_best={best_run['final_best']:.4f} "
          f"(r6-recovered compliance={rec['compliance']:.4f}) "
          f"lam_per_ring={lam_per_ring_opt} section_idx_global={sec_idx_opt} "
          f"(D={CHS_CATALOG[sec_idx_opt]['D_mm']}mm)")

    LAM_LEVELS = np.linspace(0.05, 0.5, 8)
    lam_ref = float(LAM_LEVELS[3])       # mid-catalog reference lambda level
    sec_ref = 3                          # mid-catalog reference section
    RINGS, SPAN, RISE = 2, 16.0, 3.0

    model_ref = dome(RINGS, span=SPAN, rise=RISE, lam_per_ring=[lam_ref] * 3)
    model_opt = dome(RINGS, span=SPAN, rise=RISE, lam_per_ring=lam_per_ring_opt)

    fig, axes = plt.subplots(1, 2, figsize=(180 * MM, 100 * MM), subplot_kw={'projection': '3d'})
    cmap = plt.get_cmap('viridis')

    panels = (
        (axes[0], model_ref, sec_ref,
         "(a) Initial/reference design\n"
         f"($\\lambda$={lam_ref:.2f} all rings, mid-catalog D)"),
        (axes[1], model_opt, sec_idx_opt,
         "(b) FMQA optimum\n"
         "($\\lambda$=" + ", ".join(f"{v:.2f}" for v in lam_per_ring_opt) +
         " by ring; D=" + f"{CHS_CATALOG[sec_idx_opt]['D_mm']:.1f}mm)"),
    )
    for ax, model, sec_idx, title in panels:
        nodes = np.array(model.nodes)
        tpos = true_node_positions(model)
        beams = physical_beams(model)
        mids = _link_midpoints(model, nodes)
        color = cmap(sec_idx / (N_SECTIONS - 1))
        D = CHS_CATALOG[sec_idx]['D_mm']
        lw = 0.6 + 2.2 * (D / CHS_CATALOG[-1]['D_mm'])
        for bm in beams:
            p1, p2 = tpos[bm['support_idx']], tpos[bm['tip_idx']]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=color, lw=lw)
        if len(mids):
            ax.scatter(mids[:, 0], mids[:, 1], mids[:, 2], color=COLORS['accent'],
                       marker='o', s=6, zorder=6, depthshade=False, linewidth=0)
        sup = nodes[list(model.support_idx)]
        ax.scatter(sup[:, 0], sup[:, 1], sup[:, 2], color='k', marker='^',
                   s=14, zorder=10, depthshade=False)
        ax.set_title(title, fontsize=7.4)
        ax.set_box_aspect([1, 1, 0.32])
        ax.view_init(elev=24, azim=-60)
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
        ax.grid(False)
        for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
            pane.pane.set_alpha(0.0)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(
        vmin=CHS_CATALOG[0]['D_mm'], vmax=CHS_CATALOG[-1]['D_mm']))
    cbar = fig.colorbar(sm, ax=axes, orientation='horizontal', fraction=0.05, pad=0.02, shrink=0.5)
    cbar.set_label("CHS outside diameter D [mm]  --  dots: beam-to-beam connection points")
    savefig(fig, "fig_cs4_design")


# --------------------------------------------------------------------------
# fig_cs4_fmqa -- GRAYSCALE
# --------------------------------------------------------------------------
def fig_cs4_fmqa():
    d = json.load(open(f"{RESULTS}/cs4_dome.json"))
    fig, ax = plt.subplots(figsize=(90 * MM, 70 * MM))
    labels = {'fmqa': 'FMQA (F2)', 'ga': 'GA', 'random': 'Random search'}
    for i, method in enumerate(('fmqa', 'ga', 'random')):
        st = _BW_SERIES[i]
        runs = d['methods'][method]
        hist = [r['best_history'] for r in runs]
        mean, std, n = _mean_std_bands(hist)
        x = np.arange(1, n + 1)
        ax.plot(x, mean, color=st['color'], ls=st['ls'], marker=st['marker'],
                markevery=_markevery(n), ms=3.5, mfc='white', mew=0.7,
                label=labels[method])
        ax.fill_between(x, mean - std, mean + std, color=st['color'], alpha=0.16, linewidth=0)
    ax.set_xlabel("real FE evaluations")
    ax.set_ylabel("compliance (best-so-far)")
    ax.set_yscale('log')
    ax.set_title("CS4 dome: FMQA vs baselines (10 seeds)")
    ax.legend(loc='upper right')
    fig.tight_layout()
    savefig(fig, "fig_cs4_fmqa")


# --------------------------------------------------------------------------
# fig_cs5_pareto -- GRAYSCALE. R5 matched 264-eval budget, 10 seeds,
# representative non-dominated fronts (panel a) + hypervolume mean+/-std
# with per-seed jittered dots (panel b).
# --------------------------------------------------------------------------
def fig_cs5_pareto():
    d = json.load(open(f"{RESULTS}/study/r5_analysis.json"))

    sa_front = np.array(d['sa_qubo']['representative_front'])
    nsga_front = np.array(d['nsga2']['representative_front'])
    sa_rep = d['sa_qubo']['representative_seed']
    nsga_rep = d['nsga2']['representative_seed']

    fig, axes = plt.subplots(1, 2, figsize=(180 * MM, 78 * MM))

    ax = axes[0]
    order_sa = np.argsort(sa_front[:, 0])
    order_nsga = np.argsort(nsga_front[:, 0])
    ax.plot(sa_front[order_sa, 0], sa_front[order_sa, 1], color='k', ls='-', marker='o',
            ms=3, lw=1.0, mfc='white', mew=0.6, label=f'SA-QUBO weighted-sum (seed {sa_rep})')
    ax.plot(nsga_front[order_nsga, 0], nsga_front[order_nsga, 1], color='0.45', ls='--',
            marker='s', ms=3, lw=1.0, mfc='white', mew=0.6, label=f'NSGA-II (seed {nsga_rep})')
    ax.set_xlabel("normalized compliance")
    ax.set_ylabel("normalized mass")
    ax.set_xscale('log')
    ax.set_title("(a) representative-seed non-dom. fronts\n(matched 264-eval budget)")
    ax.legend(loc='upper right', fontsize=6)

    ax2 = axes[1]
    methods = ['sa_qubo', 'nsga2']
    labels = ['SA-QUBO\nweighted-sum', 'NSGA-II']
    marker_shape = ['o', 's']
    grays = ['k', '0.45']
    means = [d[m]['hv_mean'] for m in methods]
    stds = [d[m]['hv_std'] for m in methods]
    x = np.arange(2)
    for i, m in enumerate(methods):
        seeds_hv = d[m]['hv_per_seed']
        jitter = (np.random.default_rng(0).random(len(seeds_hv)) - 0.5) * 0.12
        ax2.scatter(np.full(len(seeds_hv), x[i]) + jitter, seeds_hv, s=11,
                    color=grays[i], alpha=0.30, zorder=2, linewidth=0, marker=marker_shape[i])
    ax2.errorbar(x, means, yerr=stds, fmt='none', capsize=4, color='k', elinewidth=1.1, zorder=3)
    for i in range(2):
        ax2.scatter([x[i]], [means[i]], s=46, color=grays[i], marker=marker_shape[i], zorder=4,
                    edgecolor='k', linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xlim(-0.5, 1.5)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("2D hypervolume (ref (1.2,1.2), norm. units)")
    ax2.set_title(f"(b) hypervolume, 10 seeds each (dots)\n(+{d['pct_hv_advantage_sa']:.1f}% SA-QUBO, "
                  f"p={d['mann_whitney_sa_vs_nsga']['p']:.1e})")

    fig.tight_layout()
    savefig(fig, "fig_cs5_pareto")


# --------------------------------------------------------------------------
# fig_ablation -- GRAYSCALE. CS2 w_mass=20 best-so-far compliance vs FE
# evaluations: SA-QUBO vs greedy-linearized vs GA. SA-QUBO and
# greedy-linearized are the PAIRED (same 10 seeds) w_mass=20 ablation runs
# from r3_ablation.json; GA is the ORIGINAL w_mass=5 CS2 run from
# results/cs2_grillage.json (no matched w_mass=20 GA rerun exists; stated
# explicitly since it is not perfectly budget/weight-matched to the two
# w_mass=20 curves).
# --------------------------------------------------------------------------
def fig_ablation():
    d3 = json.load(open(f"{RESULTS}/study/r3_ablation.json"))
    d2 = json.load(open(f"{RESULTS}/cs2_grillage.json"))

    sa_hist = [r['compliance_best_so_far'] for r in d3['methods']['sa_qubo_paired_w20']]
    greedy_hist = [r['compliance_best_so_far'] for r in d3['methods']['greedy_w20']]
    ga_hist = [r['compliance_best_so_far'] for r in d2['methods']['ga']]

    fig, ax = plt.subplots(figsize=(95 * MM, 75 * MM))
    series = [('SA-QUBO (F1, w=20)', sa_hist, _BW_SERIES[0]),
              ('Greedy-linearized (w=20)', greedy_hist, _BW_SERIES[1]),
              ('GA (w=5, original)', ga_hist, _BW_SERIES[2])]
    for label, hist, st in series:
        mean, std, n = _mean_std_bands(hist)
        x = np.arange(1, n + 1)
        ax.plot(x, mean, color=st['color'], ls=st['ls'], marker=st['marker'],
                markevery=_markevery(n), ms=3.5, mfc='white', mew=0.7, label=label)
        ax.fill_between(x, mean - std, mean + std, color=st['color'], alpha=0.16, linewidth=0)
    ax.set_xlabel("real FE evaluations")
    ax.set_ylabel("compliance (best-so-far)")
    ax.set_yscale('log')
    ax.set_title("CS2 ablation: SA-QUBO vs greedy-linearized vs GA\n(mean±std, 10 seeds)")
    ax.legend(loc='upper right', fontsize=6.5)
    fig.tight_layout()
    savefig(fig, "fig_ablation")


# --------------------------------------------------------------------------
# fig_sensitivity -- GRAYSCALE (twin-axis; series distinguished by
# linestyle + marker, combined into a single legend)
# --------------------------------------------------------------------------
def fig_sensitivity():
    d = json.load(open(f"{RESULTS}/penalty_sensitivity.json"))
    runs = d['runs']
    w_levels = sorted(set(r['w_mass'] for r in runs))
    mean_err, std_err, mean_C, std_C = [], [], [], []
    for w in w_levels:
        rs = [r for r in runs if r['w_mass'] == w]
        mean_err.append(np.mean([abs(r['mass_error_pct']) for r in rs]))
        std_err.append(np.std([abs(r['mass_error_pct']) for r in rs]))
        mean_C.append(np.mean([r['final_compliance'] for r in rs]))
        std_C.append(np.std([r['final_compliance'] for r in rs]))

    fig, ax1 = plt.subplots(figsize=(95 * MM, 75 * MM))
    l1 = ax1.errorbar(w_levels, mean_err, yerr=std_err, color='k', marker='o', ls='-',
                       mfc='white', mew=0.9, capsize=3, label='|mass error| [%]  (solid, o)')
    ax1.set_xlabel(r"mass-target penalty weight $w_{mass}$")
    ax1.set_ylabel("|mass target error| [%]")
    ax1.set_xscale('log')

    ax2 = ax1.twinx()
    l2 = ax2.errorbar(w_levels, mean_C, yerr=std_C, color='0.45', marker='s', ls='--',
                       mfc='white', mew=0.9, capsize=3, label='final compliance  (dashed, s)')
    ax2.set_ylabel("final compliance")

    ax1.legend([l1, l2], [l1.get_label(), l2.get_label()], loc='lower center',
               bbox_to_anchor=(0.5, 1.02), ncol=1, fontsize=6.3, framealpha=0.95)
    ax1.set_title("CS2 penalty-weight sensitivity (3 seeds/level)", pad=34)
    fig.tight_layout()
    savefig(fig, "fig_sensitivity")


# --------------------------------------------------------------------------
# fig_fan_geometry (NEW, B/W) -- 2D top-view schematic of the rotation-method
# reciprocal fan construction. Coordinates are derived ANALYTICALLY here
# (not read off the 3-D/eccentric FE generator in reciprocal_geom.py, whose
# oblique projection made the rotor unreadable) so every point drawn is
# provably exact:
#   - S_k sit on the support circle of radius R, k=0..n-1, 90 deg apart.
#   - beam k's straight axis is tangent to the engagement circle r_e=lambda*R.
#   - beam k's own TIP is exactly X_k := line(k) ^ line(k+1), i.e. it lands
#     exactly on the axis of the NEXT beam -- that is the bearing/connection
#     point. Beam k is drawn only from S_k to X_k (never past its own tip).
#   - X_{k-1} (where the PREVIOUS beam bears on beam k) lies, by construction,
#     strictly between S_k and X_k on that same straight line, so no dot is
#     ever off a beam and no beam ever needs to be split/kinked to hit it.
# This matches reciprocal_geom.pinwheel_fan's sign convention (chirality=-1)
# and its lambda = engagement-length ratio, r_e = lambda*R, definition.
#
# REVISION (author feedback -- "does not read as a true reciprocal fan" /
# "labels overlap"): the previous single-panel version put every callout
# (support circle, engagement circle, engagement-length bracket, rotation
# angle) on long diagonal leader lines running from the outer axes corners
# all the way into the crowded central engagement zone, so leaders crossed
# beams, each other, and (for lambda=0.15, where the whole engagement
# cluster spans <5% of the panel) crossed straight through the connection
# dots themselves -- unreadable, and for the tight rotor the four beams
# nearly converging on one another over a tiny central patch did not read
# as a 4-point mutually-supporting rotor at all. Fix: every panel now
# carries (a) a clean overview -- support circle, all four beams, open
# triangle supports, S_k labels, and the rotation-angle arc at S0, none of
# which ever needs a leader line that crosses the fan -- plus (b) a
# magnified "engagement detail" inset (mpl axes_grid1 mark_inset) that
# contains ONLY the small central cluster (engagement circle, the 4
# connection dots, the tangent beam stubs, and the engagement-length
# dimension bracket), each with a short LOCAL leader inside that dedicated,
# uncluttered inset canvas. Nothing in either axes ever crosses anything
# else; the mark_inset connector lines are the only lines that cross beam
# graphics, and those are the standard, unambiguous "this region is
# magnified over there" convention.
# --------------------------------------------------------------------------
def _fan_geometry(n, R, lam, chirality=-1):
    r_e = lam * R
    phi = np.array([k * 2 * np.pi / n for k in range(n)])
    S = np.stack([R * np.cos(phi), R * np.sin(phi)], axis=1)

    # tangent line from each support to the engagement circle: the
    # tangent point sits at angular offset alpha_c (seen from the centre)
    # from the support's own angle; psi = arcsin(r_e/R) is the equivalent
    # deviation seen FROM the support (angle between the radial spoke
    # S_k->centre and the beam axis) -- the textbook "rotation angle" of
    # the rotation method.
    alpha_c = np.arccos(np.clip(r_e / R, -1, 1))
    psi = np.arcsin(np.clip(r_e / R, -1, 1))
    tpt_theta = phi - chirality * alpha_c
    Tpt = r_e * np.stack([np.cos(tpt_theta), np.sin(tpt_theta)], axis=1)
    D = Tpt - S
    D = D / np.linalg.norm(D, axis=1, keepdims=True)

    def isect(pa, da, pb, db):
        A = np.array([[da[0], -db[0]], [da[1], -db[1]]])
        t, _ = np.linalg.solve(A, pb - pa)
        return pa + t * da

    X = np.zeros((n, 2))  # X[k] = beam k's own tip = bearing point on beam k+1
    for k in range(n):
        kp1 = (k + 1) % n
        X[k] = isect(S[k], D[k], S[kp1], D[kp1])

    return dict(n=n, R=R, r_e=r_e, phi=phi, S=S, D=D, X=X, psi=psi)


def _draw_fan(ax, g, hi=0, dot_size=30, sup_size=62, tri_lw=1.2,
              hi_lw=2.6, lo_lw=1.15, support_labels=False, ha_va=None,
              label_r_frac=0.32, fontsize=7.5, circles=True):
    """Draw beams + connection dots (+ optionally supports/circles) for one
    fan geometry dict `g` into axes `ax`. Shared by the main overview panel
    and the magnified detail inset so both stay geometrically identical."""
    n, R, r_e, S, X = g['n'], g['R'], g['r_e'], g['S'], g['X']
    for k in range(n):
        hl = (k == hi)
        ax.plot([S[k, 0], X[k, 0]], [S[k, 1], X[k, 1]],
                color='k' if hl else '0.5', lw=hi_lw if hl else lo_lw,
                solid_capstyle='round', zorder=4 if hl else 3)
    ax.scatter(X[:, 0], X[:, 1], color='k', s=dot_size, zorder=6)
    if sup_size > 0:
        ax.scatter(S[:, 0], S[:, 1], marker='^', s=sup_size, zorder=6,
                   facecolor='white', edgecolor='k', linewidth=tri_lw)
    if support_labels:
        for k in range(n):
            d = S[k] / np.linalg.norm(S[k])
            lp = S[k] + label_r_frac * R * d
            ha, va = ha_va[k]
            ax.annotate(f"$S_{{{k}}}$", xy=(lp[0], lp[1]), ha=ha, va=va, fontsize=fontsize)
    if circles:
        th = np.linspace(0, 2 * np.pi, 200)
        ax.plot(R * np.cos(th), R * np.sin(th), color='0.65', lw=0.8, ls='--', zorder=1)
        ax.plot(r_e * np.cos(th), r_e * np.sin(th), color='0.3', lw=0.9, ls='-.', zorder=1)


def _dim_bracket(ax, p0, p1, nhat, off_mag, tick_frac=0.16, label=None,
                  label_gap=1.15, fontsize=6.2, lead_color='0.35', color='k'):
    """Standard offset dimension line: tie lines from p0,p1 out to the
    offset line, perpendicular end ticks, a double-headed arrow between
    them, and the label pushed CLEAR of that arrow (another off_mag*
    label_gap further out along the same offset direction -- never on top
    of the arrow, never back on the beam)."""
    off = off_mag * nhat
    q0, q1 = p0 + off, p1 + off
    ax.plot([p0[0], q0[0]], [p0[1], q0[1]], color=lead_color, lw=0.5, zorder=5)
    ax.plot([p1[0], q1[0]], [p1[1], q1[1]], color=lead_color, lw=0.5, zorder=5)
    perp = np.array([-nhat[1], nhat[0]])
    ticklen = tick_frac * off_mag
    for q in (q0, q1):
        ax.plot([q[0] - ticklen * perp[0], q[0] + ticklen * perp[0]],
                [q[1] - ticklen * perp[1], q[1] + ticklen * perp[1]],
                color=color, lw=0.8, zorder=7)
    ax.annotate("", xy=tuple(q1), xytext=tuple(q0),
                arrowprops=dict(arrowstyle='<->', color=color, lw=0.85), zorder=7)
    if label:
        mid = 0.5 * (q0 + q1) + label_gap * off_mag * nhat
        ax.annotate(label, xy=tuple(mid), xycoords='data', ha='center', va='center',
                    fontsize=fontsize, color='k', zorder=9,
                    bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=1.0))


def fig_fan_geometry():
    n = 4
    R = 3.0
    lam_vals = [0.15, 0.45]
    lam_desc = ['tight rotor', 'open rotor']
    bbox = dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.92)
    ha_va = {0: ('left', 'center'), 1: ('center', 'bottom'),
             2: ('right', 'center'), 3: ('center', 'top')}
    # only the tight rotor's central cluster is too small to annotate
    # in-place; the open rotor's cluster is already a sizeable fraction of
    # the panel and is annotated directly (short LOCAL leaders only).
    INSET_LAM_MAX = 0.25
    INSET_BOX = [0.03, 0.05, 0.34, 0.34]  # bottom-left corner, axes fraction

    fig, axes = plt.subplots(1, 2, figsize=(190 * MM, 100 * MM))
    for ax, lam, desc in zip(axes, lam_vals, lam_desc):
        g = _fan_geometry(n, R, lam)
        S, X, D, r_e = g['S'], g['X'], g['D'], g['r_e']
        psi_deg = np.degrees(g['psi'])
        R_cluster = np.max(np.linalg.norm(X, axis=1))
        use_inset = lam <= INSET_LAM_MAX

        # --- main overview: fan + supports + circles + S_k labels -------
        _draw_fan(ax, g, hi=0, support_labels=True, ha_va=ha_va,
                   dot_size=(14 if use_inset else 30), label_r_frac=0.40)

        ax.set_aspect('equal')
        pad = 1.55 * R
        ax.set_xlim(-pad, pad); ax.set_ylim(-pad, pad)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        # support-circle callout: short leader anchored in the SAME
        # (upper-left) corner as its target angle on the circle -- the
        # leader stays entirely in the empty gap between S1 and S2 and
        # never has to cross the fan to get there.
        ang_R = np.radians(135)
        ax.annotate("support circle $R$",
                    xy=(R * np.cos(ang_R), R * np.sin(ang_R)), xycoords='data',
                    xytext=(-0.98 * pad, 0.96 * pad), textcoords='data', bbox=bbox,
                    fontsize=6.4, color='0.35', ha='left', va='top',
                    arrowprops=dict(arrowstyle='-', color='0.6', lw=0.6,
                                     shrinkA=0, shrinkB=2))

        # rotation angle psi at S0: small arc between the radial spoke
        # (S0 -> centre) and the beam-0 axis, both measured at S0 -- S0
        # sits out at radius R, far from the crowded centre, so the label
        # is placed immediately next to the arc (no leader needed at all).
        v_rad = -S[0] / np.linalg.norm(S[0])
        v_beam = D[0]
        ang_rad = np.degrees(np.arctan2(v_rad[1], v_rad[0]))
        ang_beam = np.degrees(np.arctan2(v_beam[1], v_beam[0]))
        # shortest signed angular difference (avoids the +-180 deg wrap
        # turning a small ~psi-degree wedge into a near-full circle)
        dpsi = (ang_beam - ang_rad + 180) % 360 - 180
        ang_beam = ang_rad + dpsi
        r_arc = 0.20 * R
        arc = mpatches.Arc(S[0], 2 * r_arc, 2 * r_arc, angle=0,
                            theta1=min(ang_rad, ang_beam), theta2=max(ang_rad, ang_beam),
                            color='k', lw=0.9, zorder=7)
        ax.add_patch(arc)
        # label sits BELOW S0 (tangential direction, perpendicular to the
        # radial spoke, on the side opposite the beam): the beam always
        # bends to +y at S0 for this chirality, and the "S0" support label
        # itself sits further out along +x, so -y is clear of both.
        lbl_pt = S[0] + np.array([0.0, -1.5 * r_arc])
        ax.annotate(f"$\\psi\\approx{psi_deg:.0f}^\\circ$", xy=(lbl_pt[0], lbl_pt[1]),
                    xycoords='data', ha='center', va='center', fontsize=6.6, color='k',
                    bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='none', alpha=0.85))

        ax.set_title(f"$\\lambda = {lam:.2f}$ ({desc})", fontsize=8.5)

        C0, T0 = X[n - 1], X[0]  # bearing point on beam0 <- tip of beam0
        dhat = (T0 - C0) / np.linalg.norm(T0 - C0)
        nhat = np.array([-dhat[1], dhat[0]])
        mid_out = 0.5 * (C0 + T0)
        mid_out = mid_out / (np.linalg.norm(mid_out) + 1e-12)
        if np.dot(nhat, mid_out) < 0:
            nhat = -nhat

        if use_inset:
            # --- TIGHT ROTOR: the whole engagement cluster (circle, 4
            # dots, bracket) is far too small to label in place, so it is
            # redrawn, magnified, in a dedicated bottom-left inset with
            # only LOCAL leaders -- the inset is placed opposite S0/psi
            # and below S1/S2, so it never touches any main-panel label.
            zoom_r = 1.9 * R_cluster
            zth = np.linspace(0, 2 * np.pi, 100)
            ax.plot(zoom_r * np.cos(zth), zoom_r * np.sin(zth),
                    color='0.65', lw=0.6, ls=':', zorder=2)
            axins = ax.inset_axes(INSET_BOX)
            _draw_fan(axins, g, hi=0, dot_size=46, sup_size=0, hi_lw=3.0, lo_lw=1.5,
                       circles=True)
            axins.set_xlim(-zoom_r, zoom_r); axins.set_ylim(-zoom_r, zoom_r)
            axins.set_aspect('equal')
            axins.set_xticks([]); axins.set_yticks([])
            for spine in axins.spines.values():
                spine.set_color('0.35'); spine.set_linewidth(0.8)
            axins.set_title("engagement detail (zoom)", fontsize=5.8, pad=2.2)
            # single connector: dotted circle (region shown) -> RIGHT-EDGE
            # midpoint of the inset box (never the top corner, which sits
            # right under the inset's own title text).
            conn_ang = np.radians(233)
            p_from = (zoom_r * np.cos(conn_ang), zoom_r * np.sin(conn_ang))
            inset_right_mid = (INSET_BOX[0] + INSET_BOX[2],
                                INSET_BOX[1] + 0.5 * INSET_BOX[3])
            ax.annotate("", xy=inset_right_mid,
                        xycoords='axes fraction', xytext=p_from, textcoords='data',
                        arrowprops=dict(arrowstyle='-', color='0.6', lw=0.6,
                                         shrinkA=0, shrinkB=0), zorder=2)

            # engagement-circle label anchored at the BOTTOM of the inset,
            # dimension bracket occupies the top (beam0 runs low-right to
            # up-left here) -- opposite corners so the two never collide.
            ang_re = np.radians(255)
            axins.annotate(r"$r_e=\lambda R$",
                            xy=(r_e * np.cos(ang_re), r_e * np.sin(ang_re)), xycoords='data',
                            xytext=(-0.20 * zoom_r, -0.92 * zoom_r), textcoords='data',
                            fontsize=6.0, color='0.2', ha='center', va='bottom',
                            arrowprops=dict(arrowstyle='-', color='0.5', lw=0.55,
                                             shrinkA=0, shrinkB=2))
            _dim_bracket(axins, C0, T0, nhat, off_mag=0.34 * R_cluster,
                         label_gap=1.3, label="engagement\nlength", fontsize=5.6)
        else:
            # --- OPEN ROTOR: cluster is already large enough to label
            # in-place. NOTE the 4 beams' angular sweeps overlap and
            # together cover the full 360 deg (that overlap is exactly
            # what makes it read as a rotor), so ANY leader threading in
            # from outside generically clips a beam or a connection dot
            # somewhere. The one region that is PROVABLY beam-free is the
            # engagement circle's own interior (every beam is tangent to
            # it, i.e. stays at radius >= r_e everywhere), so the label is
            # simply set INSIDE the circle it names -- self-evident,
            # zero-length leader, zero crossing risk by construction.
            ax.annotate("engagement circle\n$r_e=\\lambda R$",
                        xy=(0, 0), xycoords='data', ha='center', va='center',
                        fontsize=6.1, color='0.25', zorder=8,
                        bbox=dict(boxstyle='round,pad=0.12', fc='white', ec='0.6', lw=0.5, alpha=0.95))
            _dim_bracket(ax, C0, T0, nhat, off_mag=0.22 * R_cluster,
                         label="engagement length", fontsize=6.2)

    # NOTE: in-figure title/footer removed (2026 study) -- legend text
    # ("Rotation-method reciprocal fan, top view (n=4). Black (bold):
    # highlighted beam 0; gray: beams 1-3; filled dot: connection (bearing)
    # point; triangle: support.") now given in the manuscript caption
    # instead (see latex/figure_title_migration.md).
    # aspect='equal' shrinks each panel to a square that fits the SMALLER
    # of its allocated width/height; with default matplotlib margins the
    # per-column width was well under the row height, so each square
    # ended up letterboxed with a large blank strip above/below. Pushing
    # left/right out to nearly the figure edges brings the column width
    # up to match the row height so the squares fill the panel.
    fig.subplots_adjust(left=0.02, right=0.98, wspace=0.30, top=0.96, bottom=0.03)
    savefig(fig, "fig_fan_geometry")


# --------------------------------------------------------------------------
# fig_encoding_qubo (NEW, B/W) -- (a) one-hot member-group encoding diagram;
# (b) REAL |Q| heatmap for one F1 QUBO iteration (CS2 grillage, seed 0,
# one real FE solve), rebuilt with the actual sizing.py/qubo.py functions.
# --------------------------------------------------------------------------
def _draw_onehot_panel(ax):
    """Stacked (not side-by-side) layout: one-hot vector row on top, CHS
    table below -- avoids cramming both into the narrow ~95mm panel width
    of the two-panel figure."""
    from sections import CHS_CATALOG, N_SECTIONS

    ax.set_xlim(0, 10.0); ax.set_ylim(0, 10.0)
    ax.axis('off')
    chosen = 3

    # --- top row: member group -> one-hot vector ---------------------
    ax.add_patch(mpatches.Rectangle((0.1, 7.35), 1.35, 1.5, fill=False, ec='k', lw=1.0))
    ax.text(0.78, 8.1, "group\n$k$", ha='center', va='center', fontsize=6.8)
    ax.annotate("", xy=(1.95, 8.1), xytext=(1.5, 8.1), arrowprops=dict(arrowstyle='->', lw=0.9))

    x0, box_w, gap = 2.05, 0.66, 0.11
    for c in range(N_SECTIONS):
        x = x0 + c * (box_w + gap)
        fc, tc = ('k', 'white') if c == chosen else ('white', 'k')
        ax.add_patch(mpatches.Rectangle((x, 7.75), box_w, box_w, fill=True, fc=fc, ec='k', lw=0.8))
        ax.text(x + box_w / 2, 7.75 + box_w / 2, "1" if c == chosen else "0",
                ha='center', va='center', fontsize=6.5, color=tc)
        ax.text(x + box_w / 2, 7.60, f"$x_{{{c}}}$", ha='center', va='top', fontsize=5.0)
    x_end = x0 + N_SECTIONS * (box_w + gap) - gap
    ax.text((x0 + x_end) / 2, 8.85, "one-hot vector (8 CHS sections)", ha='center', fontsize=6.4)
    ax.text((x0 + x_end) / 2, 6.95, f"argmax $\\rightarrow\\ c={chosen}$", ha='center', fontsize=6.0)

    # --- bottom: CHS catalog table ------------------------------------
    tx0, tx_w = 1.35, 7.3
    col_w = [1.0, 3.1, 3.2]
    headers = ["$c$", "$D$ [mm]", "$t$ [mm]"]
    ty0, row_h = 6.1, 0.60
    cx = tx0
    for h, w in zip(headers, col_w):
        ax.text(cx + w / 2, ty0, h, ha='center', va='center', fontsize=6.4, weight='bold')
        cx += w
    ax.plot([tx0 - 0.15, tx0 + tx_w + 0.15], [ty0 - row_h * 0.58] * 2, color='k', lw=0.8)
    for c in range(N_SECTIONS):
        ry = ty0 - (c + 1) * row_h - 0.14
        is_sel = (c == chosen)
        if is_sel:
            ax.add_patch(mpatches.Rectangle((tx0 - 0.15, ry - row_h / 2), tx_w + 0.30, row_h,
                                             fc='0.82', ec='none', zorder=0))
        cx = tx0
        vals = [str(c), f"{CHS_CATALOG[c]['D_mm']:.1f}", f"{CHS_CATALOG[c]['t_mm']:.1f}"]
        for v, w in zip(vals, col_w):
            ax.text(cx + w / 2, ry, v, ha='center', va='center', fontsize=6.0,
                    weight='bold' if is_sel else 'normal')
            cx += w
    ax.text(tx0 + tx_w / 2, ty0 + 0.42, "CHS section catalog", ha='center', fontsize=6.6)
    ax.set_title("(a) one-hot member-group $\\rightarrow$ section encoding", fontsize=8)


def _build_real_f1_qubo():
    """Rebuild ONE real F1 (strain-energy-linearized) QUBO exactly as
    sizing.iterative_qubo_sizing() does for its first outer iteration on
    the CS2 grillage (mid-catalog init, seed 0): one real FE solve, per
    (group, section) strain energies via the real local element stiffness
    matrices, then the actual QUBOBuilder one-hot + mass-target penalty
    terms. (The move-limit big-penalty exclusion term is omitted here so
    the underlying one-hot/mass-coupling block structure is visible in the
    heatmap; every other term is identical to sizing.py.)"""
    from reciprocal_geom import square_grillage, model_mass
    from sections import CHS_CATALOG, N_SECTIONS, E_STEEL, G_STEEL
    from sizing import element_geom_cache, group_element_indices, evaluate_design
    from qubo import QUBOBuilder
    from frame3d import local_stiffness

    # NOTE: n_rings=2 here (not the n_rings=3 used by the actual CS2 case
    # study elsewhere) -- a 4x4 grid only ever realises ring indices 0,1
    # (see square_grillage: ring = min(ix,iy,nx-1-ix,ny-1-iy) tops out at 1
    # for nx=ny=4), so with n_rings=3 the last 4 groups (32 bits) would be
    # pure placeholders with zero elements and hence a Q-matrix block that
    # is identically zero everywhere off its own one-hot diagonal --
    # exactly the kind of "floating" always-empty region a callout must
    # never point at. n_rings=2 makes every one of the n_groups blocks
    # correspond to a real, populated group.
    model = square_grillage(4, 4, cell=3.0, lam=0.25, n_rings=2)
    gs_max = {k: CHS_CATALOG[-1] for k in range(model.n_groups)}
    mass_target = 0.6 * model_mass(model, gs_max)
    n_groups, n_sec = model.n_groups, N_SECTIONS
    cur = [n_sec // 2] * n_groups
    _, _, frm = evaluate_design(model, cur, -2000.0)

    geom = element_geom_cache(model)
    groups = group_element_indices(model)
    SE = np.zeros((n_groups, n_sec))
    mass_coef = np.zeros((n_groups, n_sec))
    for k, elem_idxs in groups.items():
        for ei in elem_idxs:
            g = geom[ei]
            u_local = g['T'] @ frm.U[g['dofs']]
            for c in range(n_sec):
                sp = CHS_CATALOG[c]
                kl = local_stiffness(E_STEEL, G_STEEL, sp['A'], sp['Iy'], sp['Iz'], sp['J'], g['L'])
                SE[k, c] += 0.5 * u_local @ kl @ u_local
                mass_coef[k, c] += sp['mass_per_len'] * g['L']

    n_bits = n_groups * n_sec
    qb = QUBOBuilder(n_bits)

    def vidx(k, c):
        return k * n_sec + c

    SEmax = SE.max() if SE.max() > 0 else 1.0
    w_stiff, w_mass = 1.0, 5.0
    for k in range(n_groups):
        for c in range(n_sec):
            qb.add_linear(vidx(k, c), -w_stiff * SE[k, c] / SEmax)
    all_idx = [vidx(k, c) for k in range(n_groups) for c in range(n_sec)]
    all_coef = [mass_coef[k, c] for k in range(n_groups) for c in range(n_sec)]
    qb.add_target_penalty(all_idx, all_coef, mass_target, w_mass / max(mass_target**2, 1e-9))
    w_onehot = 3.0 * (w_stiff + w_mass)
    for k in range(n_groups):
        qb.add_onehot_penalty([vidx(k, c) for c in range(n_sec)], w_onehot)
    return qb.Q, n_groups, n_sec


def fig_encoding_qubo():
    Q, n_groups, n_sec = _build_real_f1_qubo()
    n_bits = n_groups * n_sec

    fig = plt.figure(figsize=(185 * MM, 98 * MM))
    gsfig = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.38)

    ax1 = fig.add_subplot(gsfig[0, 0])
    _draw_onehot_panel(ax1)

    ax2 = fig.add_subplot(gsfig[0, 1])
    absQ = np.abs(Q)
    vmax = np.percentile(absQ, 99.5)
    # PowerNorm (gamma<1) instead of a plain linear scale: the one-hot
    # diagonal terms are ~40x larger than the cross-group mass-target
    # coupling, so a linear map renders the (real, nonzero) coupling as
    # pure white -- indistinguishable from an actual zero, which left the
    # "mass-target coupling" callout pointing at what looked like empty
    # cells. The power-law compression makes that faint texture visible as
    # light gray while the diagonal blocks stay visually dominant (black).
    from matplotlib.colors import PowerNorm
    im = ax2.imshow(absQ, cmap='Greys', norm=PowerNorm(gamma=0.35, vmin=0, vmax=vmax))
    for b in range(0, n_bits + 1, n_sec):
        ax2.axhline(b - 0.5, color='0.55', lw=0.35)
        ax2.axvline(b - 0.5, color='0.55', lw=0.35)
    ax2.add_patch(mpatches.Rectangle((-0.5, -0.5), n_sec, n_sec, fill=False, ec='k', lw=1.2))
    ax2.set_xticks([]); ax2.set_yticks([])
    ax2.set_title(f"(b) $|Q|$ -- one real CS2 F1 QUBO iteration\n"
                  f"({n_groups} groups $\\times$ {n_sec} sections = {n_bits} bits, seed 0)", fontsize=8)
    # Extra top/bottom margin so BOTH callouts sit entirely OUTSIDE the
    # matrix data, one in the (small) top-left corner and one in the
    # (larger) bottom-left corner -- i.e. stacked on the SAME side but far
    # apart vertically, well clear of the title above and of each other.
    # y is inverted (imshow row-index convention: row 0 at the top).
    ax2.set_ylim(n_bits + 15, -22)
    _lbl_bbox = dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.85)
    # (1) one-hot penalty block: label in the top-left corner, well below
    # the title (small top margin keeps it clear of "set_title"'s text,
    # which lives in figure/axes space ABOVE the y=-0.5 data edge). Arrow
    # ends exactly on the group-0/group-1 boundary at the block's
    # mid-height -- i.e. right on the edge of the one-hot block it names.
    ax2.annotate("one-hot penalty\nblock (group 0)", xy=(n_sec - 0.5, n_sec / 2 - 0.5), xycoords='data',
                 xytext=(n_sec + 6, -11), textcoords='data', fontsize=6.2, ha='left', va='center',
                 bbox=_lbl_bbox, zorder=9,
                 arrowprops=dict(arrowstyle='->', lw=0.7, color='k', shrinkA=2, shrinkB=2))
    # (2) mass-target coupling: label in the BOTTOM-left corner (opposite
    # corner from callout (1), so the two never come close), short arrow
    # pointing up into the bottom-left cell (row=n_bits-3, col=2) -- group
    # (last group, row) x group 0 (col), i.e. a genuine cross-group,
    # off-diagonal cell, far from any diagonal one-hot block -- and
    # carries a real, nonzero (weak) mass-coupling value (visible under
    # the PowerNorm scale used here; |Q|=0.011, same magnitude as its
    # symmetric partner cell used for the (row=2, col=n_bits-3) check).
    ax2.annotate("mass-target coupling\n(weak, cross-group)",
                 xy=(2, n_bits - 3), xycoords='data',
                 xytext=(n_sec + 6, n_bits + 11), textcoords='data', fontsize=6.2, ha='left', va='center',
                 bbox=_lbl_bbox, zorder=9,
                 arrowprops=dict(arrowstyle='->', lw=0.7, color='k', shrinkA=2, shrinkB=2))
    cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label("$|Q_{ij}|$")
    savefig(fig, "fig_encoding_qubo")


# --------------------------------------------------------------------------
# fig_connection_detail (NEW, B/W) -- close-up of ONE fan from the real
# generator (3-4 beams + connection dots) with a 2D inset schematic of the
# eccentric-bearing idealisation (two nodes offset e, short stiff link).
# --------------------------------------------------------------------------
def fig_connection_detail():
    from reciprocal_geom import fan_unit, physical_beams, true_node_positions

    model = fan_unit(4, R=3.0, lam=0.28, ecc_ratio=0.06, single_group=False)
    nodes = np.array(model.nodes)
    tpos = true_node_positions(model)
    sup = nodes[list(model.support_idx)]

    fig = plt.figure(figsize=(150 * MM, 50 * MM))
    ax = fig.add_axes([0.0, 0.05, 0.52, 0.88], projection='3d')
    ax.view_init(elev=26, azim=-52)
    # 3D axes leave a large unused margin around the actual data by default
    # (matplotlib does not tight-bbox-crop Axes3D); zoom compensates so the
    # fan fills its allotted panel instead of floating in white space.
    ax.set_box_aspect([1, 1, 0.22], zoom=2.15)
    # each PHYSICAL beam drawn as ONE straight rod (true, non-eccentric
    # support->tip axis) -- not the FE support->interior->tip polyline,
    # which kinks at the interior node's +-ecc/2 offset.
    for bm in physical_beams(model):
        p1, p2 = tpos[bm['support_idx']], tpos[bm['tip_idx']]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='k', lw=2.0, zorder=3)
    mids = _link_midpoints(model, nodes)
    ax.scatter(mids[:, 0], mids[:, 1], mids[:, 2], color='k', marker='o',
               s=34, zorder=6, depthshade=False)
    ax.scatter(sup[:, 0], sup[:, 1], sup[:, 2], color='white', edgecolor='k', marker='^',
               s=70, zorder=6, depthshade=False, linewidth=1.2)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.grid(False)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_alpha(0.0)
        pane.line.set_alpha(0.0)
    fig.suptitle("Connection detail: one 4-beam fan (real generator output)\n"
                 "black lines: beams; dots: beam-to-beam connections; triangles: supports",
                 fontsize=7.6, y=1.01)

    axin = fig.add_axes([0.60, 0.14, 0.37, 0.55])
    axin.plot([-1.1, 0.0], [0.28, 0.28], color='k', lw=1.6, solid_capstyle='round')
    axin.plot([0.0, 1.1], [-0.28, -0.28], color='0.4', lw=1.6, ls='--', solid_capstyle='round')
    axin.plot([0.0, 0.0], [0.28, -0.28], color='k', lw=1.0)
    axin.scatter([0.0, 0.0], [0.28, -0.28], color='k', s=18, zorder=5)
    axin.annotate("", xy=(0.55, 0.28), xytext=(0.55, -0.28),
                  arrowprops=dict(arrowstyle='<->', lw=0.7, color='k'))
    axin.text(0.68, 0.0, "$e$", fontsize=7, va='center')
    axin.text(-1.05, 0.42, "beam $k$ tip node", fontsize=5.3)
    axin.text(-0.15, -0.52, "beam $k{+}1$ interior node", fontsize=5.3)
    axin.text(0.07, 0.0, "link", fontsize=5.3, rotation=90, va='center', ha='left')
    axin.set_xlim(-1.35, 1.35); axin.set_ylim(-0.78, 0.62)
    axin.set_xticks([]); axin.set_yticks([])
    for spine in axin.spines.values():
        spine.set_visible(False)
    axin.set_title("eccentric bearing idealisation", fontsize=6.3)

    savefig(fig, "fig_connection_detail")


if __name__ == "__main__":
    import sys as _sys
    if len(_sys.argv) > 1 and _sys.argv[1] == "study":
        fig_cs5_pareto()
        fig_ablation()
    elif len(_sys.argv) > 1 and _sys.argv[1] == "all_typologies":
        fig_cs1_design()
        fig_cs3_design()
        fig_cs4_design()
    else:
        fig_typologies()
        fig_cs1_validation()
        fig_convergence("cs2_grillage.json", "CS2 square grillage sizing convergence",
                         "fig_cs2_convergence")
        fig_convergence("cs3_hex.json", "CS3 hex grillage sizing convergence",
                         "fig_cs3_convergence")
        fig_cs2_design()
        fig_cs1_design()
        fig_cs3_design()
        fig_cs4_design()
        fig_cs4_fmqa()
        fig_cs5_pareto()
        fig_sensitivity()
        fig_ablation()
        fig_fan_geometry()
        fig_encoding_qubo()
        fig_connection_detail()

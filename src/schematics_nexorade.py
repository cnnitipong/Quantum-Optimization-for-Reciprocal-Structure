import os as _os_bootstrap
ROOT = _os_bootstrap.path.dirname(_os_bootstrap.path.dirname(_os_bootstrap.path.abspath(__file__)))
"""
schematics_nexorade.py -- publication figures (PDF+PNG, 300dpi, serif fonts) for
the v2 (literature-faithful nexorade) geometries and results, built from
results/*.json and src/reciprocal_geom_v2.py (square_nexorade,
hex_nexorade, rf_dome, davinci_bridge; fan_unit unchanged). Mirrors the
style established in schematics_fan.py: serif fonts, short in-figure text,
black/white for pure data figures, color reserved for typology overviews,
design (before/after sizing) maps, and the fig_encoding_qubo |Q|-matrix
heatmap panel (viridis; REVISION 2026, was grayscale in the prior
study), beam-to-beam connections drawn as single filled dots (never as
separate "link" line segments), pinned supports as open/filled triangles.

Every v2 geometry (square_nexorade/hex_nexorade/rf_dome/davinci_bridge) is
a "per-hub-fan + member fusion" construction: interior tessellation edges
are ONE physical member shared between the two fans on either side, and
supports occur ONLY at the assembly boundary. Because a fused interior
member's node chain is 2-3 FE sub-elements long (not always exactly 2, as
v1's physical_beams() assumed), every plotting routine below uses
reciprocal_geom_v2.physical_members_v2() (chain-based, arbitrary length)
+ reciprocal_geom.true_node_positions() (collapses every eccentric
link-node pair back to its true bearing point) to draw each physical
member as ONE straight line from its first to its last chain node -- this
works uniformly for fan_unit, square_nexorade, hex_nexorade, rf_dome AND
davinci_bridge (whose longitudinal/rung members are single-sub-element
chains), so a single helper (_member_lines) replaces v1's physical_beams()
call everywhere.

Design-map figures (fig_cs2_design, fig_cs3_design, fig_cs4_design,
fig_cs6_design) decode the actual best design vector achieved by the
stored best-compliance SA-QUBO/FMQA run by calling the SAME recovery
functions already written and reproducibility-verified in r6_uls_v2.py
(recover_cs2/recover_cs3/recover_cs4/recover_cs6 -- each asserts/records
`reproducibility_match` against the stored best_compliance/final_best in
results/r6_uls_v2.json before this script trusts it), rather than
re-deriving that logic a second time.
"""
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
from mpl_toolkits.mplot3d import Axes3D  # noqa

RESULTS = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
CODE = os.path.join(ROOT, "src")
sys.path.insert(0, CODE)
os.makedirs(FIG, exist_ok=True)

from reciprocal_geom import true_node_positions  # noqa: E402

MM = 1 / 25.4
COLORS = {  # colorblind-safe (Okabe-Ito) -- used ONLY by the COLOR figures
    # (fig_typologies, fig_cs2_design, fig_cs3_design, fig_cs4_design,
    # fig_cs6_design, fig_cs1_design)
    'sa': '#0072B2', 'ga': '#E69F00', 'random': '#009E73',
    'fmqa': '#0072B2', 'nsga': '#E69F00', 'black': '#000000',
    'accent': '#D55E00', 'grey': '#999999',
}

# --------------------------------------------------------------------------
# DISCRETE per-catalog-section diameter palette (author feedback, 2026:
# "can't see difference of each member size" -- under the earlier CONTINUOUS
# viridis gradient, the three largest adjacent CHS catalog sizes (168.3 /
# 193.7 / 244.5 mm) mapped to nearly-identical colors near the bright end of
# the ramp). Every design-map figure (fig_cs1_design, fig_cs2_design,
# fig_cs3_design, fig_cs4_design, fig_cs6_design) now colors each member by
# ONE of exactly 8 hand-separated, print-friendly colors -- one per CHS
# catalog section (sections.CHS_CATALOG, ascending D) -- instead of a
# gradient position. Colors are 'turbo' sampled at 8 equally spaced points
# (0, 1/7, ..., 1): this still reads as an ordered dark-blue -> cyan ->
# green -> yellow -> orange -> dark-red ramp, but with every adjacent pair
# (including the previously-confusable 168.3/193.7/244.5 trio) obviously
# distinct at a glance, in print, and to color-vision-deficient readers
# (checked: consecutive entries differ in both hue and lightness, not hue
# alone). DIAMETER_COLORS[i] corresponds to sections.CHS_CATALOG[i].
DIAMETER_COLORS = [
    '#30123B',  # 60.3 mm  -- dark blue/violet
    '#4675ED',  # 76.1 mm  -- blue
    '#1BCFD4',  # 88.9 mm  -- cyan
    '#61FC6C',  # 114.3 mm -- green
    '#D1E834',  # 139.7 mm -- yellow
    '#FE9B2D',  # 168.3 mm -- orange
    '#D93806',  # 193.7 mm -- red-orange
    '#7A0402',  # 244.5 mm -- dark red
]
_DIAMETER_CMAP = ListedColormap(DIAMETER_COLORS)

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


def _clamp_log_ylim(ax, mean_curves, lo_pad=0.72, hi_pad=1.45):
    """On a log y-axis a +-1sd band whose lower edge goes negative is clipped
    by matplotlib at an arbitrarily small positive value, which stretches the
    axis over one or two entirely empty decades.  Re-clamp the view to the
    span of the PLOTTED MEAN curves (padded), so the converged region -- the
    part a reader actually compares -- fills the panel.  Purely a view-limit
    change: no data is altered, hidden or rescaled."""
    finite = [np.asarray(m)[np.isfinite(m)] for m in mean_curves]
    finite = [m[m > 0] for m in finite if len(m)]
    if not finite:
        return
    lo = min(float(m.min()) for m in finite)
    hi = max(float(m.max()) for m in finite)
    ax.set_ylim(lo * lo_pad, hi * hi_pad)


def _runs(method_data):
    """Normalize a results methods[...] entry, which may be stored as a
    plain list (cs2_v2/cs6_v2/r3_ablation_v2/penalty_sensitivity_v2 style)
    OR as a dict keyed by seed-string (cs3_v2/cs4_v2 style), into a plain
    list of per-seed run dicts."""
    if isinstance(method_data, dict):
        return list(method_data.values())
    return method_data


def _link_midpoints(model, nodes):
    """(N,3) array of beam-to-beam connection points: midpoint of every
    'link' element (the short eccentric-bearing connector at each
    engagement point). Works identically for v1 and v2 models (the 'link'
    convention is unchanged)."""
    mids = [(nodes[e['i']] + nodes[e['j']]) / 2.0
            for e in model.elements if e['kind'] == 'link']
    return np.array(mids) if mids else np.zeros((0, 3))


def _member_lines(model):
    """Every PHYSICAL member of a v2 model, drawn as a sequence of
    consecutive STRAIGHT, exactly-collinear segments along the ONE true
    straight axis between that member's two true outer endpoints -- each
    segment colored by its own RAW sub-element's own 'group', not collapsed
    to one group per fused physical member.

    REVISION (author feedback, 2026: "Fig 16 fig_cs2_design beams must be
    straight lines, like the fans") -- the prior per-raw-element-coloring
    fix (see history below) drew each raw beam element as its OWN line
    between its OWN true endpoints (tpos[e['i']], tpos[e['j']]). That is
    correct for boundary members (2 sub-elements, exactly collinear by
    construction) but for INTERIOR fused members (3-4 sub-elements: e.g.
    tip_A-lower_A-lower_B-tip_B) the intermediate bearing points
    (lower_A/lower_B) are only APPROXIMATELY on the tip_A-tip_B line --
    each is offset by the small eccentricity ecc used to build its own
    fan's engagement geometry, and fan A's/fan B's eccentric offsets do not
    generally cancel exactly. Plotting each raw sub-element as its own
    literal chord therefore reintroduced a small but visible KINK at every
    such intermediate point -- the opposite of the fan figures (fig_cs1_design
    etc.), whose members are always single straight chords.

    Fixed by treating "straight" and "colored by raw sub-element" as two
    SEPARATE concerns: for each physical member (reciprocal_geom_v2.
    physical_members_v2(), chain-based, arbitrary length), the true straight
    axis is fixed once as the line from true_pos[chain[0]] to
    true_pos[chain[-1]] (the member's genuine two outer endpoints -- always
    exactly on-axis by definition). Every INTERMEDIATE chain node is then
    PROJECTED onto that single straight line (orthogonal projection, not
    just re-used as-is), so every drawn sub-segment endpoint -- outer or
    intermediate -- lies exactly on one straight axis: consecutive
    sub-segments are geometrically collinear to floating-point precision,
    with NO kink, while each sub-segment is still colored by its own raw
    element's own true `group` (so fan A's/fan B's independently-optimized
    diameters both remain visible, exactly as the prior color fix
    intended). Boundary members (2 sub-elements) and fan_unit/
    davinci_bridge_complex members (already single-sub-element, no fusion)
    are geometrically unaffected (already exactly collinear / a single
    segment) -- this is a no-op there, so every single-color panel
    (fig_typologies, fig_rotation_method, fig_cs1_design, fig_cs4_design)
    renders identically to before. Returns a FLAT list of dicts (one per
    drawn straight sub-segment, several per fused physical member): p1, p2
    (3,) arrays -- both exactly on the parent member's single straight
    axis -- group (int, the owning raw sub-element's own group), is_boundary
    (bool, inherited from the parent physical member).

    History: this function used to call physical_members_v2() and draw ONE
    line per fused PHYSICAL member colored by that helper's single `group`
    field (segs[0]'s group only), which silently discarded fan B's group
    from every 2-groups-per-member's rendered color (verified: 66% of
    CS2-v2's members, 66/87 of CS3-v2's) -- fixed by keying color off each
    raw element's own group (see above), which is what this straight-line
    study preserves while removing the kink that per-raw-element-as-its-
    own-chord approach had reintroduced.
    """
    from reciprocal_geom_v2 import physical_members_v2
    tpos = true_node_positions(model)
    members = physical_members_v2(model)

    # (member_id, frozenset({i, j})) -> raw sub-element's own group, read
    # directly off model.elements (NOT physical_members_v2's single
    # per-member `group` field -- see docstring history above).
    edge_group = {}
    for e in model.elements:
        if e['kind'] != 'beam':
            continue
        edge_group[(e['member_id'], frozenset((e['i'], e['j'])))] = e['group']

    out = []
    for m in members:
        chain = m['chain']
        p1 = tpos[chain[0]]
        p2 = tpos[chain[-1]]
        d = p2 - p1
        dd = float(np.dot(d, d))
        # exact endpoints, orthogonal projection for every intermediate
        # chain node -> every point below lies exactly on the p1-p2 axis.
        pts = [p1]
        for node in chain[1:-1]:
            if dd < 1e-12:
                pts.append(p1)
            else:
                t = float(np.dot(tpos[node] - p1, d)) / dd
                pts.append(p1 + t * d)
        pts.append(p2)
        for k in range(len(chain) - 1):
            g = edge_group[(m['member_id'], frozenset((chain[k], chain[k + 1])))]
            out.append(dict(p1=pts[k], p2=pts[k + 1], group=g,
                             is_boundary=m['is_boundary']))
    return out


def _member_lines_by_member(model):
    """Every PHYSICAL member of `model`, drawn as exactly ONE straight line
    segment (the member's own two true outer endpoints) and colored
    UNIFORMLY by its DOMINANT-length raw sub-element group -- i.e. the
    SAME per-member classification _member_size_counts()/
    _size_count_legend() already use for the in-figure "Members per size"
    legend (ties broken by the lower catalog index, for determinism,
    identically to _member_size_counts).

    REVISION (author feedback, 2026: "member size mentioned is not the
    same as the color displayed") -- _member_lines() colors each drawn
    sub-segment of a fused interior member by that sub-segment's OWN raw
    group, so a fused member whose two fan-side portions land in different
    sizing groups is drawn with TWO colors while the legend (which counts
    each PHYSICAL member once, credited to its dominant/majority-length
    group) only ever lists ONE of those two colors for it -- the minority
    portion's color is then visible in the drawing but absent from the
    legend. That two-concerns split (straight-line geometry vs. per-raw-
    element coloring) was intentional for the single-global-color panels
    (fig_cs1_design/fig_cs4_design/fig_vault_geo_design, where `group` is
    never used to pick the color at all -- every member in the panel gets
    the same color regardless), but for any panel that DOES key color off
    a per-group idx_vec (fig_cs2_design, fig_cs3_design, fig_cs6_design,
    fig_dome_sizing_design, fig_vault_sizing_design), it silently broke the
    drawing/legend correspondence. This helper is the fix for those five
    figures: each physical member is now drawn as one uniformly-colored
    straight line (still exactly on its own true p1-p2 axis, so the
    "beams must be straight lines" fix in _member_lines() is unaffected),
    classified by the identical dominant-length rule the legend uses --
    so the set of colors visible in the drawing and the set of sizes
    listed in the legend agree EXACTLY, by construction.

    Returns a list of dicts (one per physical member): p1, p2 (3,) arrays
    (the member's own two true endpoints), group (int, the DOMINANT raw
    sub-element group -- the same one _member_size_counts credits this
    member to), is_boundary (bool)."""
    from reciprocal_geom_v2 import physical_members_v2
    tpos = true_node_positions(model)
    members = physical_members_v2(model)

    edge_group = {}
    for e in model.elements:
        if e['kind'] != 'beam':
            continue
        edge_group[(e['member_id'], frozenset((e['i'], e['j'])))] = e['group']

    out = []
    for m in members:
        chain = m['chain']
        p1 = tpos[chain[0]]
        p2 = tpos[chain[-1]]
        length_by_group = {}
        for k in range(len(chain) - 1):
            i, j = chain[k], chain[k + 1]
            g = edge_group[(m['member_id'], frozenset((i, j)))]
            L = float(np.linalg.norm(tpos[j] - tpos[i]))
            length_by_group[g] = length_by_group.get(g, 0.0) + L
        dominant_group = max(length_by_group, key=lambda g: (length_by_group[g], -g))
        out.append(dict(p1=p1, p2=p2, group=dominant_group,
                         is_boundary=m['is_boundary']))
    return out


def _mean_std_bands(histories):
    n = min(len(h) for h in histories)
    arr = np.array([h[:n] for h in histories])
    return arr.mean(axis=0), arr.std(axis=0), n


# --------------------------------------------------------------------------
# Shared helpers for the "add numbers, not just color, for outside
# diameter" study (2026): every design-map figure (fig_cs1_design,
# fig_cs2_design, fig_cs3_design, fig_cs4_design, fig_cs6_design) uses
# _diameter_colorbar() so the shared colorbar carries DISCRETE ticks at
# exactly the 8 catalog D values (never a generic continuous 25-mm-step
# axis). REVISION (author feedback, 2026: "can't see difference of each
# member size") -- the colorbar itself is now DISCRETE too: 8 solid blocks
# (one per DIAMETER_COLORS / CHS_CATALOG entry, via ListedColormap +
# BoundaryNorm) rather than a continuous gradient, each block labeled with
# its own real catalog D value, so the legend matches exactly how members
# are colored (one flat color per section, not a position on a ramp).
# _callout() is a small offset-label-with-leader helper (screen-point
# offset, so it is immune to each panel's data aspect/scale) used to place
# in-figure numeric D labels on/near individual members in 2D panels
# without them colliding with the drawn geometry. REVISION (2026, CS6 3D
# conversion): fig_cs6_design no longer calls it -- matplotlib's Axes3D has
# no equivalent screen-space offset+leader annotation (see fig_cs6_design's
# own docstring), so that figure now uses a text2D whole-structure member-
# count-per-size read-out instead (_size_count_legend; see that function's
# docstring for the current convention). Kept here (currently unused) since
# any future 2D per-member design map could still use it.
# --------------------------------------------------------------------------
def _diameter_colorbar(fig, axes, cmap, D_values, label, shrink=0.5, label_fontsize=None):
    n = len(D_values)
    norm = BoundaryNorm(np.arange(-0.5, n + 0.5, 1.0), n)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(np.arange(n))
    cbar = fig.colorbar(sm, ax=axes, orientation='horizontal', fraction=0.05,
                         pad=0.02, shrink=shrink, ticks=np.arange(n))
    cbar.ax.set_xticklabels([f"{d:g}" for d in D_values], rotation=40,
                             ha='right', fontsize=6)
    cbar.ax.tick_params(length=3, pad=1)
    if label_fontsize is not None:
        cbar.set_label(label, fontsize=label_fontsize)
    else:
        cbar.set_label(label)
    return cbar


def _callout(ax, xy, text, offset=(0, 12), fontsize=5.5, ha='center',
             va='center', color='k', arrow=True):
    """Small numeric label at data point `xy`, nudged by a fixed SCREEN
    offset (points, not data units) so it reads cleanly regardless of a
    panel's aspect ratio/zoom -- with a thin leader line back to `xy` when
    `arrow` is True so the reader can tell exactly which member/point the
    number belongs to."""
    ax.annotate(text, xy=xy, xycoords='data', xytext=offset,
                textcoords='offset points', fontsize=fontsize, ha=ha, va=va,
                color=color, zorder=20,
                arrowprops=(dict(arrowstyle='-', lw=0.5, color=color,
                                 shrinkA=0, shrinkB=2) if arrow else None))


def _n_physical_members(model):
    """Total count of PHYSICAL members (fused chains collapsed to one) in
    `model`, via reciprocal_geom_v2.physical_members_v2() -- the same
    physical-member notion _member_lines()/_member_size_counts() use, NOT a
    count of raw FE sub-elements or sizing groups."""
    from reciprocal_geom_v2 import physical_members_v2
    return len(physical_members_v2(model))


def _member_size_counts(model, idx_vec, CHS_CATALOG):
    """Count each PHYSICAL member of `model` exactly ONCE, classified by
    catalog size under section assignment idx_vec (sizing-group index ->
    CHS_CATALOG index).

    REVISION (author feedback, 2026: "don't show counts per group; show
    counts of ALL members in the structure per size") -- the prior
    "D per group: X mm x n_groups" legend counted SIZING GROUPS (an
    abstraction of the optimization formulation: how many of the model's
    n_groups decision variables landed on each catalog size), which is not
    the same number as "how many physical members are that size" whenever
    a group's member count varies (e.g. boundary vs interior groups spanning
    different member counts) -- and for a FUSED interior member whose two
    raw sub-chains sit in two DIFFERENT sizing groups (see _member_lines
    docstring), no single group-based count could represent that member at
    all. Convention adopted here (documented per the task's own
    instruction to pick one simple, and state it): each physical
    member is counted once, credited to whichever of its (one or two)
    raw sub-element groups accounts for the GREATER total built length
    along the member's own straight axis -- i.e. its DOMINANT/majority-
    length size (ties broken by the lower catalog index, for
    determinism). This is stated explicitly in the manuscript caption note
    that used to describe the old "D per group" convention.

    Returns a collections.Counter keyed by CHS_CATALOG section index ->
    physical-member count (sums to _n_physical_members(model))."""
    from collections import Counter
    from reciprocal_geom_v2 import physical_members_v2
    tpos = true_node_positions(model)
    edge_group = {}
    for e in model.elements:
        if e['kind'] != 'beam':
            continue
        edge_group[(e['member_id'], frozenset((e['i'], e['j'])))] = e['group']

    counts = Counter()
    for m in physical_members_v2(model):
        chain = m['chain']
        length_by_group = {}
        for k in range(len(chain) - 1):
            i, j = chain[k], chain[k + 1]
            g = edge_group[(m['member_id'], frozenset((i, j)))]
            L = float(np.linalg.norm(tpos[j] - tpos[i]))
            length_by_group[g] = length_by_group.get(g, 0.0) + L
        dominant_group = max(length_by_group, key=lambda g: (length_by_group[g], -g))
        counts[idx_vec[dominant_group]] += 1
    return counts


def _size_count_legend(model, idx_vec, CHS_CATALOG):
    """Compact 'Members per size: D mm x count, ... (N total)' legend text
    -- counts of PHYSICAL MEMBERS across the whole structure per catalog
    size (see _member_size_counts for the dominant-length-portion fused-
    member convention), NOT a count of sizing groups (the prior
    "D per group" convention -- see _member_size_counts docstring for why
    that changed)."""
    counts = _member_size_counts(model, idx_vec, CHS_CATALOG)
    total = sum(counts.values())
    parts = []
    for sec_idx in sorted(counts, key=lambda i: CHS_CATALOG[i]['D_mm']):
        D = CHS_CATALOG[sec_idx]['D_mm']
        parts.append(f"{D:g}\u2009mm\u00d7{counts[sec_idx]}")
    return "Members per size:  " + ",  ".join(parts) + f"  ({total} total)"


# --------------------------------------------------------------------------
# fig_typologies (COLOR) -- FIVE v2 typologies: (a) 4-beam fan [unchanged
# fan_unit, 3D], (b) rotated-square nexorade, (c) hexagonal nexorade [PLAN
# view -- both are exactly planar (z=0) constructions, so a true 2D plot is
# used rather than a 3D view faked flat; engagement windows/pinwheel
# pattern read exactly as in the nexorade literature this way], (d)
# reciprocal dome, (e) Da Vinci bridge [3D -- both have genuine
# out-of-plane/lateral extent]. REVISION (author feedback, 2026): the
# 3-beam fan panel is REMOVED from the manuscript typology overview (the
# author wants the 3-beam fan unit out of the manuscript entirely, not just
# de-emphasized; its data/analysis stay in results/cs1_v2.json and the
# reproducibility package, only this figure and fig_cs1_design/
# fig_cs1_validation change) -- the remaining 5 panels are re-lettered
# (a)-(e) and laid out as a balanced 3-top/2-bottom grid (2x6 GridSpec,
# each top panel spanning 2 of 6 columns, each bottom panel spanning 2 of 6
# columns but offset by 1 so the pair is centered under the 3 panels
# above) rather than the previous 2x3 grid. Small illustrative instances
# (not the full case-study sizes) are used purely so the assembled
# pinwheel pattern is legible at this scale for (a)-(c); supports
# (triangles) are drawn at every node in model.support_idx, which by
# construction (see reciprocal_geom_v2 module docstring) is
# perimeter/abutment-only for (b)-(e).
#
# REVISION (complex dome/bridge upgrade, 2026): panels (d)/(f) now show the
# ACTUAL case-study geometries -- rf_dome_complex (rings=4, span=18.5m,
# rise=3.15m; 5 concentric ring families, 92 face-hub fans, 152 physical
# members) and davinci_bridge_complex (n_bays=12, span=20m, width=6m; 3
# parallel rails + 2 rung segments/station, 58 physical members) -- rather
# than the earlier small illustrative rf_dome(2,...)/davinci_bridge(6,...)
# instances, because the whole point of this typology figure's 3D panels is
# to show what these typologies actually look like, and the denser
# CS4/CS6-complex geometries are what the paper's dome/bridge case studies
# now use: (d) reads as a genuinely dense multi-ring lattice (not a single
# shallow ring of fans), (f) clearly shows 3 longitudinal rails woven with
# alternating-parity transverse rungs rather than 2.
#
# REVISION (vault typology, 2026): a SIXTH panel, (e) reciprocal vault
# (rf_vault, the new woven-barrel-vault typology backing CS-VAULT-GEO/
# dome_sizing/vault_sizing), is added between the dome and the bridge, using
# the SAME actual case-study geometry the vault campaigns use (nx=10, ny=8,
# cell_x=2.0m, span_transverse=14.0m, rise=2.6m, default lam_bands -- 142
# physical members) rather than a shrunk illustrative instance, matching the
# "show the real case-study geometry" convention already used for (d)/(f).
# The 5-panel 3-top/2-bottom layout is replaced by a balanced 2x6 GridSpec
# with THREE panels per row (each spanning 2 of 6 columns, no more
# centering offset needed since 3+3 exactly tiles 6 columns): (a)-(c) top,
# (d)-(f) bottom. View angle for (e): elev=28, azim=-40, zaspect=0.35.
#
# REVISION (author feedback, 2026: "vault figures can't be seen well" --
# barrel-vault arch AND woven pattern must both be legible) -- the earlier
# near-front-on view (elev=17, azim=-15, zaspect=0.20) was chosen only to
# keep the transverse arc from flattening, but at that shallow elevation
# and near-zero azimuth the diagonal pinwheel weave (clearly visible in
# panel (b)'s plan view) is seen almost perfectly edge-on: consecutive
# zigzag members overlap into what reads as plain parallel "ribs", the
# defining rotated-square weave pattern essentially invisible. Re-rendered
# and visually compared a systematic elev in {20,22,25,26,28,30,32,33,35}
# x azim in {-15,-20,-30,-35,-40,-45,-50,-55,-60,-70} x zaspect in
# {0.20,0.30,0.35,0.40,0.45,0.50,0.55} grid (see
# _scratch_qa/view_test_vault.py/vault_view_grid*.png): elev<25 keeps the
# weave too edge-on regardless of azim; azim closer to 0 than -30 does the
# same; zaspect<0.30 visually flattens the arch back toward a tilted
# rectangle. elev=28, azim=-40, zaspect=0.35 (within the elev 25-35 /
# azim -35..-60 / moderate-zaspect family) is the first combination where
# BOTH the transverse arch curvature AND the diagonal weave (individual
# zigzag members + their connection dots) read clearly at once, without
# adjacent members visually merging into a mush -- applied consistently to
# every vault design-map figure in this module (see _VAULT_VIEW below).
#
# REVISION (hex-pattern vault upgrade, 2026): panel (e) now shows
# `rf_vault_hex` (the HEXAGONAL/TRIANGULAR face-hub barrel-vault pattern --
# see that generator's own docstring in reciprocal_geom_v2.py) INSTEAD OF
# the earlier `rf_vault` rotated-SQUARE vertex-hub pattern, using its own
# real case-study geometry (n_rows=7, length=17.0m, span_transverse=14.0m,
# rise=2.6m -- same span_transverse/rise as `rf_vault`, matching the
# "show the real case-study geometry" convention already used for (d)/(f)/
# the old (e)). The SAME _VAULT_VIEW (elev=28, azim=-40, zaspect=0.35) is
# reused unchanged: re-rendered and visually compared against the same
# elev/azim/zaspect family documented above, and at this member density
# (147 physical members vs `rf_vault`'s 142 -- comparable) it already keeps
# both the transverse arch curvature AND the true hexagonal engagement
# windows (six alternating up/down triangular fans per interior vertex, vs
# the square pattern's four-per-vertex windows) legible at once, so no
# further search was needed. `rf_vault` itself is UNCHANGED and still used
# by nothing in this figure; its own data/figures live on unmodified in
# vault_posthoc.json/vault_geo.json/vault_sizing.json for any figure that
# still needs the square pattern for comparison.
# --------------------------------------------------------------------------
def _draw_3d_model(ax, model, elev, azim, zaspect, lw=1.3, dot_s=10, sup_s=16):
    for m in _member_lines(model):
        ax.plot([m['p1'][0], m['p2'][0]], [m['p1'][1], m['p2'][1]],
                [m['p1'][2], m['p2'][2]], color=COLORS['sa'], lw=lw, zorder=3)
    nodes = np.array(model.nodes)
    mids = _link_midpoints(model, nodes)
    if len(mids):
        ax.scatter(mids[:, 0], mids[:, 1], mids[:, 2], color=COLORS['accent'],
                   marker='o', s=dot_s, zorder=6, depthshade=False, linewidth=0)
    tpos = true_node_positions(model)
    sup = tpos[list(model.support_idx)]
    ax.scatter(sup[:, 0], sup[:, 1], sup[:, 2], color='k', marker='^',
               s=sup_s, zorder=10, depthshade=False)
    ax.set_box_aspect([1, 1, zaspect])
    ax.view_init(elev=elev, azim=azim)
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
    ax.grid(False)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_alpha(0.0)


def _draw_plan_model(ax, model, lw=1.1, dot_s=9, sup_s=15):
    for m in _member_lines(model):
        ax.plot([m['p1'][0], m['p2'][0]], [m['p1'][1], m['p2'][1]],
                color=COLORS['sa'], lw=lw, zorder=3)
    nodes = np.array(model.nodes)
    mids = _link_midpoints(model, nodes)
    if len(mids):
        ax.scatter(mids[:, 0], mids[:, 1], color=COLORS['accent'], marker='o',
                   s=dot_s, zorder=6, linewidth=0)
    tpos = true_node_positions(model)
    sup = tpos[list(model.support_idx)]
    ax.scatter(sup[:, 0], sup[:, 1], color='k', marker='^', s=sup_s, zorder=10)
    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def fig_typologies():
    from reciprocal_geom_v2 import (fan_unit, square_nexorade, hex_nexorade,
                                     rf_dome2, rf_vault_hex2, davinci_bridge2,
                                     physical_members_v2)

    fig = plt.figure(figsize=(190 * MM, 128 * MM))
    # 2x6 grid: 3 panels per row, each spanning 2 of 6 columns (0:2, 2:4,
    # 4:6) -- a balanced 3-top/3-bottom layout for the 6 typologies (3-beam
    # fan panel removed in an earlier study, vault panel (e) added here).
    gs = fig.add_gridspec(2, 6, wspace=0.12, hspace=0.20)

    ax = fig.add_subplot(gs[0, 0:2], projection='3d')
    _draw_3d_model(ax, fan_unit(4, R=3.0, lam=0.25), 22, -60, 0.16)
    ax.set_title("(a) 4-beam fan unit", pad=-4, fontsize=8.5)

    ax = fig.add_subplot(gs[0, 2:4])
    _draw_plan_model(ax, square_nexorade(4, 4, cell=2.5, lam=0.25))
    ax.set_title("(b) Rotated-square nexorade (plan)", fontsize=8.5)

    ax = fig.add_subplot(gs[0, 4:6])
    _draw_plan_model(ax, hex_nexorade(rings=2, edge=2.2, lam=0.25), lw=0.95, dot_s=7, sup_s=13)
    ax.set_title("(c) Hexagonal nexorade (plan)", fontsize=8.5)

    ax = fig.add_subplot(gs[1, 0:2], projection='3d')
    # REVISION (dome redesign, 2026): rf_dome_complex -> rf_dome2, the
    # Kiewitt polar-lattice reciprocal dome whose 28 pinned supports lie on a
    # TRUE CIRCLE of radius span/2 = 9.25 m (the old frame's supports were the
    # ragged boundary of a lattice patch); 182 physical members / 112 fans /
    # 12 section groups. See results/dome2_verification.json.
    _draw_3d_model(ax, rf_dome2(n_sect=7, n_ring=4, span=18.5, rise=3.15, lam=0.25,
                                max_group_rings=4),
                   26, -60, 0.30, lw=0.55, dot_s=3, sup_s=9)
    ax.set_title("(d) Reciprocal dome (circular springing)", pad=-4, fontsize=8.5)

    ax = fig.add_subplot(gs[1, 2:4], projection='3d')
    # REVISION (vault redesign, 2026): rf_vault_hex -> rf_vault_hex2, the
    # complete-fans-only, mirror-symmetric hex vault with EQUAL pin counts on
    # both springing lines (210 physical members / 120 fans / 16 supports).
    _draw_3d_model(ax, rf_vault_hex2(n_rows=8, n_cols=8, span_transverse=14.0, rise=2.6),
                   28, -40, 0.35, lw=0.55, dot_s=3, sup_s=9)
    ax.set_title("(e) Reciprocal vault (hex weave)", pad=-4, fontsize=8.5)

    ax = fig.add_subplot(gs[1, 4:6], projection='3d')
    # REVISION (bridge redesign, 2026): davinci_bridge_complex ->
    # davinci_bridge2 (load-path-corrected 3-rail, 24-bay bridge; fixes the
    # strain-energy-floor defect diagnosed in results/bridge_diagnosis.json
    # -- 118 physical members / 16 groups, vs the old bridge's 58 members;
    # see reciprocal_geom_v2.davinci_bridge2 and
    # results/bridge2_verification.json). Same view convention (elev=20,
    # azim=-58, zaspect=0.20) as before -- already verified to keep the 3
    # rails visually separated and the arch profile legible.
    _bridge2_model = davinci_bridge2(n_bays=24, span=20.0, width=6.0, lam=0.25)
    _draw_3d_model(ax, _bridge2_model, 20, -58, 0.20, lw=0.9, dot_s=5, sup_s=11)
    ax.set_title("(f) Da Vinci bridge (redesigned, 3 rails)", pad=-4, fontsize=8.5)
    # Member-count legend (NEW, 2026 bridge redesign QA requirement):
    # longitudinal-rail vs rung-half-segment physical-member counts, so the
    # panel is self-contained about how many members the redesign contains.
    _bridge2_members = physical_members_v2(_bridge2_model)
    _n_long = sum(1 for m in _bridge2_members if m['member_id'][0] == 'long')
    _n_rung = sum(1 for m in _bridge2_members if m['member_id'][0] == 'rung')
    ax.text2D(0.5, 0.03,
              f"{len(_bridge2_members)} members: {_n_long} rails + {_n_rung} rungs "
              f"({_bridge2_model.n_groups} groups)",
              transform=ax.transAxes, fontsize=6.0, ha='center', va='bottom',
              bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.85))

    # NOTE: in-figure title/footer removed (2026 study) -- legend text
    # ("blue: physical members; dots: beam-to-beam connections; triangles:
    # pinned supports, perimeter-only for (b)-(f)") now given in the
    # manuscript caption instead (see latex/figure_title_migration.md).
    fig.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.02)
    savefig(fig, "fig_typologies")


# --------------------------------------------------------------------------
# fig_rotation_method (NEW, B/W) -- the square_nexorade "rotation method"
# construction step by step, using the REAL v2 hub-fan math: (a) a base
# square grid patch with one interior vertex (hub) highlighted, (b) that
# hub's 4 grid edges rotated by the engagement/rotation angle psi around
# the vertex (reciprocal_geom.pinwheel_fan called with the SAME 4
# grid-neighbour support points square_nexorade() itself builds for every
# interior hub) to form a mutually-supporting fan with its engagement
# window, (c) the assembled nexorade patch (plan) that results once every
# hub in a small grid is built and interior edges are member-fused
# (square_nexorade() itself, unmodified).
# --------------------------------------------------------------------------
def _grid_true_xy(fan):
    """Collapse a raw pinwheel_fan() dict's eccentric link-node pairs to
    their true bearing midpoint (same rule as reciprocal_geom.
    true_node_positions, applied here to a standalone fan dict rather than
    a full RFModel) and return (N,2) planar coordinates."""
    nodes = np.array(fan['nodes'])
    true_pos = nodes.copy()
    for (i, j, kind, bid) in fan['elements']:
        if kind == 'link':
            mid = 0.5 * (nodes[i] + nodes[j])
            true_pos[i] = mid
            true_pos[j] = mid
    return true_pos[:, :2]


def fig_rotation_method():
    from reciprocal_geom import pinwheel_fan
    from reciprocal_geom_v2 import _beam_pairs

    cell = 2.5
    lam = 0.25
    ecc_ratio = 0.07
    ecc = ecc_ratio * cell
    NX = NY = 4  # small patch: hubs at ix,iy in 1..NX-1 (interior vertices)

    def pos(ix, iy):
        return (ix * cell, iy * cell, 0.0)

    hub = (2, 2)  # a genuinely interior vertex of the NXxNY grid
    NEI_DIRS = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # E, N, W, S -- exactly
    # square_nexorade's own neighbour-direction order
    neigh = [(hub[0] + dx, hub[1] + dy) for dx, dy in NEI_DIRS]
    support_pts = [pos(*nk) for nk in neigh]
    hub_xy = np.array(pos(*hub)[:2])

    fan = pinwheel_fan(support_pts, lam, ecc, chirality=-1)
    txy = _grid_true_xy(fan)
    beam_pairs = _beam_pairs(fan)  # (support_local, lower_local, upper_local) x4
    R = cell  # all 4 neighbours are exactly `cell` from the hub on this grid
    r_e = lam * R
    psi = np.degrees(np.arcsin(np.clip(r_e / R, -1, 1)))

    fig, axes = plt.subplots(1, 3, figsize=(190 * MM, 68 * MM))
    bbox = dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.9)

    # ---- (a) base grid patch, one interior vertex highlighted -----------
    ax = axes[0]
    for ix in range(NX + 1):
        ax.plot([ix * cell] * (NY + 1), [j * cell for j in range(NY + 1)],
                color='0.65', lw=0.8, ls='-', zorder=1)
    for iy in range(NY + 1):
        ax.plot([i * cell for i in range(NX + 1)], [iy * cell] * (NX + 1),
                color='0.65', lw=0.8, ls='-', zorder=1)
    gx, gy = np.meshgrid(np.arange(NX + 1) * cell, np.arange(NY + 1) * cell)
    ax.scatter(gx.ravel(), gy.ravel(), color='0.55', s=9, zorder=2)
    ax.scatter([hub_xy[0]], [hub_xy[1]], color='k', marker='o', s=70, zorder=5,
               edgecolor='k', linewidth=1.0)
    ax.annotate("interior vertex\n(hub)", xy=(hub_xy[0], hub_xy[1]), xycoords='data',
                xytext=(hub_xy[0] + 0.55 * cell, hub_xy[1] + 0.95 * cell),
                textcoords='data', fontsize=6.8, ha='left', va='bottom', bbox=bbox,
                arrowprops=dict(arrowstyle='->', lw=0.8, color='k'))
    ax.set_title("(a) base square grid patch", fontsize=8.3)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # ---- (b) rotated members -> fan + engagement window ------------------
    # R (=cell) >> r_e (=lambda*cell), so the whole engagement cluster is a
    # small fraction of the panel: draw a clean OVERVIEW (grid reference,
    # beams, supports, rotation angle psi -- all far from the crowded
    # centre, so none of this needs a leader line crossing the fan) plus a
    # magnified INSET (mark_inset-style) containing ONLY the small central
    # engagement circle / connection dots / engagement-window label, each
    # with a short LOCAL leader inside that dedicated inset canvas -- same
    # two-scale convention used for the tight-rotor case in fig_fan_geometry.
    ax = axes[1]
    sup_pts = np.array(support_pts)[:, :2]
    for k in range(4):
        ax.plot([hub_xy[0], support_pts[k][0]], [hub_xy[1], support_pts[k][1]],
                color='0.72', lw=0.9, ls=':', zorder=1)
    for k, (s_local, l_local, u_local) in enumerate(beam_pairs):
        p_s, p_u = txy[s_local], txy[u_local]
        ax.plot([p_s[0], p_u[0]], [p_s[1], p_u[1]], color='k', lw=1.5,
                solid_capstyle='round', zorder=4)
    tip_pts = np.array([txy[u_local] for (_, _, u_local) in beam_pairs])
    ax.scatter(tip_pts[:, 0], tip_pts[:, 1], color='k', s=16, zorder=6)
    ax.scatter(sup_pts[:, 0], sup_pts[:, 1], marker='^', s=60, zorder=6,
               facecolor='white', edgecolor='k', linewidth=1.1)
    zoom_r = 2.6 * r_e
    zth = np.linspace(0, 2 * np.pi, 100)
    ax.plot(hub_xy[0] + zoom_r * np.cos(zth), hub_xy[1] + zoom_r * np.sin(zth),
            color='0.7', lw=0.6, ls=':', zorder=2)

    # rotation angle psi at support k=0 (E neighbour): angle between the
    # un-rotated radial spoke (support->hub) and the actual beam axis --
    # this angle is exact anywhere along the two lines, so it reads cleanly
    # out here near the (uncrowded) support even though the lines visually
    # coincide near the hub.
    s0 = sup_pts[0]
    v_rad = (hub_xy - s0); v_rad /= np.linalg.norm(v_rad)
    s_local0, _, u_local0 = beam_pairs[0]
    v_beam = txy[u_local0] - txy[s_local0]; v_beam /= np.linalg.norm(v_beam)
    ang_rad = np.degrees(np.arctan2(v_rad[1], v_rad[0]))
    ang_beam = np.degrees(np.arctan2(v_beam[1], v_beam[0]))
    dpsi = (ang_beam - ang_rad + 180) % 360 - 180
    ang_beam2 = ang_rad + dpsi
    r_arc = 0.34 * R
    arc = mpatches.Arc(s0, 2 * r_arc, 2 * r_arc, angle=0,
                        theta1=min(ang_rad, ang_beam2), theta2=max(ang_rad, ang_beam2),
                        color='k', lw=1.3, zorder=7)
    ax.add_patch(arc)
    # label placed PERPENDICULAR to the (radial-spoke, beam-axis) bisector
    # at s0, not along either line, then connected back to the arc with a
    # short leader -- keeps the text off both nearly-parallel lines it
    # describes.
    bis = np.radians(0.5 * (ang_rad + ang_beam2))
    perp = np.array([-np.sin(bis), np.cos(bis)])
    if np.dot(perp, hub_xy - s0) > 0:  # push the label AWAY from the hub side
        perp = -perp
    lbl_psi = s0 + 1.9 * r_arc * perp
    ax.annotate(f"$\\psi\\approx{psi:.0f}^\\circ$", xy=(s0[0] + 0.85 * r_arc * np.cos(bis),
                                                         s0[1] + 0.85 * r_arc * np.sin(bis)),
                xycoords='data', xytext=(lbl_psi[0], lbl_psi[1]), textcoords='data',
                ha='center', va='center', fontsize=7.2, color='k', zorder=9,
                bbox=dict(boxstyle='round,pad=0.12', fc='white', ec='none', alpha=0.95),
                arrowprops=dict(arrowstyle='-', lw=0.6, color='0.4'))
    # "grid edge" callout: anchored on the WEST dotted spoke (no beam runs
    # close to that spoke past its own tangent point), offset clear above it.
    anchor_pt = hub_xy + 0.75 * (sup_pts[2] - hub_xy)  # 2 = West neighbour
    lbl_pt = anchor_pt + np.array([0.0, 0.55])
    ax.annotate("grid edge\n(un-rotated)", xy=(anchor_pt[0], anchor_pt[1]), xycoords='data',
                xytext=(lbl_pt[0], lbl_pt[1]), textcoords='data',
                ha='center', va='bottom', fontsize=6.0, color='0.35', bbox=bbox, zorder=5,
                arrowprops=dict(arrowstyle='-', lw=0.6, color='0.55'))

    # magnified inset: engagement circle, connection dots, window label.
    # Generous margin (1.7x) keeps every beam stub strictly inside the
    # inset's own frame, well clear of its title text above. Placed at
    # axes-fraction [0.70,0.02,0.28,0.28]: its near (upper-left) corner maps
    # to data coordinates ~2.2*R_cluster from the hub (verified numerically),
    # comfortably outside the tangent-square cluster (radius ~0.35*R) --
    # unlike a larger/closer box, no beam chord can reach into this one.
    zoom_r_in = 1.7 * zoom_r
    axins = ax.inset_axes([0.70, 0.02, 0.28, 0.28])
    for k, (s_local, l_local, u_local) in enumerate(beam_pairs):
        p_s, p_u = txy[s_local], txy[u_local]
        axins.plot([p_s[0], p_u[0]], [p_s[1], p_u[1]], color='k', lw=1.8,
                   solid_capstyle='round', zorder=4)
    axins.scatter(tip_pts[:, 0], tip_pts[:, 1], color='k', s=28, zorder=6)
    th = np.linspace(0, 2 * np.pi, 200)
    axins.plot(hub_xy[0] + r_e * np.cos(th), hub_xy[1] + r_e * np.sin(th),
               color='0.3', lw=1.0, ls='-.', zorder=2)
    axins.set_xlim(hub_xy[0] - zoom_r_in, hub_xy[0] + zoom_r_in)
    axins.set_ylim(hub_xy[1] - zoom_r_in, hub_xy[1] + zoom_r_in)
    axins.set_aspect('equal')
    axins.set_xticks([]); axins.set_yticks([])
    for spine in axins.spines.values():
        spine.set_color('0.35'); spine.set_linewidth(0.8)
    # NOTE: axins.set_title() would center a title on the inset's OWN (tiny,
    # 0.28x0.28-fraction) width -- far narrower than the string "engagement
    # window (zoom)" at any legible font size, so the text overflows well
    # past the inset box and back across the main panel. Placed instead as
    # a short two-line label in the PARENT axes' fraction coordinates,
    # centered on the inset box's own x-centre, sized to actually fit.
    ax.annotate("engagement\nwindow (zoom)", xy=(0.84, 0.315), xycoords='axes fraction',
                ha='center', va='bottom', fontsize=5.6, color='0.15')
    # r_e label anchored at the angle (measured numerically) FARTHEST from
    # every beam segment on the engagement circle -- avoids the label sitting
    # under a beam stub.
    ang_re = np.radians(210)
    p_re = hub_xy + r_e * np.array([np.cos(ang_re), np.sin(ang_re)])
    axins.annotate(r"$r_e=\lambda R$", xy=(p_re[0], p_re[1]), xycoords='data',
                   xytext=(hub_xy[0] - 0.75 * zoom_r_in, hub_xy[1] - 0.85 * zoom_r_in),
                   textcoords='data', fontsize=5.8, color='0.15', ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.08', fc='white', ec='none', alpha=0.95),
                   arrowprops=dict(arrowstyle='-', color='0.5', lw=0.55, shrinkA=0, shrinkB=2))
    conn_ang = np.radians(140)
    ax.annotate("", xy=(0.70, 0.16), xycoords='axes fraction',
                xytext=(hub_xy[0] + zoom_r * np.cos(conn_ang), hub_xy[1] + zoom_r * np.sin(conn_ang)),
                textcoords='data', arrowprops=dict(arrowstyle='-', color='0.6', lw=0.6,
                                                    shrinkA=0, shrinkB=0), zorder=2)

    ax.set_title("(b) rotated fan at the hub", fontsize=8.3)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # ---- (c) resulting assembled nexorade patch (plan) -------------------
    from reciprocal_geom_v2 import square_nexorade
    ax = axes[2]
    model = square_nexorade(5, 5, cell=cell, lam=lam, ecc_ratio=ecc_ratio)
    for m in _member_lines(model):
        ax.plot([m['p1'][0], m['p2'][0]], [m['p1'][1], m['p2'][1]],
                color='k', lw=1.0, zorder=3)
    nodes = np.array(model.nodes)
    mids = _link_midpoints(model, nodes)
    ax.scatter(mids[:, 0], mids[:, 1], color='k', marker='o', s=8, zorder=6, linewidth=0)
    tpos = true_node_positions(model)
    sup = tpos[list(model.support_idx)]
    ax.scatter(sup[:, 0], sup[:, 1], marker='^', s=22, zorder=10,
               facecolor='white', edgecolor='k', linewidth=1.0)
    ax.set_title("(c) resulting nexorade patch (plan)", fontsize=8.3)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # NOTE: in-figure title/footer removed (2026 study) -- construction
    # explanation ("each interior vertex's 4 grid edges are rotated by
    # psi=arcsin(lambda) so every member becomes tangent to an engagement
    # circle r_e=lambda*R, producing a closed 4-beam mutually-supporting
    # fan; interior edges are then fused member-to-member across adjacent
    # hubs (c)") now given in the manuscript caption instead (see
    # latex/figure_title_migration.md).
    fig.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.04, wspace=0.18)
    savefig(fig, "fig_rotation_method")


# --------------------------------------------------------------------------
# fig_cs1_validation -- GRAYSCALE. results/cs1_fans_456.json: the CS1
# enumeration-validation study, EXTENDED (author request, 2026) from the
# single 4-beam fan to 4-, 5- and 6-beam fan_unit rotors, same protocol
# (8 lambda-levels x 8 CHS sections, 16-bit one-hot QUBO, SA-QUBO 100
# reads x 10 seeds) for every n. The 4-beam numbers are UNCHANGED from the
# pre-extension study (cs1_fans_456.py re-runs cs1_v2.run_case(4) with the
# identical seeded code path and asserts an EXACT match against
# results/cs1_v2.json's 'fan4' block -- see
# cs1_fans_456.json['reproducibility_check']).
#
# Panel (a): grouped bar chart, one bar per n in {4,5,6}, SA-QUBO success
# probability with a Wilson 95% CI error whisker on each bar (each n's own
# exact 10-seed Wilson CI, computed in cs1_fans_456.py via exact_mw.
# wilson_ci -- not re-derived here, just plotted).
# Panel (b): all three n's sorted 64-combo brute-force compliance
# landscapes OVERLAID on one log-y axis (distinct B/W linestyle/marker per
# n, via _BW_SERIES) rather than 3 small panels -- chosen after a VIEW
# pass: since every n's landscape spans a near-identical ~2.5-3 decades in
# relative terms (max/min ratio 662/540/521 for n=4/5/6) the three curves
# sit at recognizably different absolute compliance levels (bigger fans are
# less compliant... no: MORE compliant, i.e. higher J, at every rank) but
# with visually near-parallel shapes on a single shared log axis, which
# reads far more clearly as "same shape, shifted" in one panel than as
# three separate small panels that would each need their own axis labels
# repeated -- rank-0 (global optimum) of each curve is marked with a filled
# symbol, all independently verified against
# brute_force_matches_table_argmin=True for that n.
# --------------------------------------------------------------------------
def fig_cs1_validation():
    d = json.load(open(f"{RESULTS}/cs1_fans_456.json"))
    n_list = (4, 5, 6)

    ps, los, his = [], [], []
    for nb in n_list:
        r = d[f"fan{nb}"]
        p = r['sa_success_probability']
        w = r['wilson_95ci']
        ps.append(p); los.append(p - w['lo']); his.append(w['hi'] - p)
        print(f"[fig_cs1_validation] fan{nb}: p={p:.2f}  "
              f"Wilson 95% CI [{w['lo']:.3f}, {w['hi']:.3f}]")

    fig, axes = plt.subplots(1, 2, figsize=(150 * MM, 68 * MM))

    # ---- panel (a): grouped success-probability bars, Wilson 95% CI whiskers
    ax = axes[0]
    x = np.arange(len(n_list))
    bars = ax.bar(x, ps, width=0.5, color='0.6', edgecolor='k', linewidth=0.9, zorder=2)
    hatches = ['///', '\\\\\\', 'xxx']
    for bi, bar in enumerate(bars):
        bar.set_hatch(hatches[bi])
    ax.errorbar(x, ps, yerr=[los, his], fmt='none', ecolor='k',
                elinewidth=1.1, capsize=4, zorder=5)
    for xi, p in zip(x, ps):
        ax.text(xi, min(p + 0.10, 1.0), f"{p*100:.0f}%", ha='center', va='bottom', fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{nb}-beam" for nb in n_list])
    ax.set_xlim(-0.7, len(n_list) - 0.3)
    ax.set_ylabel("SA-QUBO success probability")
    ax.set_ylim(0, 1.15)
    ax.set_title("(a) success probability (Wilson 95% CI)", fontsize=8)

    # ---- panel (b): sorted brute-force compliance landscapes, all three n
    # overlaid on one shared log-y axis (NOT bar charts; NOT 3 small panels
    # -- see rationale above), global optimum of each marked at rank 0.
    ax2 = axes[1]
    for si, nb in enumerate(n_list):
        r = d[f"fan{nb}"]
        st = _BW_SERIES[si]
        T = np.array(r['compliance_table'])
        energies = np.sort(T.flatten())
        ax2.plot(energies, color=st['color'], lw=1.1, ls=st['ls'],
                  label=f"{nb}-beam fan (sorted, 64 combos)")
        opt = r['global_opt_compliance']
        assert r['brute_force_matches_table_argmin'], f"fan{nb}: global opt is not the table argmin"
        ax2.scatter([0], [opt], color=st['color'], marker=st['marker'], s=30,
                     zorder=6, edgecolor='k', linewidth=0.7)
    ax2.set_xlabel("rank (sorted compliance)")
    ax2.set_ylabel("compliance [J]")
    ax2.set_yscale('log')
    ax2.legend(loc='upper left', fontsize=6.2)
    ax2.set_title("(b) brute-force compliance landscapes", fontsize=8)

    fig.tight_layout()
    savefig(fig, "fig_cs1_validation")


# --------------------------------------------------------------------------
# fig_cs2_convergence / fig_cs3_convergence / fig_cs6_convergence (NEW) --
# GRAYSCALE. Handles both list- and dict-(seed-string)-keyed `methods`
# entries via _runs(). Renamed to a leading-underscore _impl (2026, complex
# dome/bridge upgrade) so it is no longer swept up by the fig_* prefix
# auto-discovery in __main__ (it requires positional args and would raise
# TypeError if called bare) -- the three thin wrapper functions below are
# the actual fig_cs2_convergence/fig_cs3_convergence/fig_cs6_convergence
# entry points auto-discovery finds and calls.
# --------------------------------------------------------------------------
def _fig_convergence_impl(json_name, title, outname, ylabel="compliance", annotation=None):
    d = json.load(open(f"{RESULTS}/{json_name}"))
    fig, ax = plt.subplots(figsize=(90 * MM, 70 * MM))
    labels = {'sa_qubo': 'SA-QUBO (F1)', 'ga': 'GA', 'random': 'Random search'}
    for i, method in enumerate(('sa_qubo', 'ga', 'random')):
        st = _BW_SERIES[i]
        runs = _runs(d['methods'][method])
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
    # NOTE: in-figure title removed (2026 study) -- `title` (e.g. "CS2
    # rotated-square nexorade sizing convergence") now given in the
    # manuscript caption instead (see latex/figure_title_migration.md).
    ax.legend(loc='upper right')
    if annotation:
        ax.text(0.02, 0.02, annotation, transform=ax.transAxes, fontsize=6.5,
                ha='left', va='bottom',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='gray', alpha=0.85))
    fig.tight_layout()
    savefig(fig, outname)


def fig_cs2_convergence():
    _fig_convergence_impl('cs2_v2.json', 'CS2 rotated-square nexorade sizing convergence',
                           'fig_cs2_convergence')


def fig_cs3_convergence():
    _fig_convergence_impl('cs3_v2.json', 'CS3 hexagonal nexorade sizing convergence',
                           'fig_cs3_convergence')


def fig_cs6_convergence():
    # REVISION (bridge redesign, 2026): results/complex/cs6_complex.json
    # now holds the RE-RUN campaign on davinci_bridge2 (load-path-corrected
    # 3-rail, 24-bay bridge; 118 physical members / 16 groups, vs the old
    # bridge's 58 members -- see reciprocal_geom_v2.davinci_bridge2 and
    # results/bridge2_verification.json). Same 'sa_qubo'/'ga'/'random'
    # method keys, same list-of-per-seed-dict shape (handled by
    # _runs()/_fig_convergence_impl unchanged), 10 seeds/method; the OLD
    # davinci_bridge_complex campaign is preserved verbatim at
    # results/complex/cs6_complex_bridge1.json. Member-count annotation
    # (NEW, 2026 bridge redesign QA requirement) added so the convergence
    # curves are self-contained about which geometry/group count they refer to.
    d = json.load(open(f"{RESULTS}/complex/cs6_complex.json"))
    n_groups = d.get('n_groups', 16)
    _fig_convergence_impl('complex/cs6_complex.json',
                           'CS6 davinci_bridge2 (redesigned, 3-rail) sizing convergence',
                           'fig_cs6_convergence',
                           annotation=f"davinci_bridge2: 118 physical members, {n_groups} groups")


# --------------------------------------------------------------------------
# fig_cs4_fmqa -- GRAYSCALE. REVISION (complex dome/bridge upgrade, 2026):
# now reads results/complex/cs4_complex.json (rf_dome_complex, rings=4 /
# 5 ring families / 48-bit black box) instead of the earlier rf_dome
# (rings=3 / 40-bit) cs4_v2.json -- methods are STILL dict-keyed by seed
# string (handled by _runs()), same 'fmqa'/'ga'/'random' keys, same
# 'best_history' per-seed field, so no other plotting logic changes. HONEST
# result on the denser geometry: FMQA now achieves the lowest mean
# compliance of the three methods (5.9% better than GA, 14.4% better than
# random), a clear trend but NOT significant vs GA at 10 seeds (exact
# Mann-Whitney p=0.072) -- see results/complex_summary.md Section 1 --
# still plotted with no visual distortion of that (real, if not
# significant) FMQA/GA gap.
# --------------------------------------------------------------------------
def fig_cs4_fmqa():
    d = json.load(open(f"{RESULTS}/complex/cs4_complex.json"))
    fig, ax = plt.subplots(figsize=(90 * MM, 70 * MM))
    labels = {'fmqa': 'FMQA (F2)', 'ga': 'GA', 'random': 'Random search'}
    _all_mean = []
    for i, method in enumerate(('fmqa', 'ga', 'random')):
        st = _BW_SERIES[i]
        runs = _runs(d['methods'][method])
        hist = [r['best_history'] for r in runs]
        mean, std, n = _mean_std_bands(hist)
        _all_mean.append(mean)
        x = np.arange(1, n + 1)
        ax.plot(x, mean, color=st['color'], ls=st['ls'], marker=st['marker'],
                markevery=_markevery(n), ms=3.5, mfc='white', mew=0.7,
                label=labels[method])
        ax.fill_between(x, mean - std, mean + std, color=st['color'], alpha=0.16, linewidth=0)
    ax.set_xlabel("real FE evaluations")
    ax.set_ylabel("compliance (best-so-far)")
    ax.set_yscale('log')
    _clamp_log_ylim(ax, _all_mean)
    # NOTE: in-figure title removed (2026 study) -- "CS4-complex
    # reciprocal dome: FMQA vs baselines (10 seeds)" now given in the
    # manuscript caption instead (see latex/figure_title_migration.md).
    ax.legend(loc='upper right')
    fig.tight_layout()
    savefig(fig, "fig_cs4_fmqa")


# --------------------------------------------------------------------------
# fig_ablation -- GRAYSCALE. CS2-v2 w_mass=20 ablation: SA-QUBO (paired)
# vs greedy-linearized (paired, r3_ablation_v2.json) vs the ORIGINAL
# w_mass=5 GA baseline from cs2_v2.json (not budget/weight-matched to the
# two w20 curves -- stated explicitly, matching v1's r3 convention).
# --------------------------------------------------------------------------
def fig_ablation():
    d3 = json.load(open(f"{RESULTS}/r3_ablation_v2.json"))
    d2 = json.load(open(f"{RESULTS}/cs2_v2.json"))

    sa_hist = [r['compliance_best_so_far'] for r in _runs(d3['methods']['sa_qubo_paired_w20'])]
    greedy_hist = [r['compliance_best_so_far'] for r in _runs(d3['methods']['greedy_w20'])]
    ga_hist = [r['compliance_best_so_far'] for r in _runs(d2['methods']['ga'])]

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
    # NOTE: in-figure title removed (2026 study) -- "CS2-v2 ablation:
    # SA-QUBO vs greedy-linearized vs GA (mean+/-std, 10 seeds)" now given
    # in the manuscript caption instead (see
    # latex/figure_title_migration.md).
    ax.legend(loc='upper right', fontsize=6.5)
    fig.tight_layout()
    savefig(fig, "fig_ablation")


# --------------------------------------------------------------------------
# fig_cs5_pareto -- GRAYSCALE. CS5-v2: square_nexorade multi-objective,
# 5 seeds (reduced from 10; stated explicitly in title/axis, per
# summary_v2.md Section 13). Representative-seed fronts renormalized here
# from cs5_v2.json's raw (compliance, mass) points using the SAME min-max
# ranges recorded in cs5_analysis_v2.json; hypervolume mean/std/per-seed
# dots read directly from cs5_analysis_v2.json.
# --------------------------------------------------------------------------
def fig_cs5_pareto():
    d = json.load(open(f"{RESULTS}/cs5_v2.json"))
    da = json.load(open(f"{RESULTS}/cs5_analysis_v2.json"))
    from qubo import pareto_front

    comp_lo, comp_hi = da['compliance_range']
    mass_lo, mass_hi = da['mass_range']

    def norm(pts):
        C = np.array([p['compliance'] for p in pts])
        M = np.array([p['mass'] for p in pts])
        Cn = (C - comp_lo) / (comp_hi - comp_lo)
        Mn = (M - mass_lo) / (mass_hi - mass_lo)
        return np.stack([Cn, Mn], axis=1)

    per_seed = da['per_seed']
    seeds = list(per_seed.keys())

    def rep_seed(key):
        vals = np.array([per_seed[s][key] for s in seeds])
        mean = vals.mean()
        return seeds[int(np.argmin(np.abs(vals - mean)))]

    sa_rep = rep_seed('sa_hv')
    nsga_rep = rep_seed('nsga_hv')
    sa_pts = norm(d['sa_points'][sa_rep])
    nsga_pts = norm(d['nsga_points'][nsga_rep])
    sa_front = sa_pts[pareto_front(sa_pts)]
    nsga_front = nsga_pts[pareto_front(nsga_pts)]

    fig, axes = plt.subplots(1, 2, figsize=(180 * MM, 78 * MM))

    ax = axes[0]
    o1 = np.argsort(sa_front[:, 0]); o2 = np.argsort(nsga_front[:, 0])
    ax.plot(sa_front[o1, 0], sa_front[o1, 1], color='k', ls='-', marker='o',
            ms=3, lw=1.0, mfc='white', mew=0.6, label=f'SA-QUBO weighted-sum (seed {sa_rep})')
    ax.plot(nsga_front[o2, 0], nsga_front[o2, 1], color='0.45', ls='--',
            marker='s', ms=3, lw=1.0, mfc='white', mew=0.6, label=f'NSGA-II (seed {nsga_rep})')
    ax.set_xlabel("normalized compliance")
    ax.set_ylabel("normalized mass")
    ax.set_xscale('log')
    # NOTE: in-figure title trimmed to a bare panel identifier (2026
    # study) -- the removed detail ("matched {n_evals_target}-eval
    # budget, n={n_seeds} seeds") now given in the manuscript caption
    # instead (see latex/figure_title_migration.md).
    ax.set_title("(a) representative-seed non-dominated fronts", fontsize=8)
    ax.legend(loc='upper right', fontsize=6)

    ax2 = axes[1]
    methods = ['sa', 'nsga']
    labels = ['SA-QUBO\nweighted-sum', 'NSGA-II']
    marker_shape = ['o', 's']
    grays = ['k', '0.45']
    means = [da['sa_hv_mean'], da['nsga_hv_mean']]
    stds = [da['sa_hv_std'], da['nsga_hv_std']]
    x = np.arange(2)
    for i, m in enumerate(methods):
        vals = [per_seed[s][f'{m}_hv'] for s in seeds]
        jitter = (np.random.default_rng(0).random(len(vals)) - 0.5) * 0.12
        ax2.scatter(np.full(len(vals), x[i]) + jitter, vals, s=13, color=grays[i],
                    alpha=0.35, zorder=2, linewidth=0, marker=marker_shape[i])
    ax2.errorbar(x, means, yerr=stds, fmt='none', capsize=4, color='k', elinewidth=1.1, zorder=3)
    for i in range(2):
        ax2.scatter([x[i]], [means[i]], s=46, color=grays[i], marker=marker_shape[i], zorder=4,
                    edgecolor='k', linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xlim(-0.5, 1.5)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("2D hypervolume (ref (1.2,1.2), norm. units)")
    pct = 100.0 * (da['sa_hv_mean'] - da['nsga_hv_mean']) / da['nsga_hv_mean']
    # NOTE: in-figure title trimmed to a bare panel identifier (2026
    # study) -- the removed detail ("n={n_seeds} seeds each (dots),
    # +{pct:.1f}% SA-QUBO, p={mann_whitney_p:.1e}") now given in the
    # manuscript caption instead (see latex/figure_title_migration.md).
    ax2.set_title("(b) hypervolume comparison", fontsize=8)

    fig.tight_layout()
    savefig(fig, "fig_cs5_pareto")


# --------------------------------------------------------------------------
# fig_sensitivity -- GRAYSCALE (twin-axis; series distinguished by
# linestyle + marker, combined into a single legend). penalty_sensitivity_v2.json
# --------------------------------------------------------------------------
def fig_sensitivity():
    d = json.load(open(f"{RESULTS}/penalty_sensitivity_v2.json"))
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
    # NOTE: in-figure title removed (2026 study) -- "CS2-v2 penalty-weight
    # sensitivity (3 seeds/level)" now given in the manuscript caption
    # instead (see latex/figure_title_migration.md).
    fig.tight_layout()
    savefig(fig, "fig_sensitivity")


# --------------------------------------------------------------------------
# fig_encoding_qubo (COLOR, panel (b) only) -- (a) one-hot member-group
# encoding diagram (unchanged content, CHS catalog is v2-identical, kept as
# mono line-art per author instruction); (b) REAL |Q| heatmap rebuilt from
# one actual v2 CS2 F1 QUBO iteration: square_nexorade(6,6,cell=2.5,
# lam=0.25) -- the SAME geometry/mass-target/w_mass=5.0 used for the real
# CS2-v2 headline result (12 groups x 8 sections = 96 bits; mid-catalog
# init, matching sizing.iterative_qubo_sizing's own default
# init_idx=[N_SECTIONS//2]*n_groups). REVISION (author feedback, 2026):
# panel (b) is now a COLOR (viridis, colorblind-safe) heatmap instead of
# v1's grayscale 'Greys' map -- this is the one required color exception
# for this otherwise-B/W figure. Annotation layout (corner callouts,
# top-layer text z-order, exact arrow targets) is unchanged from v1 (still
# legible: opaque white text boxes over the colored matrix); only the
# block count/matrix size grew (64->96 bits) since v2's real CS2 geometry
# has 12 groups instead of the artificially-reduced 8-group illustrative
# instance used in v1.
# --------------------------------------------------------------------------
def _draw_onehot_panel(ax):
    from sections import CHS_CATALOG, N_SECTIONS

    ax.set_xlim(0, 10.0); ax.set_ylim(0, 10.0)
    ax.axis('off')
    chosen = 3

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


def _build_real_f1_qubo_v2():
    """Rebuild ONE real v2 F1 (strain-energy-linearized) QUBO exactly as
    sizing.iterative_qubo_sizing() does for its first outer iteration on
    the ACTUAL CS2-v2 case-study geometry/mass-target (square_nexorade(6,6,
    cell=2.5,lam=0.25), mid-catalog init, w_mass=5.0): one real FE solve,
    per (group, section) strain energies via the real local element
    stiffness matrices, then the actual QUBOBuilder one-hot + mass-target
    penalty terms (the move-limit big-penalty exclusion term is omitted so
    the underlying one-hot/mass-coupling block structure is visible in the
    heatmap; every other term is identical to sizing.py)."""
    from reciprocal_geom_v2 import square_nexorade
    from reciprocal_geom import model_mass
    from sections import CHS_CATALOG, N_SECTIONS, E_STEEL, G_STEEL
    from sizing import element_geom_cache, group_element_indices
    from qubo import QUBOBuilder
    from frame3d import local_stiffness

    model = square_nexorade(6, 6, 2.5, 0.25)  # identical to cs2_v2.py's make_model()
    load_per_node = model.meta['load_per_node']
    gs_max = {k: CHS_CATALOG[-1] for k in range(model.n_groups)}
    mass_target = 0.6 * model_mass(model, gs_max)
    n_groups, n_sec = model.n_groups, N_SECTIONS
    cur = [n_sec // 2] * n_groups  # sizing.iterative_qubo_sizing's own default init_idx

    from sizing import evaluate_design
    _, _, frm = evaluate_design(model, cur, load_per_node)

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
    w_stiff, w_mass = 1.0, 5.0  # matches cs2_v2.py's W_MASS
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
    Q, n_groups, n_sec = _build_real_f1_qubo_v2()
    n_bits = n_groups * n_sec

    fig = plt.figure(figsize=(185 * MM, 98 * MM))
    gsfig = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.38)

    ax1 = fig.add_subplot(gsfig[0, 0])
    _draw_onehot_panel(ax1)

    ax2 = fig.add_subplot(gsfig[0, 1])
    absQ = np.abs(Q)
    vmax = np.percentile(absQ, 99.5)
    from matplotlib.colors import PowerNorm
    # REVISION (author feedback, 2026): panel (b) is now COLOR -- viridis
    # (colorblind-safe, perceptually uniform) replaces the v1 grayscale
    # 'Greys' colormap so |Q_ij| magnitude reads unambiguously; panel (a)
    # (_draw_onehot_panel, above) stays mono line-art per the author's
    # instruction that only panel (b)'s heatmap needs color. Gridlines are
    # white (rather than v1's mid-gray) so they stay visible against both
    # the dark (low-|Q|) and bright-yellow (high-|Q|) ends of viridis; the
    # two callout annotations already sit in opaque white text boxes
    # (_lbl_bbox, below) so they stay legible over the colored heatmap.
    # Log scale + masked zeros (rendered white): zero entries drop out
    # entirely and the weak cross-group couplings sit mid-scale instead of
    # near-black, so the one-hot blocks and the faint mass-penalty
    # couplings are both clearly distinguishable.
    from matplotlib.colors import LogNorm
    import numpy.ma as ma
    nz = absQ[absQ > 0]
    vmin_log = max(nz.min(), vmax * 1e-4)
    absQ_m = ma.masked_where(absQ <= 0, absQ)
    cmap_l = plt.get_cmap('viridis').copy()
    cmap_l.set_bad('white')
    im = ax2.imshow(absQ_m, cmap=cmap_l, norm=LogNorm(vmin=vmin_log, vmax=vmax))
    for b in range(0, n_bits + 1, n_sec):
        ax2.axhline(b - 0.5, color='0.75', lw=0.35, alpha=0.8)
        ax2.axvline(b - 0.5, color='0.75', lw=0.35, alpha=0.8)
    ax2.add_patch(mpatches.Rectangle((-0.5, -0.5), n_sec, n_sec, fill=False, ec='white', lw=1.3))
    ax2.add_patch(mpatches.Rectangle((-0.5, -0.5), n_sec, n_sec, fill=False, ec='k', lw=0.6))
    ax2.set_xticks([]); ax2.set_yticks([])
    # NOTE: in-figure title trimmed to a bare panel identifier (2026
    # study) -- the removed detail ("one real CS2-v2 F1 QUBO iteration,
    # square_nexorade: {n_groups} groups x {n_sec} sections = {n_bits}
    # bits") now given in the manuscript caption instead (see
    # latex/figure_title_migration.md).
    ax2.set_title("(b) $|Q|$ matrix", fontsize=8)
    # Extra top/bottom margin so BOTH callouts sit entirely OUTSIDE the
    # matrix data (unchanged v1 layout convention): one in the top-left
    # corner, one in the bottom-left corner, well clear of the title and
    # of each other. y is inverted (imshow row-index convention).
    ax2.set_ylim(n_bits + 15, -22)
    _lbl_bbox = dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.85)
    ax2.annotate("one-hot penalty\nblock (group 0)", xy=(n_sec - 0.5, n_sec / 2 - 0.5), xycoords='data',
                 xytext=(n_sec + 6, -11), textcoords='data', fontsize=6.2, ha='left', va='center',
                 bbox=_lbl_bbox, zorder=9,
                 arrowprops=dict(arrowstyle='->', lw=0.7, color='k', shrinkA=2, shrinkB=2))
    ax2.annotate("mass-target coupling\n(weak, cross-group)",
                 xy=(2, n_bits - 3), xycoords='data',
                 xytext=(n_sec + 6, n_bits + 11), textcoords='data', fontsize=6.2, ha='left', va='center',
                 bbox=_lbl_bbox, zorder=9,
                 arrowprops=dict(arrowstyle='->', lw=0.7, color='k', shrinkA=2, shrinkB=2))
    cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label("$|Q_{ij}|$")
    savefig(fig, "fig_encoding_qubo")


# --------------------------------------------------------------------------
# fig_qubo_grid (COLOR) -- 3x3 grid of REAL |Q|-matrix heatmaps, one per
# distinct QUBO instance actually used somewhere in the manuscript (author
# feedback, 2026: "present all Q matrixes, you can do it as grid plot").
# Every panel is rebuilt from the SAME real code paths used by the actual
# campaigns (cs1_v2 / cs2_v2 / cs3_v2 / cs6_complex_bridge2 / dome2_sizing /
# vault_hex2_sizing / fmqa+resumable_v2 / cs7_stress_constrained / cs5_v2's
# sizing.weighted_sum_qubo_sizing) -- NOT re-derived or illustrative. Same
# masked-zero, log-scale viridis convention as fig_encoding_qubo's panel
# (b), but each panel gets its OWN colorbar: per-panel absolute |Q_ij|
# ranges span ~3 orders of magnitude across panels (from O(1)-O(20) for
# the strain-energy/mass-linearized F1 QUBOs and the CS5 weighted-sum
# QUBO's purely-linear objective, up to O(500)-O(3000) for CS7's dominant
# stress-feasibility penalty and CS1's un-normalized compliance-table
# entries) -- a single shared colorbar would wash out the low-magnitude
# panels; decided by inspection (see results/qubo_grid_provenance.json
# for the exact per-panel |Q| ranges).
# --------------------------------------------------------------------------
def _panel_qmatrix(ax, Q, block_size, label):
    """Shared rendering helper (masked-zero + log-scale viridis + block
    gridlines every `block_size` bits), reproducing fig_encoding_qubo's
    panel (b) convention exactly, generalized to any bit count / one-hot
    block size."""
    from matplotlib.colors import LogNorm
    import numpy.ma as ma
    n_bits = Q.shape[0]
    absQ = np.abs(Q)
    vmax = np.percentile(absQ, 99.5)
    nz = absQ[absQ > 0]
    vmin_log = max(nz.min(), vmax * 1e-4) if nz.size else 1e-9
    absQ_m = ma.masked_where(absQ <= 0, absQ)
    cmap_l = plt.get_cmap('viridis').copy()
    cmap_l.set_bad('white')
    im = ax.imshow(absQ_m, cmap=cmap_l, norm=LogNorm(vmin=vmin_log, vmax=vmax))
    for b in range(0, n_bits + 1, block_size):
        ax.axhline(b - 0.5, color='0.75', lw=0.3, alpha=0.8)
        ax.axvline(b - 0.5, color='0.75', lw=0.3, alpha=0.8)
    ax.add_patch(mpatches.Rectangle((-0.5, -0.5), n_bits, n_bits, fill=False, ec='k', lw=0.6))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(label, fontsize=7.4)
    cbar = plt.gcf().colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.ax.tick_params(labelsize=5.0)
    # NARROW-DYNAMIC-RANGE panels (e.g. CS5's purely-linear weighted-sum
    # QUBO, whose |Q_ij| all sit within one decade ~2.99-3.0) produce a
    # cramped multi-decimal-digit LogLocator tick ladder by default; swap
    # in a coarse 4-tick linear-position locator + short '%.3g' labels
    # whenever the span is under one decade so the colorbar stays legible.
    if vmax / vmin_log < 10.0:
        from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator, NullFormatter
        ticks = np.geomspace(vmin_log, vmax, 4)
        cbar.ax.yaxis.set_minor_locator(NullLocator())
        cbar.ax.yaxis.set_minor_formatter(NullFormatter())
        cbar.ax.yaxis.set_major_locator(FixedLocator(ticks))
        cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:.4g}"))
    return im


def _build_f1_qubo(model, load_per_node=None, w_mass=5.0, mass_frac=0.6,
                    check_cond=True):
    """Generic real-F1-QUBO builder (ONE outer iteration, mid-catalog
    init, matching sizing.iterative_qubo_sizing's own defaults) -- same
    construction as _build_real_f1_qubo_v2 above (move-limit big-penalty
    exclusion term omitted so the one-hot/mass-coupling block structure is
    visible), generalized to any v2 geometry model so it can be reused for
    CS2 (square_nexorade), CS3 (hex_nexorade), CS6 (davinci_bridge2),
    dome-fixed (rf_dome2) and vault-fixed (rf_vault_hex2). check_cond
    matches each geometry's own real campaign convention (True for
    CS2/CS3, False/fastsolve for the heavier bridge2/dome2/vault_hex2
    classes, exactly as sizing_campaign.py / cs6_complex_bridge2.py use)."""
    from reciprocal_geom import model_mass
    from sections import CHS_CATALOG, N_SECTIONS, E_STEEL, G_STEEL
    from sizing import element_geom_cache, group_element_indices, evaluate_design
    from qubo import QUBOBuilder
    from frame3d import local_stiffness

    if load_per_node is None:
        load_per_node = model.meta['load_per_node']
    gs_max = {k: CHS_CATALOG[-1] for k in range(model.n_groups)}
    mass_target = mass_frac * model_mass(model, gs_max)
    n_groups, n_sec = model.n_groups, N_SECTIONS
    cur = [n_sec // 2] * n_groups
    _, _, frm = evaluate_design(model, cur, load_per_node, check_cond=check_cond)

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
    w_stiff = 1.0
    for k in range(n_groups):
        for c in range(n_sec):
            qb.add_linear(vidx(k, c), -w_stiff * SE[k, c] / SEmax)
    all_idx = [vidx(k, c) for k in range(n_groups) for c in range(n_sec)]
    all_coef = [mass_coef[k, c] for k in range(n_groups) for c in range(n_sec)]
    qb.add_target_penalty(all_idx, all_coef, mass_target, w_mass / max(mass_target**2, 1e-9))
    w_onehot = 3.0 * (w_stiff + w_mass)
    for k in range(n_groups):
        qb.add_onehot_penalty([vidx(k, c) for c in range(n_sec)], w_onehot)
    return qb.Q, n_groups, n_sec, mass_target


def _build_cs1_qubo_grid():
    """Panel (a): CS1 fan-unit 16-bit table QUBO, n_beams=4 (the CS1
    typology actually reported in the manuscript, results/cs1_v2.json's
    'fan4' block / results/cs1_fans_456.json's reproducibility-checked
    4-beam re-run) -- SAME cs1_v2.compliance_table + build_table_qubo call
    used by the real CS1 validation study, not re-derived."""
    from cs1_v2 import compliance_table, build_table_qubo
    T, M = compliance_table(4)
    Q, lam_idx, sec_idx, P = build_table_qubo(T)
    return Q, 16


def _build_cs5_weighted_qubo(w=0.5):
    """Panel (i): ONE real outer iteration of CS5-v2's weighted-sum
    multi-objective QUBO (sizing.weighted_sum_qubo_sizing), at a
    representative weight w=0.5 (midpoint of cs5_v2.py's W_LEVELS =
    linspace(0,1,11) sweep), on the SAME square_nexorade(6,6,cell=2.5,
    lam=0.25) geometry/mass_scale (=all-max-section mass, no 0.6 mass_frac)
    used by the real CS5-v2 campaign. This QUBO's objective is PURELY
    LINEAR in stiffness+mass (no quadratic mass-target penalty, unlike
    CS2/CS3/CS6/dome/vault's F1 QUBO) -- so its |Q| matrix is exactly
    block-diagonal (only the per-group one-hot penalty blocks are
    nonzero); verified numerically (max |Q_ij| outside the 12 diagonal
    8x8 blocks = 0.0)."""
    from reciprocal_geom_v2 import square_nexorade
    from reciprocal_geom import model_mass
    from sections import CHS_CATALOG, N_SECTIONS, E_STEEL, G_STEEL
    from sizing import element_geom_cache, group_element_indices, evaluate_design
    from qubo import QUBOBuilder
    from frame3d import local_stiffness

    model = square_nexorade(6, 6, 2.5, 0.25)  # identical to cs5_v2.py's build_setup()
    load_per_node = model.meta['load_per_node']
    n_groups, n_sec = model.n_groups, N_SECTIONS
    geom = element_geom_cache(model)
    groups = group_element_indices(model)
    gs_max = {k: CHS_CATALOG[-1] for k in range(n_groups)}
    mass_scale = model_mass(model, gs_max)

    cur = [n_sec // 2] * n_groups  # weighted_sum_qubo_sizing's own default init_idx
    _, _, frm = evaluate_design(model, cur, load_per_node)

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

    qb = QUBOBuilder(n_groups * n_sec)

    def vidx(k, c):
        return k * n_sec + c

    SEmax = SE.max() if SE.max() > 0 else 1.0
    for k in range(n_groups):
        for c in range(n_sec):
            stiff_term = -w * (SE[k, c] / SEmax)
            mass_term = (1 - w) * (mass_coef[k, c] / mass_scale)
            qb.add_linear(vidx(k, c), stiff_term + mass_term)
    w_onehot = 3.0
    for k in range(n_groups):
        qb.add_onehot_penalty([vidx(k, c) for c in range(n_sec)], w_onehot)
    return qb.Q, n_groups, n_sec


def _build_cs7_stress_qubo():
    """Panel (h): ONE real outer iteration of RI's stress-constrained
    sizing QUBO (cs7_stress_constrained.sa_stress_chunk's per-iteration
    construction) on the SAME scaled-load square_nexorade(6,6,cell=2.5,
    lam=0.25) CS2 geometry used by the real RI campaign (load scaled so
    the uniform mid-catalog design sits at max utilization ~=0.9,
    w_mass=20, w_stress=500 dominant linear feasibility penalty added to
    every bit x_{g,c} with c below the frozen-force feasible section
    c_min,g) -- reuses cs7_stress_constrained.build_setup/group_forces/
    frozen_force_cmin verbatim, not re-derived."""
    # cs7_stress_constrained.py reads sys.argv[1] at MODULE import time (its
    # own TIME_BUDGET_S CLI default) -- shield that against schematics_nexorade.
    # py's own argv (figure names) by importing with a clean argv, then
    # restoring it immediately (does not affect anything else in this run).
    _argv_save = sys.argv[:]
    sys.argv = [sys.argv[0]]
    try:
        import cs7_stress_constrained as RI
    finally:
        sys.argv = _argv_save
    from sections import CHS_CATALOG, N_SECTIONS, E_STEEL, G_STEEL
    from sizing import element_geom_cache, group_element_indices, evaluate_design
    from qubo import QUBOBuilder
    from frame3d import local_stiffness

    setup = RI.build_setup()
    model = setup['model']
    geom = element_geom_cache(model)
    groups = group_element_indices(model)
    n_groups, n_sec = model.n_groups, N_SECTIONS
    cur = [n_sec // 2] * n_groups
    _, _, frm = evaluate_design(model, cur, setup['load_scaled'])
    gforces = RI.group_forces(model, frm, groups)
    c_min, _ = RI.frozen_force_cmin(gforces, n_groups, n_sec)

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

    qb = QUBOBuilder(n_groups * n_sec)

    def vidx(k, c):
        return k * n_sec + c

    SEmax = SE.max() if SE.max() > 0 else 1.0
    W_MASS, W_STRESS = RI.W_MASS, RI.W_STRESS
    for k in range(n_groups):
        for c in range(n_sec):
            coef = -SE[k, c] / SEmax
            if c < c_min[k]:
                coef += W_STRESS
            qb.add_linear(vidx(k, c), coef)
    all_idx = [vidx(k, c) for k in range(n_groups) for c in range(n_sec)]
    all_coef = [mass_coef[k, c] for k in range(n_groups) for c in range(n_sec)]
    qb.add_target_penalty(all_idx, all_coef, setup['mass_target'],
                           W_MASS / max(setup['mass_target']**2, 1e-9))
    w_onehot = 3.0 * (1.0 + W_MASS)
    for k in range(n_groups):
        qb.add_onehot_penalty([vidx(k, c) for c in range(n_sec)], w_onehot)
    return qb.Q, n_groups, n_sec, c_min.tolist()


def _build_fmqa_qubo(seed=9, cyc=30):
    """Panel (g): the ACTUAL factorization-machine-derived QUBO from a
    real CS4 (rf_dome2) FMQA geometry-search acquisition cycle, read back
    from the checkpointed run state in results/complex/cs4_complex.json
    (seed 9's fmqa_states block is the only one of the 10 saved with the
    full 80-eval run, i.e. init_done=20/20, cyc=60/60) -- refits the SAME
    rank-8 factorization machine (fmqa.FactorizationMachine) on the first
    `n_init + cyc` real (x,y) pairs actually evaluated by that run, using
    the EXACT seed formula resumable_v2.fmqa_step_cycles uses
    (seed*1000+cyc), then reads off build_fm_qubo's learned quadratic form
    + one-hot penalties -- i.e. this is not a fresh/synthetic FM fit, it
    is a reproduction of the QUBO the real run solved at that acquisition
    cycle (cyc=30 of 60, a representative mid-run surrogate: FM trained on
    50 of the eventual 80 real FE evaluations)."""
    from fmqa import FactorizationMachine, build_fm_qubo
    with open(f"{RESULTS}/complex/cs4_complex.json") as f:
        d = json.load(f)
    st = d['fmqa_states'][str(seed)]
    cat_counts = st['cat_counts']
    offsets = np.cumsum([0] + list(cat_counts))[:-1]
    n_bits = int(sum(cat_counts))
    blocks = [list(range(int(offsets[b]), int(offsets[b]) + cat_counts[b]))
              for b in range(len(cat_counts))]
    n_used = st['n_init'] + cyc
    X = np.array(st['X_data'][:n_used])
    y = np.array(st['y_data'][:n_used])
    fm = FactorizationMachine(n_bits, k=8, seed=seed * 1000 + cyc)
    fm.fit(X, y, epochs=80, lr=0.08, seed=seed * 1000 + cyc)
    Q, pw = build_fm_qubo(fm, blocks)
    return Q, n_bits, n_used, cat_counts


def fig_qubo_grid():
    panels = []  # (label, Q, block_size, provenance dict)

    Q1, n1 = _build_cs1_qubo_grid()
    panels.append(("(a) CS1 fan-unit table\n16 bits", Q1, 8, dict(
        panel="a", n_bits=16, case_study="CS1",
        description="16-bit table QUBO (8 lambda one-hot x 8 CHS-section one-hot), "
                     "n_beams=4 fan_unit, exact 8x8 compliance table",
        source="src/cs1_v2.py (compliance_table(4)+build_table_qubo); "
                "results/cs1_v2.json 'fan4' / results/cs1_fans_456.json")))

    from reciprocal_geom_v2 import square_nexorade, hex_nexorade, davinci_bridge2, rf_dome2, rf_vault_hex2

    m2 = square_nexorade(6, 6, 2.5, 0.25)
    Q2, ng2, ns2, mt2 = _build_f1_qubo(m2, w_mass=5.0, check_cond=True)
    panels.append((f"(b) CS2 square nexorade F1\n{ng2*ns2} bits", Q2, ns2, dict(
        panel="b", n_bits=ng2 * ns2, n_groups=ng2, n_sections=ns2, case_study="CS2",
        description="F1 strain-energy-linearized sizing QUBO, one real outer iteration "
                     "(mid-catalog init, move-limit exclusion omitted), w_mass=5.0, "
                     "mass_target=0.6*mass_max",
        source="src/cs2_v2.py make_model() (square_nexorade(6,6,cell=2.5,lam=0.25)) "
                "+ src/sizing.py iterative_qubo_sizing's per-iteration QUBO construction")))

    m3 = hex_nexorade(3, 2.2, 0.25)
    Q3, ng3, ns3, mt3 = _build_f1_qubo(m3, w_mass=5.0, check_cond=True)
    panels.append((f"(c) CS3 hex nexorade F1\n{ng3*ns3} bits", Q3, ns3, dict(
        panel="c", n_bits=ng3 * ns3, n_groups=ng3, n_sections=ns3, case_study="CS3",
        description="F1 strain-energy-linearized sizing QUBO, one real outer iteration, "
                     "w_mass=5.0, mass_target=0.6*mass_max",
        source="src/cs3_v2.py make_model() (hex_nexorade(3,edge=2.2,lam=0.25)) "
                "+ src/sizing.py iterative_qubo_sizing")))

    m6 = davinci_bridge2(n_bays=24, span=20.0, width=6.0, lam=0.25)
    Q6, ng6, ns6, mt6 = _build_f1_qubo(m6, w_mass=5.0, check_cond=False)
    panels.append((f"(d) CS6 bridge F1\n{ng6*ns6} bits", Q6, ns6, dict(
        panel="d", n_bits=ng6 * ns6, n_groups=ng6, n_sections=ns6, case_study="CS6",
        description="F1 strain-energy-linearized sizing QUBO, one real outer iteration "
                     "(fastsolve check_cond=False path), w_mass=5.0, mass_target=0.6*mass_max",
        source="src/cs6_complex_bridge2.py make_model() (davinci_bridge2(n_bays=24,span=20.0,"
                "width=6.0,lam=0.25)) + src/sizing.py iterative_qubo_sizing")))

    LAM_LEVELS = np.linspace(0.05, 0.5, 8)
    dome_best_idx = [1, 6, 6, 6, 7]
    lam_dome = [float(LAM_LEVELS[dome_best_idx[i]]) for i in range(4)]
    m_dome = rf_dome2(n_sect=7, n_ring=4, span=18.5, rise=3.15, lam_per_ring=lam_dome,
                       load_per_node=-3500.0, max_group_rings=4)
    Qd, ngd, nsd, mtd = _build_f1_qubo(m_dome, w_mass=5.0, check_cond=False)
    panels.append((f"(e) dome-fixed F1\n{ngd*nsd} bits", Qd, nsd, dict(
        panel="e", n_bits=ngd * nsd, n_groups=ngd, n_sections=nsd, case_study="dome-fixed",
        description="F1 sizing QUBO on rf_dome2 frozen at the CS4 FMQA-best geometry "
                     "(seed 1, idx=[1,6,6,6,7], best compliance 1378.55), one real outer "
                     "iteration, w_mass=5.0, mass_target=0.6*mass_max",
        source="src/dome2_sizing.py make_model() (rf_dome2 n_sect=7,n_ring=4,span=18.5,"
                "rise=3.15) + src/sizing_campaign.py/src/sizing.py")))

    vault_best_idx = [5, 7, 6, 7, 7, 5, 7]
    lam_vault = [float(LAM_LEVELS[vault_best_idx[i]]) for i in range(6)]
    m_vault = rf_vault_hex2(n_rows=8, n_cols=8, span_transverse=14.0, rise=2.6,
                             lam_bands=lam_vault, load_per_node=-2200.0, n_section_groups=12)
    Qv, ngv, nsv, mtv = _build_f1_qubo(m_vault, w_mass=5.0, check_cond=False)
    panels.append((f"(f) vault-fixed F1\n{ngv*nsv} bits", Qv, nsv, dict(
        panel="f", n_bits=ngv * nsv, n_groups=ngv, n_sections=nsv, case_study="vault-fixed",
        description="F1 sizing QUBO on rf_vault_hex2 frozen at the CS-VAULT-HEX-GEO "
                     "FMQA-best geometry (seed 2, idx=[5,7,6,7,7,5,7], best compliance "
                     "713.11), one real outer iteration, w_mass=5.0, mass_target=0.6*mass_max",
        source="src/vault_hex2_sizing.py make_model() (rf_vault_hex2 n_rows=8,n_cols=8,"
                "span_transverse=14.0,rise=2.6) + src/sizing_campaign.py/src/sizing.py")))

    Qg, nbg, n_used, cat_counts_g = _build_fmqa_qubo(seed=9, cyc=30)
    panels.append((f"(g) FMQA surrogate\n(dome search, {nbg} bits)", Qg, 8, dict(
        panel="g", n_bits=nbg, case_study="CS4 (dome geometry search)",
        description=(f"real rank-8 factorization-machine-derived QUBO, refit on the first "
                      f"{n_used} real FE evaluations (20 random-init + 30 acquisition cycles) "
                      f"of the actual FMQA run, seed=9 (the only cs4_complex.json fmqa_state "
                      f"with a full saved X/y trace), cat_counts={cat_counts_g} "
                      f"(4 ring-family lambda one-hot [8 levels] + 1 global-section one-hot "
                      f"[8 levels])"),
        source="src/fmqa.py FactorizationMachine+build_fm_qubo; src/resumable_v2.py "
                "fmqa_step_cycles (seed formula reproduced exactly); "
                "results/complex/cs4_complex.json fmqa_states['9']")))

    Q7, ng7, ns7, cmin7 = _build_cs7_stress_qubo()
    panels.append((f"(h) CS7 stress-constrained\n{ng7*ns7} bits", Q7, ns7, dict(
        panel="h", n_bits=ng7 * ns7, n_groups=ng7, n_sections=ns7, case_study="CS7",
        description=("RI stress-constrained F1 sizing QUBO on CS2 geometry (square_nexorade "
                      "6,6,cell=2.5,lam=0.25) with load scaled so mid-catalog max utilization "
                      "~=0.9; one real outer iteration, w_mass=20, w_stress=500 dominant "
                      f"linear feasibility penalty, frozen-force per-group feasible section "
                      f"c_min={cmin7}"),
        source="src/cs7_stress_constrained.py build_setup/group_forces/frozen_force_cmin/"
                "sa_stress_chunk's per-iteration QUBO construction")))

    Q5, ng5, ns5 = _build_cs5_weighted_qubo(w=0.5)
    panels.append((f"(i) CS5 weighted-sum\n(w=0.5, {ng5*ns5} bits)", Q5, ns5, dict(
        panel="i", n_bits=ng5 * ns5, n_groups=ng5, n_sections=ns5, case_study="CS5",
        description="F4 weighted-sum multi-objective QUBO (w*compliance_norm + "
                     "(1-w)*mass_norm, purely linear -- no quadratic mass-target penalty), "
                     "representative weight w=0.5 (midpoint of the real 11-point W_LEVELS "
                     "sweep), one real outer iteration on square_nexorade(6,6,cell=2.5,"
                     "lam=0.25); |Q| is EXACTLY block-diagonal (only one-hot penalty blocks "
                     "nonzero, no cross-group coupling)",
        source="src/cs5_v2.py make_model()/W_LEVELS + src/sizing.py weighted_sum_qubo_sizing")))

    fig, axes = plt.subplots(3, 3, figsize=(190 * MM, 195 * MM))
    for ax, (label, Q, blk, prov) in zip(axes.flat, panels):
        _panel_qmatrix(ax, Q, blk, label)
        prov['abs_Q_max'] = float(np.abs(Q).max())
        nz = np.abs(Q)[np.abs(Q) > 0]
        prov['abs_Q_min_nonzero'] = float(nz.min()) if nz.size else 0.0

    # one-hot block-structure annotation, ONCE, on panel (b) (author
    # instruction: annotate the block structure once rather than on every
    # panel -- (b) is chosen since it is the CS2 headline F1 QUBO already
    # explained in fig_encoding_qubo's own callouts).
    axb = axes.flat[1]
    _lbl_bbox = dict(boxstyle='round,pad=0.18', fc='white', ec='none', alpha=0.88)
    axb.annotate("one-hot penalty\nblock (8 sections)", xy=(ns2 - 0.5, ns2 / 2 - 0.5),
                 xycoords='data', xytext=(ns2 + 10, ns2 * 1.6), textcoords='data',
                 fontsize=5.6, ha='left', va='center', bbox=_lbl_bbox, zorder=9,
                 arrowprops=dict(arrowstyle='->', lw=0.6, color='k', shrinkA=1.5, shrinkB=1.5))

    fig.tight_layout(h_pad=1.6, w_pad=0.9)
    savefig(fig, "fig_qubo_grid")

    provenance = {p['panel']: p for _, _, _, p in panels}
    provenance['_note'] = ("All 9 QUBO instances are rebuilt from the exact same code paths "
                            "used by the real optimization campaigns reported in the "
                            "manuscript (one real FE-solve-derived outer iteration per "
                            "panel, except (a) which is the exact 16-bit brute-force-"
                            "cross-checked table QUBO and (g) which is a real mid-run FMQA "
                            "surrogate refit from a checkpointed campaign state) -- none are "
                            "synthetic/illustrative.")
    with open(f"{RESULTS}/qubo_grid_provenance.json", "w") as f:
        json.dump(provenance, f, indent=2, default=str)
    print("saved", f"{RESULTS}/qubo_grid_provenance.json")


# --------------------------------------------------------------------------
# fig_cs1_design (COLOR) -- CS1 fan-unit brute-force global optima.
# Regenerated from reciprocal_geom_v2.fan_unit + results/cs1_fans_456.
# json, with the same decode-and-assert-reproducibility check as before --
# CONFIRMED the design optimum is unchanged (lambda=0.500, D=244.5mm) for
# EVERY n in {4,5,6}, even though the SA-QUBO success rate differs by n.
#
# REVISION (author feedback, 2026): the CS1 enumeration-validation study is
# EXTENDED from the single 4-beam fan to 4-, 5- and 6-beam fan_unit rotors
# -- this figure is now a 3-panel layout, (a) 4-beam, (b) 5-beam, (c)
# 6-beam, each at ITS OWN optimum (all three independently verified to
# land on lambda=0.500, D=244.5mm -- see cs1_fans_456.json's
# comparison_summary.optimum_is_lambda_0500_D_244p5_for_all_n). Straight
# members, filled-dot beam-to-beam connections, discrete per-section
# DIAMETER_COLORS palette, and a per-panel "all N members D=... mm"
# size read-out are all kept identical to the prior single-panel figure's
# conventions, just repeated per panel; a single shared discrete colorbar
# serves all three panels (all three land on the SAME catalog section,
# D=244.5mm, so all three panels are the same flat color, which is itself
# part of the finding worth showing side by side).
# --------------------------------------------------------------------------
def fig_cs1_design():
    from reciprocal_geom_v2 import fan_unit
    from sections import CHS_CATALOG

    d = json.load(open(f"{RESULTS}/cs1_fans_456.json"))
    n_list = (4, 5, 6)
    panel_letters = ('a', 'b', 'c')

    fig = plt.figure(figsize=(190 * MM, 76 * MM))
    axes = [fig.add_subplot(1, 3, k + 1, projection='3d') for k in range(3)]

    for ax, nb, letter in zip(axes, n_list, panel_letters):
        r = d[f"fan{nb}"]
        lam_idx = r['global_opt_lambda_idx']
        sec_idx = r['global_opt_section_idx']
        lam = r['lambda_levels'][lam_idx]
        print(f"[fig_cs1_design] fan{nb}: decoded optimum lambda_idx={lam_idx} "
              f"(lambda={lam:.3f}), section_idx={sec_idx} "
              f"(D={CHS_CATALOG[sec_idx]['D_mm']}mm), "
              f"compliance={r['global_opt_compliance']:.6e}")
        assert abs(lam - 0.5) < 1e-9 and CHS_CATALOG[sec_idx]['D_mm'] == 244.5, \
            f"CS1 fan{nb} global optimum no longer matches expected lambda=0.500, D=244.5mm"

        model = fan_unit(nb, R=3.0, lam=lam, single_group=True)
        nodes = np.array(model.nodes)
        mids = _link_midpoints(model, nodes)

        color = DIAMETER_COLORS[sec_idx]
        D = CHS_CATALOG[sec_idx]['D_mm']
        lw = 0.6 + 2.2 * (D / CHS_CATALOG[-1]['D_mm'])
        for m in _member_lines(model):
            ax.plot([m['p1'][0], m['p2'][0]], [m['p1'][1], m['p2'][1]],
                    [m['p1'][2], m['p2'][2]], color=color, lw=lw)
        if len(mids):
            ax.scatter(mids[:, 0], mids[:, 1], mids[:, 2], color=COLORS['accent'],
                       marker='o', s=10, zorder=6, depthshade=False, linewidth=0)
        tpos = true_node_positions(model)
        sup = tpos[list(model.support_idx)]
        ax.scatter(sup[:, 0], sup[:, 1], sup[:, 2], color='k', marker='^',
                   s=16, zorder=10, depthshade=False)
        ax.set_title(f"({letter}) {nb}-beam fan (global optimum)", fontsize=8, pad=13)
        ax.set_box_aspect([1, 1, 0.16])
        ax.view_init(elev=24, azim=-60)
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
        ax.grid(False)
        for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
            pane.pane.set_alpha(0.0)
        # In-panel numeric size read-out -- axes-fraction 2D overlay (ax.
        # text2D), stays fixed/legible regardless of the 3D view angle.
        n_mem = _n_physical_members(model)
        ax.text2D(0.03, 0.94, f"all {n_mem} members\nD = {D:g} mm", transform=ax.transAxes,
                  fontsize=6.5, ha='left', va='top',
                  bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.85))

    _diameter_colorbar(fig, axes, _DIAMETER_CMAP,
                        [s['D_mm'] for s in CHS_CATALOG],
                        "CHS outside diameter D [mm]\n"
                        "dots: beam-to-beam connection points\n"
                        "(all three brute-force global optima: $\\lambda$=0.500, D=244.5mm)",
                        shrink=0.7, label_fontsize=6.5)
    savefig(fig, "fig_cs1_design")


# --------------------------------------------------------------------------
# fig_cs2_design (COLOR) -- square_nexorade before/after sizing maps, PLAN
# view (the geometry is exactly planar, z=0 everywhere, so plan = a plain
# 2D plot, not a faked-flat 3D view). Best v2 design decoded via
# r6_uls_v2.recover_cs2(), which deterministically reruns the exact seeded
# F1 sizing loop and asserts the recomputed best-so-far compliance matches
# the value stored in results/cs2_v2.json (reproducibility_match).
# --------------------------------------------------------------------------
def fig_cs2_design():
    from sections import CHS_CATALOG
    from r6_uls_v2 import recover_cs2

    model, gs, load_per_node, seed, match = recover_cs2()
    assert match, "CS2-v2 best-design recovery failed reproducibility check"
    idx_final = [CHS_CATALOG.index(gs[k]) for k in range(model.n_groups)]
    print(f"[fig_cs2_design] best seed={seed} decoded idx_final={idx_final}")

    fig, axes = plt.subplots(1, 2, figsize=(180 * MM, 92 * MM))
    cmap = _DIAMETER_CMAP

    # NOTE: panel titles trimmed to bare (a)/(b) identifiers (2026 study)
    # -- the removed detail ("mid-catalog, plan" / "mass-target 60%, plan")
    # now given in the manuscript caption instead (see
    # latex/figure_title_migration.md).
    for ax, idx_vec, title in ((axes[0], [3] * model.n_groups, "(a) Uniform initial design"),
                                (axes[1], idx_final, "(b) SA-QUBO optimized design")):
        for m in _member_lines_by_member(model):
            sec_idx = idx_vec[m['group']]
            color = DIAMETER_COLORS[sec_idx]
            D = CHS_CATALOG[sec_idx]['D_mm']
            lw = 0.5 + 1.8 * (D / CHS_CATALOG[-1]['D_mm'])
            ax.plot([m['p1'][0], m['p2'][0]], [m['p1'][1], m['p2'][1]], color=color, lw=lw)
        nodes = np.array(model.nodes)
        mids = _link_midpoints(model, nodes)
        if len(mids):
            ax.scatter(mids[:, 0], mids[:, 1], color=COLORS['accent'], marker='o',
                       s=5, zorder=6, linewidth=0)
        tpos = true_node_positions(model)
        sup = tpos[list(model.support_idx)]
        ax.scatter(sup[:, 0], sup[:, 1], color='k', marker='^', s=13, zorder=10)
        ax.set_title(title, fontsize=8, pad=13)
        ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        # Numeric size read-out: the grillage is too dense for per-member
        # labels, but the colorbar ticks alone don't say how many PHYSICAL
        # MEMBERS got each size. REVISION (author feedback, 2026: "don't
        # show counts per group; show counts of ALL members in the
        # structure per size") -- (a) states the total member count at its
        # single uniform D, (b) uses _size_count_legend's whole-structure
        # physical-member-count-per-size convention (dominant/majority-
        # length portion for fused members spanning two groups; see that
        # function's docstring) -- both placed in an empty corner outside
        # the rotated-square footprint.
        if idx_vec is idx_final:
            txt = _size_count_legend(model, idx_vec, CHS_CATALOG)
        else:
            txt = f"all {_n_physical_members(model)} members D = {CHS_CATALOG[idx_vec[0]]['D_mm']:g} mm"
        ax.text(0.5, 1.0, txt, transform=ax.transAxes, fontsize=6.0,
                ha='center', va='bottom')

    _diameter_colorbar(fig, axes, cmap, [s['D_mm'] for s in CHS_CATALOG],
                        "CHS outside diameter D [mm]  --  dots: beam-to-beam connection points")
    savefig(fig, "fig_cs2_design")


# --------------------------------------------------------------------------
# fig_cs3_design (COLOR) -- hex_nexorade before/after sizing maps, PLAN view
# (hex_nexorade is exactly planar, z=0 everywhere, same rationale as
# fig_cs2_design). Best v2 design decoded via r6_uls_v2.recover_cs3(),
# which deterministically reruns the exact seeded SA-QUBO sizing loop for
# the best-compliance seed and asserts the recomputed best-so-far
# compliance matches the value stored in results/cs3_v2.json
# (reproducibility_match) -- same pattern as fig_cs2_design.
# --------------------------------------------------------------------------
def fig_cs3_design():
    from sections import CHS_CATALOG
    from r6_uls_v2 import recover_cs3

    model, gs, load_per_node, seed, match = recover_cs3()
    assert match, "CS3-v2 best-design recovery failed reproducibility check"
    idx_final = [CHS_CATALOG.index(gs[k]) for k in range(model.n_groups)]
    print(f"[fig_cs3_design] best seed={seed} decoded idx_final={idx_final}")

    fig, axes = plt.subplots(1, 2, figsize=(180 * MM, 92 * MM))
    cmap = _DIAMETER_CMAP

    # NOTE: panel titles trimmed to bare (a)/(b) identifiers (2026 study)
    # -- the removed detail ("mid-catalog, plan" / "mass-target 60%, plan")
    # now given in the manuscript caption instead (see
    # latex/figure_title_migration.md).
    for ax, idx_vec, title in ((axes[0], [3] * model.n_groups, "(a) Uniform initial design"),
                                (axes[1], idx_final, "(b) SA-QUBO optimized design")):
        for m in _member_lines_by_member(model):
            sec_idx = idx_vec[m['group']]
            color = DIAMETER_COLORS[sec_idx]
            D = CHS_CATALOG[sec_idx]['D_mm']
            lw = 0.5 + 1.8 * (D / CHS_CATALOG[-1]['D_mm'])
            ax.plot([m['p1'][0], m['p2'][0]], [m['p1'][1], m['p2'][1]], color=color, lw=lw)
        nodes = np.array(model.nodes)
        mids = _link_midpoints(model, nodes)
        if len(mids):
            ax.scatter(mids[:, 0], mids[:, 1], color=COLORS['accent'], marker='o',
                       s=5, zorder=6, linewidth=0)
        tpos = true_node_positions(model)
        sup = tpos[list(model.support_idx)]
        ax.scatter(sup[:, 0], sup[:, 1], color='k', marker='^', s=13, zorder=10)
        ax.set_title(title, fontsize=8, pad=13)
        ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        # Numeric size read-out (see fig_cs2_design for rationale): (a)
        # total-member-count uniform-D label, (b) whole-structure
        # physical-member-count-per-size legend (_size_count_legend).
        if idx_vec is idx_final:
            txt = _size_count_legend(model, idx_vec, CHS_CATALOG)
        else:
            txt = f"all {_n_physical_members(model)} members D = {CHS_CATALOG[idx_vec[0]]['D_mm']:g} mm"
        ax.text(0.5, 1.0, txt, transform=ax.transAxes, fontsize=6.0,
                ha='center', va='bottom')

    _diameter_colorbar(fig, axes, cmap, [s['D_mm'] for s in CHS_CATALOG],
                        "CHS outside diameter D [mm]  --  dots: beam-to-beam connection points")
    savefig(fig, "fig_cs3_design")


# --------------------------------------------------------------------------
# fig_cs4_design (COLOR, 3D) -- REVISION (complex dome/bridge upgrade,
# 2026): rf_dome_complex initial/reference vs FMQA-optimum design (rings=4,
# span=18.5m, rise=3.15m, 5 concentric ring families -- the same denser
# geometry now used by the CS4-complex campaign, results/complex/
# cs4_complex.json), replacing the earlier rf_dome (rings=3) version. As
# before, CS4's black-box design vector ALSO controls the per-ring-family
# engagement ratio lambda, which changes the dome's physical geometry
# (member eccentricities/curvature), so both panels are independently
# regenerated rf_dome2() instances -- not two colourings of one
# fixed mesh. The best FMQA design (lambda per ring family + one global
# section) is recovered by dome2_posthoc.recover_dome2_geo_best(), which
# DETERMINISTICALLY rebuilds the exact best-seed/best-eval design and
# asserts the recomputed compliance matches both the stored final_best AND
# the y_data entry it was selected from (reproducibility_match, asserted
# below) -- the same "recover, don't just decode+trust" discipline used by
# r6_uls_v2.recover_cs2/cs3/cs6 for the un-changed typologies.
# --------------------------------------------------------------------------
def fig_cs4_design():
    from reciprocal_geom_v2 import rf_dome2
    from sections import CHS_CATALOG
    # REVISION (dome redesign, 2026): recovery switched from
    # cs_complex_uls.recover_cs4_complex (rf_dome_complex) to
    # dome2_posthoc.recover_dome2_geo_best (rf_dome2, circular springing ring)
    # -- same argmin-over-all-seeds FMQA convention, same cs4_complex.json.
    from dome2_posthoc import recover_dome2_geo_best

    model_opt, gs_opt, load_per_node, seed, match, span, meta = recover_dome2_geo_best()
    assert match, "CS4 (rf_dome2) best-design recovery failed reproducibility check"

    lam_per_ring_opt = meta['lam_per_ring']
    sec_idx_opt = meta['section_idx']
    print(f"[fig_cs4_design] CS4-complex best seed={seed} "
          f"stored_best={meta['stored_best']:.4f} recomputed={meta['recomputed']:.4f} "
          f"lam_per_ring={lam_per_ring_opt} section_idx_global={sec_idx_opt} "
          f"(D={CHS_CATALOG[sec_idx_opt]['D_mm']}mm)")

    N_SECT, N_RING, SPAN, RISE = 7, 4, 18.5, 3.15  # rf_dome2 campaign settings
    LAM_LEVELS = np.linspace(0.05, 0.5, 8)
    lam_ref = float(LAM_LEVELS[3])  # mid-catalog reference lambda level
    sec_ref = 3                     # mid-catalog reference section
    model_ref = rf_dome2(n_sect=N_SECT, n_ring=N_RING, span=SPAN, rise=RISE, lam=lam_ref,
                         max_group_rings=4)

    fig, axes = plt.subplots(1, 2, figsize=(180 * MM, 100 * MM), subplot_kw={'projection': '3d'})
    cmap = _DIAMETER_CMAP

    # NOTE: panel titles trimmed to bare (a)/(b) identifiers (2026 study)
    # -- the removed detail (lambda=lam_ref all rings, mid-catalog D for
    # (a); lambda per ring + optimum D for (b)) now given in the manuscript
    # caption instead (see latex/figure_title_migration.md).
    panels = (
        (axes[0], model_ref, sec_ref, "(a) Initial/reference design"),
        (axes[1], model_opt, sec_idx_opt, "(b) FMQA optimum design"),
    )
    for ax, model, sec_idx, title in panels:
        color = DIAMETER_COLORS[sec_idx]
        D = CHS_CATALOG[sec_idx]['D_mm']
        # Denser rf_dome2 mesh (182 physical members vs rf_dome_complex's
        # 152) -- thinner member lines + smaller
        # connection-dot/support markers than the original fig_cs4_design so
        # the multi-ring lattice stays legible instead of visually merging
        # into a solid cap.
        lw = 0.35 + 1.35 * (D / CHS_CATALOG[-1]['D_mm'])
        for m in _member_lines(model):
            ax.plot([m['p1'][0], m['p2'][0]], [m['p1'][1], m['p2'][1]],
                    [m['p1'][2], m['p2'][2]], color=color, lw=lw)
        nodes = np.array(model.nodes)
        mids = _link_midpoints(model, nodes)
        if len(mids):
            ax.scatter(mids[:, 0], mids[:, 1], mids[:, 2], color=COLORS['accent'],
                       marker='o', s=2.5, zorder=6, depthshade=False, linewidth=0)
        tpos = true_node_positions(model)
        sup = tpos[list(model.support_idx)]
        ax.scatter(sup[:, 0], sup[:, 1], sup[:, 2], color='k', marker='^',
                   s=10, zorder=10, depthshade=False)
        ax.set_title(title, fontsize=7.4)
        ax.set_box_aspect([1, 1, 0.30])
        ax.view_init(elev=26, azim=-60)
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
        ax.grid(False)
        for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
            pane.pane.set_alpha(0.0)
        # Each panel is a SINGLE global section (lambda varies per ring, D
        # does not), so one numeric label per panel fully captures the
        # size -- fixed 2D overlay text, clear of the mesh below it. States
        # the actual total physical-member count ("all N members"), per
        # the whole-structure member-count convention (see
        # _size_count_legend docstring).
        n_mem = _n_physical_members(model)
        ax.text2D(0.03, 0.94, f"all {n_mem} members D = {D:g} mm", transform=ax.transAxes,
                  fontsize=6.5, ha='left', va='top',
                  bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.85))

    _diameter_colorbar(fig, axes, cmap, [s['D_mm'] for s in CHS_CATALOG],
                        "CHS outside diameter D [mm]  --  dots: beam-to-beam connection points")
    savefig(fig, "fig_cs4_design")


# --------------------------------------------------------------------------
# fig_cs6_design (COLOR, 3D) -- REVISION (complex dome/bridge upgrade,
# 2026): davinci_bridge_complex before/after sizing maps. REVISION (author
# feedback, 2026: "the color doesn't make sense" for fig_cs6_design) -- this
# figure used to be an ELEVATION (x-z) projection, which for
# davinci_bridge_complex is actively misleading: its 3 rails sit at
# y in {-width/2, 0, +width/2} but z_of(x) does NOT depend on y (see
# reciprocal_geom_v2.davinci_bridge_complex), so all 3 rails' longitudinal
# members project onto EXACTLY the same (x,z) curve. Per-member colors were
# computed correctly (color = idx_vec[m['group']], verified numerically),
# but with 3 rails drawn on top of one another in a flat 2D axes, only the
# LAST-PAINTED rail's color was ever visible at a given (x,z) -- whichever
# of 3 (possibly differently-sized) members happened to be drawn last
# silently hid the other 2 underneath it, so the visible colors did not
# reliably correspond to "the design" a reader would infer. Fixed (per
# author request) by switching to a genuine 3D view -- the same convention
# as fig_cs4_design/fig_typologies panel (e) -- so the 3 rails are
# genuinely separated in y and every member's own true color is visible
# without occlusion. Best design decoded via
# cs_complex_uls.recover_cs6_complex(), same seeded-rerun +
# reproducibility_match pattern as r6_uls_v2.recover_cs2/cs3 -- here it
# deterministically RE-RUNS the exact seeded SA-QUBO F1 sizing loop on
# davinci_bridge_complex (16 groups, 58 physical members, no fused/mixed-
# group members -- see _member_lines docstring) rather than looking up a
# stored index vector. Colors are keyed off _member_lines_by_member()'s
# per-PHYSICAL-member dominant group (REVISION, author feedback 2026:
# "member size mentioned is not the same as the color displayed" -- see
# _member_lines_by_member docstring), the same convention as
# fig_cs2_design/fig_cs3_design/the sizing-map figures, so drawn colors and
# the _size_count_legend legend below agree exactly by construction even
# though CS6 itself has no fused members to begin with. Per-member numeric
# callouts (used in the old elevation
# version) are DROPPED here: matplotlib's Axes3D has no reliable
# screen-space leader-line/offset annotation (an `ax.annotate` offset is a
# 2D-axes-only feature; a 3D data-space label would drift under rotation
# and collide with the rotated geometry), so -- exactly as this figure's
# own docstring anticipated -- the discrete colorbar plus a compact
# whole-structure member-count-per-size text-overlay read-out (same
# _size_count_legend helper used by fig_cs2_design/fig_cs3_design) carries
# the numeric information instead.
# --------------------------------------------------------------------------
def fig_cs6_design():
    # REVISION (bridge redesign, 2026): davinci_bridge_complex ->
    # davinci_bridge2 (load-path-corrected 3-rail, 24-bay bridge; fixes
    # the strain-energy-floor defect diagnosed in
    # results/bridge_diagnosis.json -- see
    # reciprocal_geom_v2.davinci_bridge2 and
    # results/bridge2_verification.json). Best design now recovered via
    # bridge2_posthoc.recover_cs6_bridge2() (same deterministic seeded-
    # rerun + reproducibility_match pattern as the old
    # cs_complex_uls.recover_cs6_complex it replaces); 118 physical members
    # / 16 groups (vs the old bridge's 58 members) on the SAME davinci_
    # bridge2 geometry used throughout Section 39 of complex_summary.md.
    from sections import CHS_CATALOG
    from bridge2_posthoc import recover_cs6_bridge2

    model, gs, load_per_node, seed, match, span, meta = recover_cs6_bridge2()
    assert match, "CS6 bridge2 best-design recovery failed reproducibility check"
    idx_final = [CHS_CATALOG.index(gs[k]) for k in range(model.n_groups)]
    print(f"[fig_cs6_design] CS6-complex best seed={seed} "
          f"stored_best={meta['stored_best']:.4f} recomputed={meta['recomputed']:.4f} "
          f"decoded idx_final={idx_final}")

    fig, axes = plt.subplots(1, 2, figsize=(180 * MM, 96 * MM),
                              subplot_kw={'projection': '3d'})
    cmap = _DIAMETER_CMAP

    # NOTE: panel titles trimmed to bare (a)/(b) identifiers (2026 study)
    # -- the removed detail ("mid-catalog, elevation" / "mass-target 60%,
    # elevation") now given in the manuscript caption instead (see
    # latex/figure_title_migration.md).
    for ax, idx_vec, title in ((axes[0], [3] * model.n_groups, "(a) Uniform initial design"),
                                (axes[1], idx_final, "(b) SA-QUBO optimized design")):
        for m in _member_lines_by_member(model):
            sec_idx = idx_vec[m['group']]
            color = DIAMETER_COLORS[sec_idx]
            D = CHS_CATALOG[sec_idx]['D_mm']
            # same base/scale convention as the old elevation version
            # (0.4+1.4x) -- still thin enough that the now visually
            # SEPARATED 3 rails + rung segments stay legible individually.
            lw = 0.4 + 1.4 * (D / CHS_CATALOG[-1]['D_mm'])
            ax.plot([m['p1'][0], m['p2'][0]], [m['p1'][1], m['p2'][1]],
                    [m['p1'][2], m['p2'][2]], color=color, lw=lw, zorder=3)
        nodes = np.array(model.nodes)
        mids = _link_midpoints(model, nodes)
        if len(mids):
            ax.scatter(mids[:, 0], mids[:, 1], mids[:, 2], color=COLORS['accent'],
                       marker='o', s=4, zorder=6, depthshade=False, linewidth=0)
        tpos = true_node_positions(model)
        sup = tpos[list(model.support_idx)]
        ax.scatter(sup[:, 0], sup[:, 1], sup[:, 2], color='k', marker='^',
                   s=14, zorder=10, depthshade=False)
        ax.set_title(title, fontsize=8, pad=2)
        # same view convention as the davinci_bridge_complex panel of
        # fig_typologies (elev=20, azim=-58, zaspect=0.20) -- already
        # verified (VIEW step, that figure) to keep the 3 rails visually
        # separated and the arch profile legible at this span:width:rise
        # ratio (20:6:2.4 m).
        ax.set_box_aspect([1, 1, 0.20])
        ax.view_init(elev=20, azim=-58)
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
        ax.grid(False)
        for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
            pane.pane.set_alpha(0.0)

        # Numeric size read-out -- NOT per-member callouts in 3D (see
        # figure-level docstring above): (a) total-member-count uniform-D
        # label, (b) whole-structure physical-member-count-per-size legend
        # (same helper as fig_cs2_design/fig_cs3_design), both as a fixed
        # 2D text overlay (ax.text2D) so they stay legible/un-rotated
        # regardless of the 3D view angle.
        if idx_vec is idx_final:
            txt = _size_count_legend(model, idx_vec, CHS_CATALOG)
        else:
            txt = f"all {_n_physical_members(model)} members D = {CHS_CATALOG[idx_vec[0]]['D_mm']:g} mm"
        ax.text2D(0.5, 0.93, txt, transform=ax.transAxes, fontsize=5.8,
                  ha='center', va='top',
                  bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.85))

    _diameter_colorbar(fig, axes, cmap, [s['D_mm'] for s in CHS_CATALOG],
                        "CHS outside diameter D [mm]  --  dots: beam-to-beam connection points")
    savefig(fig, "fig_cs6_design")


# --------------------------------------------------------------------------
# VAULT TYPOLOGY FIGURES (NEW, 2026) -- the two-stage GEOMETRY-FIND (CS-
# VAULT-GEO, FMQA/GA/random on rf_vault's per-transverse-band engagement
# ratio lambda + one global CHS section) -> SECTION-FIND (dome_sizing/
# vault_sizing, SA-QUBO/GA/random per-group catalog sizing on the FROZEN
# best geometry from stage 1) pipeline. All decoded/recovered designs below
# go through vault_posthoc.py's own recovery functions (recover_vault_geo_
# best, recover_sizing_best + its two frozen-geometry factories), which
# already assert `reproducibility_match` by re-solving the recovered design
# and checking the recomputed compliance against the stored value -- this
# script re-asserts that flag rather than re-deriving the recovery logic a
# second time (same discipline as fig_cs2_design/fig_cs4_design/
# fig_cs6_design's use of r6_uls_v2.py/cs_complex_uls.py).
#
# REVISION (hex-pattern vault upgrade, 2026): every VAULT figure below
# (fig_vault_geo_design, fig_vault_geo_conv, fig_vault_sizing_design, and
# panel (b) of fig_sizing_fixed_conv) is switched from the SQUARE-pattern
# `rf_vault`/CS-VAULT-GEO/vault_geo.json/vault_sizing.json/vault_posthoc.py
# pipeline to the HEXAGONAL/TRIANGULAR-pattern `rf_vault_hex`/
# CS-VAULT-HEX-GEO/vault_hex_geo.json/vault_hex_sizing.json/
# vault_hex_posthoc.py pipeline (same two-stage GEOMETRY-FIND ->
# SECTION-FIND structure, same 10-seed geometry search / 5-seed sizing
# search protocol, just re-run on the new generator -- see
# vault_hex_posthoc.py's own module docstring). The dome figures
# (fig_dome_sizing_design, panel (a) of fig_sizing_fixed_conv) are
# UNAFFECTED and still use rf_dome_complex/dome_sizing.json/
# vault_posthoc.py exactly as before. The old square-pattern vault
# data/JSONs are left on disk unchanged (vault_geo.json, vault_sizing.json,
# vault_posthoc.json) -- only the figures switch to the new hex pipeline,
# nothing is deleted.
# --------------------------------------------------------------------------
_VAULT_LAM_LEVELS = np.linspace(0.05, 0.5, 8)  # matches vault_hex_posthoc.LAM_LEVELS
# REVISION (vault redesign, 2026): rf_vault_hex(n_rows=7, length=17.0) ->
# rf_vault_hex2(n_rows=8, n_cols=8), which sets its own length from n_cols.
_VAULT_HEX_KW = dict(n_rows=8, n_cols=8, span_transverse=14.0, rise=2.6)
_VAULT_N_BANDS = 6          # rf_vault_hex2 lam_bands length (was 5 for hex1)
_VAULT_N_SEC_GROUPS = 12    # rf_vault_hex2 section groups (was 10 for hex1)
_VAULT_VIEW = dict(elev=28, azim=-40, zaspect=0.35)  # see fig_typologies panel (e)


# --------------------------------------------------------------------------
# fig_vault_geo_design (NEW, COLOR, 3D) -- CS-VAULT-HEX-GEO initial/reference
# rf_vault_hex geometry (mid lambda level for all 5 transverse bands, mid-
# catalog global section) vs the FMQA-best geometry recovered by
# vault_hex_posthoc.recover_vault_hex_geo_best() (seed/idx re-derived from
# the stored FMQA y_data/idx_data and re-solved; reproducibility_match
# asserted below). Exactly like fig_cs4_design's rf_dome_complex panels,
# CS-VAULT-HEX-GEO's black-box vector controls PER-BAND lambda, which
# changes the vault's physical geometry (member eccentricities/curvature
# per transverse band) -- so both panels are independently regenerated
# rf_vault_hex instances, not two colourings of one fixed mesh. Both panels
# use a SINGLE global CHS section (the black-box search variable is one
# global section index, not per-group), so -- like fig_cs4_design -- each
# panel gets one numeric "D = ... mm" read-out rather than a per-group
# size-count legend. View convention (elev=28, azim=-40, zaspect=0.35) is
# the SAME one chosen for fig_typologies panel (e) (see that figure's
# docstring for the view-angle search that keeps BOTH the vault's
# transverse arch AND its woven hexagonal pattern legible at once), applied
# consistently across every vault figure in this module.
#
# REVISION (hex-pattern vault upgrade, 2026): switched from the SQUARE-
# pattern rf_vault/vault_geo.json/vault_posthoc.recover_vault_geo_best() to
# the HEXAGONAL-pattern rf_vault_hex/vault_hex_geo.json/vault_hex_posthoc.
# recover_vault_hex_geo_best() -- see the "VAULT TYPOLOGY FIGURES" header
# comment above for the scope of this upgrade.
# --------------------------------------------------------------------------
def fig_vault_geo_design():
    from reciprocal_geom_v2 import rf_vault_hex2
    from sections import CHS_CATALOG
    from vault_hex_posthoc import recover_vault_hex_geo_best

    model_opt, gs_opt, load_per_node, seed, match, span, meta = recover_vault_hex_geo_best()
    assert match, "CS-VAULT-HEX-GEO best-design recovery failed reproducibility check"

    lam_bands_opt = meta['lam_bands']
    sec_idx_opt = meta['section_idx']
    print(f"[fig_vault_geo_design] CS-VAULT-HEX-GEO best seed={seed} "
          f"stored_best={meta['stored_best']:.4f} recomputed={meta['recomputed']:.4f} "
          f"lam_bands={[round(l, 3) for l in lam_bands_opt]} section_idx={sec_idx_opt} "
          f"(D={CHS_CATALOG[sec_idx_opt]['D_mm']}mm)")

    lam_ref = float(_VAULT_LAM_LEVELS[3])  # mid-catalog reference lambda (all bands)
    sec_ref = 3                            # mid-catalog reference section
    model_ref = rf_vault_hex2(lam_bands=[lam_ref] * _VAULT_N_BANDS, load_per_node=-2200.0,
                              n_section_groups=_VAULT_N_SEC_GROUPS, **_VAULT_HEX_KW)

    fig, axes = plt.subplots(1, 2, figsize=(180 * MM, 92 * MM), subplot_kw={'projection': '3d'})
    cmap = _DIAMETER_CMAP

    panels = (
        (axes[0], model_ref, sec_ref, "(a) Initial/reference geometry"),
        (axes[1], model_opt, sec_idx_opt, "(b) FMQA-best geometry"),
    )
    for ax, model, sec_idx, title in panels:
        color = DIAMETER_COLORS[sec_idx]
        D = CHS_CATALOG[sec_idx]['D_mm']
        lw = 0.35 + 1.35 * (D / CHS_CATALOG[-1]['D_mm'])
        for m in _member_lines(model):
            ax.plot([m['p1'][0], m['p2'][0]], [m['p1'][1], m['p2'][1]],
                    [m['p1'][2], m['p2'][2]], color=color, lw=lw)
        nodes = np.array(model.nodes)
        mids = _link_midpoints(model, nodes)
        if len(mids):
            ax.scatter(mids[:, 0], mids[:, 1], mids[:, 2], color=COLORS['accent'],
                       marker='o', s=3, zorder=6, depthshade=False, linewidth=0)
        tpos = true_node_positions(model)
        sup = tpos[list(model.support_idx)]
        ax.scatter(sup[:, 0], sup[:, 1], sup[:, 2], color='k', marker='^',
                   s=10, zorder=10, depthshade=False)
        ax.set_title(title, fontsize=7.6)
        ax.set_box_aspect([1, 1, _VAULT_VIEW['zaspect']])
        ax.view_init(elev=_VAULT_VIEW['elev'], azim=_VAULT_VIEW['azim'])
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
        ax.grid(False)
        for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
            pane.pane.set_alpha(0.0)
        # Single-line numeric size read-out directly under each panel's own
        # title (both panels use one global section, so one D value fully
        # captures each panel's sizing) -- fixed 2D text overlay so it stays
        # legible/un-rotated regardless of the 3D view angle. States the
        # actual total physical-member count ("all N members").
        n_mem = _n_physical_members(model)
        ax.text2D(0.5, 0.90, f"all {n_mem} members D = {D:g} mm", transform=ax.transAxes,
                  fontsize=6.5, ha='center', va='top',
                  bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.85))

    _diameter_colorbar(fig, axes, cmap, [s['D_mm'] for s in CHS_CATALOG],
                        "CHS outside diameter D [mm]  --  dots: beam-to-beam connection points")
    savefig(fig, "fig_vault_geo_design")


# --------------------------------------------------------------------------
# fig_vault_geo_conv (NEW, B/W) -- CS-VAULT-HEX-GEO best-so-far convergence,
# FMQA vs GA vs random search, 10 seeds/method, results/complex/
# vault_hex_geo.json ('best_history' per-seed field -- same shape/convention
# as fig_cs4_fmqa's cs4_complex.json). REVISION (hex-pattern vault upgrade,
# 2026): switched from vault_geo.json (square pattern) to vault_hex_geo.json
# (hexagonal pattern) -- same 10-seed/method protocol, same JSON shape.
# --------------------------------------------------------------------------
def fig_vault_geo_conv():
    d = json.load(open(f"{RESULTS}/complex/vault_hex_geo.json"))
    fig, ax = plt.subplots(figsize=(90 * MM, 70 * MM))
    labels = {'fmqa': 'FMQA (F2)', 'ga': 'GA', 'random': 'Random search'}
    _all_mean = []
    for i, method in enumerate(('fmqa', 'ga', 'random')):
        st = _BW_SERIES[i]
        runs = _runs(d['methods'][method])
        hist = [r['best_history'] for r in runs]
        mean, std, n = _mean_std_bands(hist)
        _all_mean.append(mean)
        x = np.arange(1, n + 1)
        ax.plot(x, mean, color=st['color'], ls=st['ls'], marker=st['marker'],
                markevery=_markevery(n), ms=3.5, mfc='white', mew=0.7, label=labels[method])
        ax.fill_between(x, mean - std, mean + std, color=st['color'], alpha=0.16, linewidth=0)
    ax.set_xlabel("real FE evaluations")
    ax.set_ylabel("compliance (best-so-far)")
    ax.set_yscale('log')
    _clamp_log_ylim(ax, _all_mean)
    # NOTE: no in-figure title (2026 convention) -- "CS-VAULT-HEX-GEO
    # rf_vault_hex geometry search: FMQA vs baselines (10 seeds)" given in
    # the manuscript caption instead.
    ax.legend(loc='upper right')
    fig.tight_layout()
    savefig(fig, "fig_vault_geo_conv")


# --------------------------------------------------------------------------
# fig_dome_sizing_design / fig_vault_sizing_design (NEW, COLOR, 3D) --
# SECOND stage of the two-stage pipeline: F1 (SA-QUBO) per-group catalog
# sizing on the FROZEN stage-1-best geometry (rf_dome_complex's own
# CS4-complex FMQA-best geometry for dome_sizing.json; rf_vault_hex's
# CS-VAULT-HEX-GEO FMQA-best geometry for vault_hex_sizing.json -- BOTH
# geometries themselves fixed, only per-group sections change between
# (a)/(b)). Before = UNIFORM mid-catalog section on every group; after =
# the SA-QUBO best per-group design recovered by a `recover_fn()` callable
# (re-reads the best-iteration idx from the run's own already-stored
# idx_history/compliance_history and re-solves; reproducibility_match
# asserted below) -- vault_posthoc.recover_sizing_best(json_name,
# frozen_geometry_fn) for the dome, vault_hex_posthoc.
# recover_vault_hex_sizing_best() for the vault (REVISION, hex-pattern
# vault upgrade 2026: the two recovery functions now have different call
# signatures -- vault_hex_posthoc's is dedicated/zero-arg rather than the
# generic (json_name, frozen_geometry_fn) pair -- so this shared
# implementation takes an already-bound zero-arg `recover_fn` from its
# caller instead of calling vault_posthoc.recover_sizing_best directly).
# Per-member colors use _member_lines_by_member() (REVISION, author
# feedback 2026: "member size mentioned is not the same as the color
# displayed") -- colour keyed off each PHYSICAL member's own DOMINANT-length
# raw sub-element group, one flat color per member, not one color per raw
# sub-segment -- the same convention as fig_cs2_design/fig_cs3_design/
# fig_cs6_design, so the colours drawn here and the _size_count_legend
# member-count-per-size legend below agree exactly by construction, even for
# geometries with interior-fused members (both rf_dome_complex and
# rf_vault_hex have them), rendered in 3D (per-member callouts dropped, "D
# per group" text read-out instead) exactly like fig_cs4_design/
# fig_cs6_design.
# --------------------------------------------------------------------------
def _fig_sizing_design_impl(recover_fn, view, outname, label):
    from sections import CHS_CATALOG

    model, gs, load_per_node, seed, match, span, meta = recover_fn()
    assert match, f"{label} best-design recovery failed reproducibility check"
    idx_final = [CHS_CATALOG.index(gs[k]) for k in range(model.n_groups)]
    print(f"[{outname}] {label} best seed={seed} "
          f"stored_best={meta['stored_best']:.4f} recomputed={meta['recomputed']:.4f} "
          f"decoded idx_final={idx_final}")

    fig, axes = plt.subplots(1, 2, figsize=(180 * MM, 96 * MM), subplot_kw={'projection': '3d'})
    cmap = _DIAMETER_CMAP

    for ax, idx_vec, title in ((axes[0], [3] * model.n_groups, "(a) Uniform initial design"),
                                (axes[1], idx_final, "(b) SA-QUBO optimized design")):
        for m in _member_lines_by_member(model):
            sec_idx = idx_vec[m['group']]
            color = DIAMETER_COLORS[sec_idx]
            D = CHS_CATALOG[sec_idx]['D_mm']
            lw = 0.35 + 1.35 * (D / CHS_CATALOG[-1]['D_mm'])
            ax.plot([m['p1'][0], m['p2'][0]], [m['p1'][1], m['p2'][1]],
                    [m['p1'][2], m['p2'][2]], color=color, lw=lw, zorder=3)
        nodes = np.array(model.nodes)
        mids = _link_midpoints(model, nodes)
        if len(mids):
            ax.scatter(mids[:, 0], mids[:, 1], mids[:, 2], color=COLORS['accent'],
                       marker='o', s=3, zorder=6, depthshade=False, linewidth=0)
        tpos = true_node_positions(model)
        sup = tpos[list(model.support_idx)]
        ax.scatter(sup[:, 0], sup[:, 1], sup[:, 2], color='k', marker='^',
                   s=10, zorder=10, depthshade=False)
        ax.set_title(title, fontsize=8, pad=2)
        ax.set_box_aspect([1, 1, view['zaspect']])
        ax.view_init(elev=view['elev'], azim=view['azim'])
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
        ax.grid(False)
        for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
            pane.pane.set_alpha(0.0)
        if idx_vec is idx_final:
            txt = _size_count_legend(model, idx_vec, CHS_CATALOG)
        else:
            txt = f"all {_n_physical_members(model)} members D = {CHS_CATALOG[idx_vec[0]]['D_mm']:g} mm"
        ax.text2D(0.5, 0.93, txt, transform=ax.transAxes, fontsize=5.8,
                  ha='center', va='top',
                  bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.85))

    _diameter_colorbar(fig, axes, cmap, [s['D_mm'] for s in CHS_CATALOG],
                        "CHS outside diameter D [mm]  --  dots: beam-to-beam connection points")
    savefig(fig, outname)


def fig_dome_sizing_design():
    # REVISION (dome redesign, 2026): switched from vault_posthoc.
    # recover_sizing_best('dome_sizing.json', dome_sizing_frozen_geometry)
    # (rf_dome_complex) to dome2_posthoc.recover_dome2_sizing_best (rf_dome2,
    # circular springing ring) -- same dome_sizing.json path, same SA-QUBO
    # (w_mass=5) best-seed / argmin-compliance_history recovery convention.
    # Same overall dome shape/ratios as fig_cs4_design, so reuses that view.
    from dome2_posthoc import recover_dome2_sizing_best
    _fig_sizing_design_impl(recover_dome2_sizing_best, dict(elev=26, azim=-60, zaspect=0.30),
                             'fig_dome_sizing_design', 'dome2-sizing')


def fig_vault_sizing_design():
    # REVISION (hex-pattern vault upgrade, 2026): switched from
    # vault_posthoc.recover_sizing_best('vault_sizing.json',
    # vault_sizing_frozen_geometry) (rf_vault, square pattern) to
    # vault_hex_posthoc.recover_vault_hex_sizing_best() (rf_vault_hex,
    # hexagonal pattern, frozen CS-VAULT-HEX-GEO-best geometry) -- same
    # nx/ny/span/rise envelope as fig_typologies panel (e)/
    # fig_vault_geo_design, so reuses the same _VAULT_VIEW convention.
    from vault_hex_posthoc import recover_vault_hex_sizing_best
    _fig_sizing_design_impl(recover_vault_hex_sizing_best,
                             _VAULT_VIEW, 'fig_vault_sizing_design', 'vault-hex-sizing')


# --------------------------------------------------------------------------
# fig_sizing_fixed_conv (NEW, B/W, two panels) -- second-stage F1 (SA-QUBO)
# vs GA vs random best-so-far convergence on the FROZEN stage-1-best
# geometries: (a) rf_dome_complex (dome_sizing.json), (b) rf_vault_hex
# (vault_hex_sizing.json). BOTH campaigns use only 5 seeds/method (half the
# 10 seeds/method used elsewhere in this module, since this is a stretch/
# second-stage addition on top of the already-10-seed stage-1 geometry
# searches) -- stated explicitly in each panel's own title (via the n_seeds
# value each JSON itself stores, d['n_seeds']) rather than left only to the
# caption, since it differs from every other convergence figure in this
# module and a reader comparing plots side by side should not assume 10
# seeds here. REVISION (hex-pattern vault upgrade, 2026): panel (b)
# switched from vault_sizing.json (rf_vault, square pattern) to
# vault_hex_sizing.json (rf_vault_hex, hexagonal pattern) -- STILL 5
# seeds/method (vault_hex_sizing.json's own n_seeds=5, same trim rationale
# as dome_sizing.json's), so the "n=5" panel title is unchanged.
# --------------------------------------------------------------------------
def fig_sizing_fixed_conv():
    fig, axes = plt.subplots(1, 2, figsize=(180 * MM, 74 * MM))
    labels = {'sa_qubo': 'SA-QUBO (F1)', 'ga': 'GA', 'random': 'Random search'}
    specs = [(axes[0], 'complex/dome_sizing.json', "(a) rf_dome2, frozen FMQA-best geometry"),
             (axes[1], 'complex/vault_hex_sizing.json', "(b) rf_vault_hex2, frozen CS-VAULT-HEX-GEO-best geometry")]
    for ax, json_name, title in specs:
        d = json.load(open(f"{RESULTS}/{json_name}"))
        n_seeds = d['n_seeds']
        for i, method in enumerate(('sa_qubo', 'ga', 'random')):
            st = _BW_SERIES[i]
            runs = _runs(d['methods'][method])
            hist = [r['compliance_best_so_far'] for r in runs]
            mean, std, n = _mean_std_bands(hist)
            x = np.arange(1, n + 1)
            ax.plot(x, mean, color=st['color'], ls=st['ls'], marker=st['marker'],
                    markevery=_markevery(n), ms=3.5, mfc='white', mew=0.7, label=labels[method])
            ax.fill_between(x, mean - std, mean + std, color=st['color'], alpha=0.16, linewidth=0)
        ax.set_xlabel("real FE evaluations")
        ax.set_ylabel("compliance (best-so-far)")
        ax.set_yscale('log')
        ax.set_title(f"{title}\n(n={n_seeds} seeds)", fontsize=7.4)
        ax.legend(loc='upper right', fontsize=6.3)
    fig.tight_layout()
    savefig(fig, "fig_sizing_fixed_conv")


# --------------------------------------------------------------------------
# fig_quantum_annealing (NEW, B/W) -- schematic of the quantum-annealing
# theory used by the SA-QUBO solver's physical motivation: (a) a rugged
# energy landscape E(configuration) contrasting classical thermal hopping
# (over a barrier) with quantum tunnelling (through a barrier); (b) the
# annealing schedule A(t) (decreasing)/B(t) (increasing) defining the
# time-dependent Hamiltonian H(t)=A(t)H_init+B(t)H_problem; (c) the
# spin-to-QUBO mapping chain s_i in {-1,+1} -> x_i in {0,1} -> x^T Q x via
# s=2x-1. No in-figure title (caption carries the explanation), matching
# the title-free convention adopted for every other figure in this
# study (see latex/figure_title_migration.md).
# --------------------------------------------------------------------------
def fig_quantum_annealing():
    fig, axes = plt.subplots(1, 3, figsize=(140 * MM, 66 * MM))
    fig.subplots_adjust(left=0.05, right=0.985, top=0.86, bottom=0.30, wspace=0.68)
    bbox = dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.95)
    # bare panel identifiers only (no descriptive titles) -- consistent
    # with every other figure in this study (see
    # latex/figure_title_migration.md): the explanation lives in the
    # caption, not in the figure.
    for ax_i, lbl in zip(axes, ("(a)", "(b)", "(c)")):
        ax_i.set_title(lbl, loc='left', fontsize=8.5)

    # ---- (a) energy landscape: thermal hopping vs quantum tunnelling -----
    ax = axes[0]
    x = np.linspace(0, 10, 2000)
    # multi-well landscape (explicit local + global wells + barrier hump):
    # LOCAL minimum near x=2.3 (E~1.0), GLOBAL minimum near x=8.3 (E~0.2,
    # deeper well), separated by a barrier peaking near x=5.2 (E~3.9).
    E = (2.2
         - 1.2 * np.exp(-((x - 2.3) ** 2) / (2 * 0.9 ** 2))
         - 2.0 * np.exp(-((x - 8.3) ** 2) / (2 * 0.9 ** 2))
         + 1.7 * np.exp(-((x - 5.2) ** 2) / (2 * 0.65 ** 2)))
    ax.plot(x, E, color='k', lw=1.3, zorder=3)

    i_glob = int(np.argmin(E))
    x_glob, E_glob = float(x[i_glob]), float(E[i_glob])
    mask_left = x < 4.5
    i_loc = int(np.argmin(np.where(mask_left, E, np.inf)))
    x_loc, E_loc = float(x[i_loc]), float(E[i_loc])
    mask_mid = (x > 3.0) & (x < 7.0)
    i_bar = int(np.argmax(np.where(mask_mid, E, -np.inf)))
    x_bar, E_bar = float(x[i_bar]), float(E[i_bar])

    ax.scatter([x_glob], [E_glob], color='k', s=28, zorder=6)
    ax.scatter([x_loc], [E_loc], color='0.45', s=24, zorder=6)
    # labels placed straight OUTWARD (local -> upper-left, global -> upper-
    # right) and shortened to "local/min", "global/min" at a slightly
    # smaller size so each text box is narrow enough to clear x_loc /
    # x_glob with margin: the box footprint was measured directly with
    # the renderer (ann.get_window_extent + inverted transData), not
    # guessed, because the hop-arc rises very steeply immediately past
    # both wells (e.g. it reaches y=2.4 only 0.6 data units past x_loc)
    # and a wide two-line box centered near the well was found to reach
    # straight into that rise. With the measured half-widths the boxes
    # (and the straight leader lines into them) stay entirely on the
    # outward side of x_loc / x_glob, clear of the hop-arc (which only
    # exists between x_loc and x_glob) and clear of the landscape curve
    # E(x) at every x they span -- checked numerically, not just by eye.
    # zorder is also kept above every curve/arrow/scatter in this panel
    # so the opaque white bbox masks anything behind it regardless.
    _lbl_z = 9
    ax.annotate("local\nmin", xy=(x_loc, E_loc), xycoords='data',
                xytext=(x_loc - 1.25, E_loc + 1.5), textcoords='data',
                ha='center', va='bottom', fontsize=5.8, color='0.35', bbox=bbox,
                zorder=_lbl_z,
                arrowprops=dict(arrowstyle='-', lw=0.6, color='0.5', shrinkA=2, shrinkB=3))
    ax.annotate("global\nmin", xy=(x_glob, E_glob), xycoords='data',
                xytext=(x_glob + 1.2, E_glob + 2.0), textcoords='data',
                ha='center', va='bottom', fontsize=5.8, bbox=bbox, zorder=_lbl_z,
                arrowprops=dict(arrowstyle='-', lw=0.7, color='0.3', shrinkA=2, shrinkB=3))

    # thermal hopping: solid arc drawn strictly ABOVE the barrier top.
    # Built as an explicit quadratic Bezier (not connectionstyle='arc3',
    # whose apex height is hard to control) so the apex is GUARANTEED to
    # clear E_bar by a fixed margin: for a quadratic Bezier B(t), the
    # midpoint value is B(0.5) = 0.25*P0 + 0.5*C + 0.25*P2, so solving for
    # the control point C given a target apex fixes the peak height exactly.
    from matplotlib.path import Path
    p0 = (x_loc, E_loc + 0.10)
    p2 = (x_glob, E_glob + 0.10)
    apex_y = E_bar + 0.65
    c_y = 2 * apex_y - 0.5 * p0[1] - 0.5 * p2[1]
    ctrl = (x_bar, c_y)
    hop_path = Path([p0, ctrl, p2], [Path.MOVETO, Path.CURVE3, Path.CURVE3])
    arc_hop = mpatches.FancyArrowPatch(path=hop_path, arrowstyle='-|>', mutation_scale=9,
                                        lw=1.15, color='k', zorder=5)
    ax.add_patch(arc_hop)
    ax.annotate("thermal hopping", xy=(x_bar, apex_y + 0.55),
                xycoords='data', ha='center', va='bottom', fontsize=6.7, color='k',
                zorder=_lbl_z)

    # quantum tunnelling: dashed, straight, drawn strictly BELOW the
    # barrier top and BELOW both wells -- "through" the barrier rather
    # than over it, with its own clear lane above the x-axis.
    y_tunnel = -0.9
    ax.annotate("", xy=(x_glob - 0.20, y_tunnel), xytext=(x_loc + 0.20, y_tunnel),
                xycoords='data', textcoords='data',
                arrowprops=dict(arrowstyle='-|>', lw=1.15, color='0.4', ls=(0, (4, 2)),
                                 shrinkA=1, shrinkB=1))
    ax.annotate("quantum tunnelling", xy=(x_bar, y_tunnel - 0.18),
                xycoords='data', ha='center', va='top', fontsize=6.7, color='0.4',
                zorder=_lbl_z)

    ax.set_xlabel("configuration", labelpad=4)
    ax.set_ylabel("energy $E$", labelpad=3)
    ax.set_xticks([]); ax.set_yticks([])
    y_bot = -1.75
    ax.set_xlim(-0.5, 10.5); ax.set_ylim(y_bot, E_bar + 2.55)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.annotate("", xy=(10.5, y_bot), xytext=(-0.5, y_bot), xycoords='data',
                arrowprops=dict(arrowstyle='-|>', lw=0.8, color='k'), annotation_clip=False)
    ax.annotate("", xy=(-0.5, E_bar + 2.55), xytext=(-0.5, y_bot), xycoords='data',
                arrowprops=dict(arrowstyle='-|>', lw=0.8, color='k'), annotation_clip=False)

    # ---- (b) annealing schedule A(t)/B(t) and H(t) ------------------------
    ax = axes[1]
    t = np.linspace(0, 1, 200)
    A = np.cos(0.5 * np.pi * t) ** 2   # 1 -> 0
    B = np.sin(0.5 * np.pi * t) ** 2   # 0 -> 1
    ax.plot(t, A, color='k', lw=1.3, ls='-', label='$A(t)$')
    ax.plot(t, B, color='0.45', lw=1.3, ls='--', label='$B(t)$')
    ax.set_xlabel(r"time  $t\,/\,t_{\max}$", labelpad=4)
    ax.set_ylabel("schedule weight", labelpad=3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["0", "$t_{\\max}$"])
    ax.set_yticks([0, 1])
    # A(t) and B(t) cross at t=0.5, so any legend wide enough to carry
    # "(initial term)"/"(problem term)" text straddles that crossing no
    # matter where it is centered (checked with get_window_extent: that
    # wording made the box ~0.8 axes-widths wide, wider than either half
    # of the panel). Labels shortened to bare "A(t)"/"B(t)" -- the
    # A<->H_init, B<->H_problem association is already spelled out in
    # the H(t) formula below -- so the box is narrow enough to sit
    # entirely in the t~0.6-0.95 gap where A(t) is already low and B(t)
    # is already high, clear of both curves with margin (re-verified
    # numerically after shortening, not just by eye).
    ax.legend(loc='center', fontsize=5.8, framealpha=1.0, edgecolor='0.6',
              handlelength=1.3, handletextpad=0.4, borderpad=0.35,
              bbox_to_anchor=(0.82, 0.5),
              bbox_transform=ax.transAxes, borderaxespad=0.2)
    ax.annotate(r"$H(t)=A(t)\,H_{\mathrm{init}}+B(t)\,H_{\mathrm{problem}}$",
                xy=(0.5, -0.46), xycoords='axes fraction', ha='center', va='top',
                fontsize=6.8, annotation_clip=False)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)

    # ---- (c) spin -> binary -> QUBO mapping chain --------------------------
    ax = axes[2]
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.axis('off')

    def box(cx, cy, w, h, text, fontsize=6.8):
        ax.add_patch(mpatches.FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                     boxstyle='round,pad=0.10', fc='white', ec='k', lw=1.0, zorder=3))
        ax.text(cx, cy, text, ha='center', va='center', fontsize=fontsize, zorder=4)

    # single vertical chain, generously spaced: spins -> bits -> QUBO
    box(5.0, 9.05, 7.4, 1.55, "spin variables\n$s_i\\in\\{-1,+1\\}$")
    ax.annotate("", xy=(5.0, 7.15), xytext=(5.0, 8.25), xycoords='data',
                arrowprops=dict(arrowstyle='->', lw=1.0, color='k'))
    ax.text(6.55, 7.70, "$s_i=2x_i-1$", ha='left', va='center', fontsize=6.5)

    box(5.0, 6.30, 7.4, 1.55, "binary variables\n$x_i\\in\\{0,1\\}$")
    ax.annotate("", xy=(5.0, 4.40), xytext=(5.0, 5.50), xycoords='data',
                arrowprops=dict(arrowstyle='->', lw=1.0, color='k'))
    ax.text(6.55, 4.95, "assemble $Q$", ha='left', va='center', fontsize=6.5)

    box(5.0, 3.55, 7.4, 1.75, "QUBO objective\n$\\;\\mathbf{x}^{T}Q\\,\\mathbf{x}\\;$",
        fontsize=7.2)
    ax.annotate("Ising $\\leftrightarrow$ QUBO: same\nproblem, two encodings",
                xy=(5.0, 1.35), xycoords='data', ha='center', va='center',
                fontsize=6.2, color='0.35')

    savefig(fig, "fig_quantum_annealing")


# --------------------------------------------------------------------------
# fig_stress_qubo (NEW, B/W, reader-response) -- RI stress-constrained
# sizing QUBO on CS2 (square_nexorade): (a) best-so-far compliance
# convergence, mean over 10 seeds, SA-QUBO vs greedy-linearized vs GA (GA's
# own stress-hinge-penalized fitness converted to raw compliance for the
# plot, matching the other case-study convergence figures); (b) final
# compliance mean +/- std bar/dot comparison, individual seed values
# overlaid as dots (same convention as fig_cs5_pareto panel (b)).
# results/ablation/ri_stress_qubo.json.
# --------------------------------------------------------------------------
def fig_stress_qubo():
    d = json.load(open(f"{RESULTS}/review/ri_stress_qubo.json"))
    n_seeds = d['setup']['n_seeds']

    def best_so_far_hist(states, key='comp_hist'):
        hists = [np.minimum.accumulate(states[str(s)][key]) for s in range(n_seeds)]
        return hists

    sa_hist = best_so_far_hist(d['sa_states'])
    gr_hist = best_so_far_hist(d['greedy_states'])
    ga_hist = [np.minimum.accumulate(d['ga_results'][str(s)]['C_hist']) for s in range(n_seeds)]

    fig, axes = plt.subplots(1, 2, figsize=(180 * MM, 78 * MM))

    ax = axes[0]
    series = [('SA-QUBO (stress-constrained F1)', sa_hist, _BW_SERIES[0]),
              ('Greedy-linearized', gr_hist, _BW_SERIES[1]),
              ('GA (stress-hinge fitness)', ga_hist, _BW_SERIES[2])]
    for label, hist, st in series:
        mean, std, n = _mean_std_bands(hist)
        x = np.arange(1, n + 1)
        ax.plot(x, mean, color=st['color'], ls=st['ls'], marker=st['marker'],
                markevery=_markevery(n), ms=3.5, mfc='white', mew=0.7, label=label)
        ax.fill_between(x, mean - std, mean + std, color=st['color'], alpha=0.16, linewidth=0)
    ax.set_xlabel("real FE evaluations")
    ax.set_ylabel("compliance (best-so-far)")
    ax.set_yscale('log')
    ax.legend(loc='upper right', fontsize=6.3)
    ax.set_title("(a) convergence", fontsize=8)

    ax2 = axes[1]
    summ = d['summary']
    methods = ['sa', 'greedy', 'ga']
    labels = ['SA-QUBO', 'Greedy-\nlinearized', 'GA']
    marker_shape = ['o', 's', '^']
    grays = ['k', '0.45', '0.7']
    means = [summ['sa']['compliance_mean'], summ['greedy']['compliance_mean'], summ['ga']['compliance_mean']]
    stds = [summ['sa']['compliance_std'], summ['greedy']['compliance_std'], summ['ga']['compliance_std']]
    posthoc = d['posthoc']
    x = np.arange(3)
    for i, m in enumerate(methods):
        vals = [posthoc[str(s)][m]['compliance'] for s in range(n_seeds)]
        jitter = (np.random.default_rng(0).random(len(vals)) - 0.5) * 0.12
        ax2.scatter(np.full(len(vals), x[i]) + jitter, vals, s=13, color=grays[i],
                    alpha=0.35, zorder=2, linewidth=0, marker=marker_shape[i])
    ax2.errorbar(x, means, yerr=stds, fmt='none', capsize=4, color='k', elinewidth=1.1, zorder=3)
    for i in range(3):
        ax2.scatter([x[i]], [means[i]], s=46, color=grays[i], marker=marker_shape[i], zorder=4,
                    edgecolor='k', linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xlim(-0.5, 2.5)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("final compliance (post-hoc re-evaluated)")
    ax2.set_title("(b) compliance comparison, 10 seeds\n(all 100% post-hoc feasible)", fontsize=7.5)

    fig.tight_layout()
    savefig(fig, "fig_stress_qubo")


if __name__ == "__main__":
    import sys as _sys
    names = _sys.argv[1:]
    g = globals()
    if not names:
        names = [k for k in g if k.startswith("fig_") and callable(g[k])]
    for n in names:
        print("=== running", n, "===")
        g[n]()

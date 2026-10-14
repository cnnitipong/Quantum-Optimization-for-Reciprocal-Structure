"""
reciprocal_geom_v2.py -- literature-faithful reciprocal-frame (nexorade)
geometry generators, per GEOMETRY_V2_SPEC.md.

Motivation / what changed vs reciprocal_geom.py (v1)
-----------------------------------------------------
v1's square_grillage/hex_grillage/dome built one small pinwheel fan per
grid CELL (or triangle) using the cell's own corner vertices as literal,
directly-shared "support" endpoints. Adjacent cells' fans therefore met at
a single rigid multi-valent joint at every interior grid vertex -- i.e.
interior SUPPORT NODES shared between fans. That is not how built
nexorades work: in a real rotation-method nexorade (Baverel/Douthe;
Song et al. 2013's (N,R,L) tangent-circle construction) every INTERIOR
EDGE of the base tessellation is realised as ONE physical member shared
between the two fans centred at its two endpoints; supports only occur at
the assembly boundary.

v2 fixes this with a "per-hub-fan + member fusion" construction, exactly
the "robust analytic alternative" the spec allows:

  1. Identify HUB sites of the base tessellation (square_nexorade: every
     strictly-interior grid VERTEX, degree 4; hex/tri-nexorade & rf_dome:
     every triangular FACE of a hex-boundary triangular grid, degree 3 via
     its 3 edge-midpoints). Every hub gets its own closed pinwheel_fan
     rotor (the exact, already-validated tangent-circle primitive used by
     fan_unit) -- self-contained, always non-degenerate, always a closed
     n-beam circuit.
  2. Every hub-fan beam k keeps its own two "near-hub" bearing nodes
     (lower(k-1): an interior point where the previous beam of THIS fan
     passes under; upper(k): this beam's own tip, where the NEXT beam of
     THIS fan rests on top) -- these come straight out of pinwheel_fan
     unmodified.
  3. What differs from v1 is what happens further OUT along each beam,
     at the tessellation EDGE it corresponds to:
       - If the edge reaches the assembly BOUNDARY (no fan on the far
         side): keep pinwheel_fan's own outer segment unmodified, i.e. the
         beam's far end sits exactly at that boundary vertex/edge-midpoint
         and gets PINNED there (classic fan-unit spoke-on-a-post).
       - If the edge is INTERIOR (a fan on both sides): DISCARD the two
         independent "outer" segments that v1 would have built reaching
         each other's raw vertex, and instead FUSE the two fans' own
         near-hub bearing chains directly with ONE new straight bridge
         element: tip_A -- lower_A -- [bridge] -- lower_B -- tip_B. The
         result is exactly ONE physical member per interior edge, with its
         own two ends each an "end bearing" (+e/2, resting on a neighbour
         within its own fan) and 1-2 interior bearing points (-e/2,
         neighbours resting on it) -- matching the spec's definition of an
         interior nexorade member, and matching "shared members between
         fans" rather than "shared support nodes".
  4. Supports appear ONLY at the tessellation boundary (perimeter vertices
     / boundary edge-midpoints); the model has zero interior supports.

Simplifications (stated here and to be repeated in the paper):
  - Each hub-fan is built from RAW neighbour-direction reference points
    (exactly pinwheel_fan's validated tangent-circle construction), not
    from a full simultaneous relaxation that forces perfect end-to-end
    collinearity of the fused member (cf. Song et al. 2013's "coherence"
    optimisation step, which we do not run). The two near-hub segments and
    the connecting bridge segment of a fused interior member are typically
    close to collinear but not exact; we render/report the member by its
    true extreme endpoints (the two end-bearing "tip" nodes... actually
    the two outermost nodes of its node chain), consistent with the
    existing project convention (true_node_positions / physical_beams)
    of drawing every physical member as ONE straight line regardless of
    how many FE sub-elements represent it.
  - hex_nexorade / rf_dome use 3-member fans centred on triangular FACES
    (one fan per up- or down-pointing triangle of the base grid, i.e. the
    fans alternate in orientation across the domain) rather than at the
    (degree-6) vertices of the triangular lattice -- this is the reading
    of the spec's "3-member fans at alternating vertices" that is
    actually constructible with exactly-3-beam rotors and gives the
    canonical hexagonal-looking assembled pattern (cf. Song et al. 2013,
    Fig. 3 "hex & tri RF-units").
  - davinci_bridge prescribes the arch profile geometrically (circular
    segment through given span/rise; straight chords) per the spec's
    explicitly allowed simplification, and encodes the engagement/overlap
    ratio lambda as the vertical bearing eccentricity scale (which
    directly and monotonically changes both the geometry and the
    stiffness at each rung/longitudinal contact) rather than as an
    explicit drawn horizontal overlap length past each rung (which would
    add non-load-bearing cantilever stubs with no mechanical role in this
    idealisation); this is stated explicitly wherever lambda is used.

All generators return the SAME RFModel dict-schema as v1 (nodes, elements
with i/j/kind/group [+ a new, additive 'member_id' key on 'beam' elements
used only by this module's own physical_members_v2 helper], support_idx,
load_idx, meta) and reuse v1's build_frame / model_mass / true_node_positions
unchanged -- so all downstream FE + plotting utilities keep working. Because
fused interior members are chains of 2-3 sub-elements (not always exactly 2,
as v1's physical_beams() assumes), this module adds physical_members_v2(),
an explicit member_id-based reconstruction, instead of forcing v1's
physical_beams() to cope with variable-length chains.
"""
import itertools
import numpy as np

from frame3d import Frame3D
from sections import E_STEEL, G_STEEL, RHO_STEEL, CHS_CATALOG, N_SECTIONS
from reciprocal_geom import (
    pinwheel_fan, RFModel, build_frame, model_mass, true_node_positions,
    LINK_SECTION, _hex_axial_nodes, _hex_triangles,
)

__all__ = [
    "fan_unit", "square_nexorade", "hex_nexorade", "rf_dome", "davinci_bridge",
    "rf_dome_complex", "davinci_bridge_complex", "rf_vault", "rf_vault_hex", "rf_vault_hex2",
    "physical_members_v2", "verify_nonsingular", "verify_no_interior_supports",
    "verify_member_bearings", "verify_bridge_bearings", "verify_bridge_bearings_complex",
    "verify_vault_bearings", "verify_vault_hex2_bearings",
    "verify_deflection", "model_mass", "true_node_positions",
]


# --------------------------------------------------------------------------
# 1. fan_unit -- KEPT AS-IS (canonical single rotor; spec explicitly says keep)
# --------------------------------------------------------------------------
def fan_unit(n_beams, R, lam, ecc_ratio=0.06, single_group=True, load_per_node=-2200.0):
    """Standalone n-beam fan/rotor (unchanged from v1). Pinned at outer
    (support) ends, vertical load at each engagement (inner) node. This IS
    a canonical reciprocal-frame unit (Song et al. 2013's single RF-unit;
    Baverel's rotor) -- no v2 changes needed."""
    theta = np.linspace(0, 2 * np.pi, n_beams, endpoint=False)
    support_pts = [(R * np.cos(t), R * np.sin(t), 0.0) for t in theta]
    ecc = ecc_ratio * R
    fan = pinwheel_fan(support_pts, lam, ecc)

    model = RFModel()
    node_id_map = {}
    for k in range(n_beams):
        gi = model.add_node(support_pts[k])
        node_id_map[k] = gi
        model.support_idx.add(gi)

    def group_fn(beam_id):
        return 0 if single_group else beam_id

    local_to_global = model.merge_fan(fan, node_id_map, group_fn)
    # tag member_id so physical_members_v2 also works on fan_unit models
    mid_counter = itertools.count(1)
    beam_elems_idx = [idx for idx, e in enumerate(model.elements) if e['kind'] == 'beam']
    for a_idx, b_idx in zip(beam_elems_idx[0::2], beam_elems_idx[1::2]):
        mid = next(mid_counter)
        model.elements[a_idx]['member_id'] = mid
        model.elements[b_idx]['member_id'] = mid

    model.n_groups = 1 if single_group else n_beams
    model.meta = dict(kind='fan_unit', n_beams=n_beams, R=R, lam=lam,
                       t_realized=fan['t_realized'], ecc=ecc,
                       load_per_node=load_per_node)
    return model


# --------------------------------------------------------------------------
# shared small helpers
# --------------------------------------------------------------------------
def _beam_pairs(fan):
    """Split a pinwheel_fan()'s flat elements list back into its n beams'
    2 sub-elements each, in (support_local, lower_local, upper_local) form
    -- i.e. undo the flattening done inside pinwheel_fan, which appends
    beam k's two elements as consecutive entries in k order (see that
    function's docstring/source)."""
    beam_elems = [e for e in fan['elements'] if e[2] == 'beam']
    n = fan['n']
    assert len(beam_elems) == 2 * n
    out = []
    for k in range(n):
        s_local, l_local, kind_a, bid_a = beam_elems[2 * k]
        l_local2, u_local, kind_b, bid_b = beam_elems[2 * k + 1]
        assert l_local == l_local2 and bid_a == bid_b == k
        out.append((s_local, l_local, u_local))
    return out


class _MemberIdGen:
    def __init__(self):
        self._c = itertools.count(1)

    def new(self):
        return next(self._c)


# --------------------------------------------------------------------------
# 2. square_nexorade -- rotated-square (Ito/pinwheel) nexorade grillage
# --------------------------------------------------------------------------
def square_nexorade(nx, ny, cell, lam, ecc_ratio=0.07, load_per_node=-450.0,
                     chirality=-1):
    """Rotated-square nexorade grillage. See module docstring for the
    per-hub-fan + member-fusion construction. nx, ny >= 3 required for at
    least one interior-interior fused edge (a rich pattern wants nx,ny>=5)."""
    if nx < 3 or ny < 3:
        raise ValueError("square_nexorade needs nx,ny >= 3 for an interior hub")

    model = RFModel()
    ecc = ecc_ratio * cell
    midg = _MemberIdGen()

    def pos(ix, iy):
        return (ix * cell, iy * cell, 0.0)

    def is_hub(ix, iy):
        return 1 <= ix <= nx - 1 and 1 <= iy <= ny - 1

    boundary_node = {}

    def boundary_gid(ix, iy):
        key = (ix, iy)
        if key not in boundary_node:
            boundary_node[key] = model.add_node(pos(ix, iy))
        return boundary_node[key]

    NEI_DIRS = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # cyclic E, N, W, S

    hubs = [(ix, iy) for ix in range(1, nx) for iy in range(1, ny)]
    n_rings_ix = (nx - 1) // 2 + 1
    n_rings_iy = (ny - 1) // 2 + 1
    max_ring = max(1, min(n_rings_ix, n_rings_iy, 4))  # cap groups (spec: 8-16 groups)

    spoke = {}  # (hub_key, neighbor_key) -> dict(lower=gid, tip=gid, group=g)

    for (ix, iy) in hubs:
        neigh = [(ix + dx, iy + dy) for dx, dy in NEI_DIRS]
        support_pts = [pos(*nk) for nk in neigh]
        fan = pinwheel_fan(support_pts, lam, ecc, chirality=chirality)
        n = fan['n']
        loc2glob = {}

        def gid(li, _map=loc2glob, _fan=fan):
            if li not in _map:
                _map[li] = model.add_node(_fan['nodes'][li])
            return _map[li]

        for (i, j, kind, bid) in fan['elements']:
            if kind == 'link':
                model.elements.append(dict(i=gid(i), j=gid(j), kind='link', group=None))

        ring = min(ix - 1, nx - 1 - ix, iy - 1, ny - 1 - iy)
        ring = min(ring, max_ring - 1)

        for k, (s_local, l_local, u_local) in enumerate(_beam_pairs(fan)):
            lower_g = gid(l_local)
            upper_g = gid(u_local)
            group = ring * 4 + k
            neighbor_key = neigh[k]
            if is_hub(*neighbor_key):
                spoke[((ix, iy), neighbor_key)] = dict(lower=lower_g, tip=upper_g, group=group)
            else:
                bnd_g = boundary_gid(*neighbor_key)
                model.support_idx.add(bnd_g)
                mid = midg.new()
                model.elements.append(dict(i=bnd_g, j=lower_g, kind='beam', group=group, member_id=mid))
                model.elements.append(dict(i=lower_g, j=upper_g, kind='beam', group=group, member_id=mid))
                model.load_idx.append(upper_g)

    processed = set()
    for (hub_key, nb_key), info_a in list(spoke.items()):
        ek = frozenset({hub_key, nb_key})
        if ek in processed:
            continue
        info_b = spoke.get((nb_key, hub_key))
        if info_b is None:
            continue
        processed.add(ek)
        mid = midg.new()
        model.elements.append(dict(i=info_a['tip'], j=info_a['lower'], kind='beam', group=info_a['group'], member_id=mid))
        model.elements.append(dict(i=info_a['lower'], j=info_b['lower'], kind='beam', group=info_a['group'], member_id=mid))
        model.elements.append(dict(i=info_b['lower'], j=info_b['tip'], kind='beam', group=info_b['group'], member_id=mid))
        model.load_idx.append(info_a['tip'])
        model.load_idx.append(info_b['tip'])

    model.n_groups = max_ring * 4
    model.meta = dict(kind='square_nexorade', nx=nx, ny=ny, cell=cell, lam=lam, ecc=ecc,
                       n_hubs=len(hubs), load_per_node=load_per_node,
                       span=max(nx, ny) * cell)
    return model


# --------------------------------------------------------------------------
# 3/4. hex_nexorade & rf_dome -- shared face-hub core
# --------------------------------------------------------------------------
def _hex_nexorade_core(rings, edge, lam, ecc_ratio, pos_fn, chirality=-1,
                        kind_name='hex_nexorade', lam_per_ring=None,
                        max_group_rings=None):
    """Shared builder for hex_nexorade (planar pos_fn) and rf_dome (pos_fn
    lifts (x,y) onto a spherical cap). Hubs = triangular FACES of the
    hex-boundary triangular grid (both up/down orientations -- i.e. fans
    at alternating triangle orientations); each fan's 3 support points are
    the face's 3 EDGE-MIDPOINTS (not its vertices), so that every grid
    EDGE is naturally shared by exactly its two adjacent triangular faces
    (or exactly one, if it lies on the domain boundary) -- the same
    pairwise-fusion structure as square_nexorade, just face- instead of
    vertex-centred.

    lam_per_ring: optional list of length `rings` giving a DISTINCT
    engagement ratio per concentric ring FAMILY (face_ring(...) below,
    0=innermost real ring .. rings-1=outermost) -- an additive,
    backward-compatible extension (default None reproduces the original
    single-lam behaviour exactly) used only by CS4's FMQA black-box case
    study, which needs a per-ring-family engagement variable exactly as
    v1's dome(lam_per_ring=) provided. Eccentricity magnitude (ecc =
    ecc_ratio*edge) is unchanged across rings; only the tangent-circle
    engagement ratio varies.

    max_group_rings: optional int, additive/backward-compatible (default
    None reproduces one section-group family per ring, i.e.
    n_groups=rings*3). When given, ring FAMILIES beyond max_group_rings-1
    are folded into the outermost retained group family (group_ring =
    min(face_ring, max_group_rings-1)) so mesh RESOLUTION (rings, driving
    member/DOF count) and section-optimisation GROUP COUNT can be varied
    independently -- used by rf_dome_complex to keep a dense
    Song-et-al.-style multi-ring tiling inside the spec's 8-14-group
    section-search-space budget.

    PHANTOM-GROUP FIX (see results/group_index_audit.json): a
    triangular FACE's ring is the MAX axial ring of its 3 vertices
    (face_ring() below); since the single centre vertex (axial ring 0)
    can never by itself form a face's 3 vertices, no face can EVER have
    face_ring()==0 under the original 0..rings numbering -- ring family 0
    (and hence its 3 section groups, k=0,1,2) was therefore structurally
    unreachable/always-empty for EVERY (rings, edge) combination, not a
    parameter-specific accident. face_ring() now returns a contiguous
    0-based family index (true ring - 1, range [0, rings-1]) so every
    reported group_ring_cap*3 group is guaranteed non-empty; group_ring_cap
    defaults to `rings` (was `rings+1`) accordingly. Geometry (nodes,
    members, topology) is bit-for-bit unchanged -- only the group LABEL
    each beam element carries is renumbered/shifted by -1 relative to the
    old (buggy) numbering, and the count of distinct groups drops from
    (old ring range 1..rings, i.e. `rings` distinct non-empty families
    reported as `rings+1`) to the true `rings`."""
    coords = _hex_axial_nodes(rings, edge)
    faces = _hex_triangles(coords, rings)

    def midpoint(a, b):
        pa, pb = np.array(pos_fn(coords[a])), np.array(pos_fn(coords[b]))
        return tuple(0.5 * (pa + pb))

    edge_faces = {}
    for fi, (a, b, c) in enumerate(faces):
        for (p, q) in ((a, b), (b, c), (c, a)):
            edge_faces.setdefault(frozenset({p, q}), []).append(fi)

    model = RFModel()
    ecc = ecc_ratio * edge
    midg = _MemberIdGen()

    boundary_node = {}

    def boundary_gid(mid_pt):
        key = tuple(np.round(mid_pt, 9))
        if key not in boundary_node:
            boundary_node[key] = model.add_node(mid_pt)
        return boundary_node[key]

    def face_ring(a, b, c):
        r = 0
        for key in (a, b, c):
            q, s = key
            r = max(r, abs(q), abs(s), abs(q + s))
        # PHANTOM-GROUP FIX: the raw vertex-ring max is always >=1 for any
        # actual triangular face (only the single centre vertex sits at
        # ring 0, and no face has all 3 vertices there) -- shift by -1 so
        # the family index is contiguous 0..rings-1 with no unreachable
        # leading family (see docstring above / group_index_audit.json).
        return min(int(r), rings) - 1

    spoke = {}  # frozenset(edge_verts) -> list of dict(lower, tip, group, face)
    group_ring_cap = rings if max_group_rings is None else max_group_rings

    for fi, (a, b, c) in enumerate(faces):
        verts = (a, b, c)
        dir_edges = ((a, b), (b, c), (c, a))
        support_pts = [midpoint(p, q) for (p, q) in dir_edges]
        face_ring_id = face_ring(a, b, c)
        fan_lam = lam_per_ring[face_ring_id] if lam_per_ring is not None else lam
        fan = pinwheel_fan(support_pts, fan_lam, ecc, chirality=chirality)
        loc2glob = {}

        def gid(li, _map=loc2glob, _fan=fan):
            if li not in _map:
                _map[li] = model.add_node(_fan['nodes'][li])
            return _map[li]

        for (i, j, k2, bid) in fan['elements']:
            if k2 == 'link':
                model.elements.append(dict(i=gid(i), j=gid(j), kind='link', group=None))

        ring = face_ring(a, b, c)
        group_ring = min(ring, group_ring_cap - 1)
        for k, (s_local, l_local, u_local) in enumerate(_beam_pairs(fan)):
            lower_g = gid(l_local)
            upper_g = gid(u_local)
            group = group_ring * 3 + k
            ek = frozenset(dir_edges[k])
            if len(edge_faces[ek]) == 2:
                spoke.setdefault(ek, []).append(dict(lower=lower_g, tip=upper_g, group=group))
            else:
                mid_pt = midpoint(*dir_edges[k])
                bnd_g = boundary_gid(mid_pt)
                model.support_idx.add(bnd_g)
                mid = midg.new()
                model.elements.append(dict(i=bnd_g, j=lower_g, kind='beam', group=group, member_id=mid))
                model.elements.append(dict(i=lower_g, j=upper_g, kind='beam', group=group, member_id=mid))
                model.load_idx.append(upper_g)

    for ek, infos in spoke.items():
        if len(infos) != 2:
            continue
        info_a, info_b = infos
        mid = midg.new()
        model.elements.append(dict(i=info_a['tip'], j=info_a['lower'], kind='beam', group=info_a['group'], member_id=mid))
        model.elements.append(dict(i=info_a['lower'], j=info_b['lower'], kind='beam', group=info_a['group'], member_id=mid))
        model.elements.append(dict(i=info_b['lower'], j=info_b['tip'], kind='beam', group=info_b['group'], member_id=mid))
        model.load_idx.append(info_a['tip'])
        model.load_idx.append(info_b['tip'])

    model.n_groups = group_ring_cap * 3
    model.meta = dict(kind=kind_name, rings=rings, edge=edge, lam=lam, ecc=ecc,
                       n_faces=len(faces), max_group_rings=group_ring_cap)
    return model


def hex_nexorade(rings, edge, lam, ecc_ratio=0.07, load_per_node=-1400.0, chirality=-1):
    """Hexagonal/triangular nexorade: 3-member fans centred on the up/down
    triangular faces of a hex-boundary triangular grid (alternating
    orientation across the domain -- see module docstring), members shared
    pairwise across interior grid edges, boundary edges pinned at their
    midpoint."""
    m = _hex_nexorade_core(rings, edge, lam, ecc_ratio, pos_fn=lambda xy: (xy[0], xy[1], 0.0),
                            chirality=chirality, kind_name='hex_nexorade')
    m.meta['load_per_node'] = load_per_node
    m.meta['span'] = 2 * rings * edge
    return m


def rf_dome(rings, span, rise, lam, ecc_ratio=0.07, load_per_node=-3500.0, chirality=-1,
            lam_per_ring=None, max_group_rings=None):
    """Reciprocal dome: the hex_nexorade planar face-hub pattern mapped
    onto a spherical cap of given base span/rise (Song et al. 2013
    surface-mapping approach -- members stay straight chords between their
    mapped 3D endpoints; pinwheel_fan's own SVD local-plane basis handles
    the resulting mild non-planarity of each fan without modification).

    lam_per_ring: optional list of length (rings+1), one engagement ratio
    per concentric ring family (see _hex_nexorade_core docstring) -- used
    by CS4's FMQA black-box case study; default None keeps the single
    scalar `lam` applied uniformly (unchanged behaviour).

    max_group_rings: optional int, additive/backward-compatible passthrough
    to _hex_nexorade_core (default None keeps the original n_groups=(rings+1)*3
    behaviour) -- lets rf_dome_complex increase mesh density (`rings`)
    without blowing up the section-group count; see that function and
    _hex_nexorade_core's docstrings."""
    edge = (span / 2.0) / max(rings, 1)
    coords = _hex_axial_nodes(rings, edge)
    max_r = max(np.hypot(x, y) for x, y in coords.values())
    scale = (span / 2.0) / max_r if max_r > 0 else 1.0
    a = span / 2.0
    Rs = (a ** 2 + rise ** 2) / (2 * rise)

    def lift(xy):
        x, y = xy[0] * scale, xy[1] * scale
        r_xy = min(np.hypot(x, y), a)
        z = rise - (Rs - np.sqrt(max(Rs ** 2 - r_xy ** 2, 0.0)))
        return (x, y, z)

    m = _hex_nexorade_core(rings, edge, lam, ecc_ratio, pos_fn=lift,
                            chirality=chirality, kind_name='rf_dome',
                            lam_per_ring=lam_per_ring,
                            max_group_rings=max_group_rings)
    m.meta['span'] = span
    m.meta['rise'] = rise
    m.meta['Rs'] = Rs
    m.meta['load_per_node'] = load_per_node
    return m


def rf_dome_complex(rings=4, span=18.5, rise=3.15, lam=0.25, ecc_ratio=0.07,
                     load_per_node=-3500.0, chirality=-1, lam_per_ring=None,
                     max_group_rings=4):
    """Denser Song et al. 2013 style RF dome -- a thin wrapper around
    rf_dome() with defaults chosen to match the DENSE, many-ring RF-dome
    tessellations shown in Song et al.'s Figure 15 (fifteen tri/quad/hex
    RF-tessellation patterns covering a full hemispherical cap) and Figure
    19 (the "Da Vinci dome"), rather than rf_dome's coarser rings=3 default
    (which only reads as a single shallow ring of fans, not the dense
    multi-ring "many crossing rods with small triangular engagement
    windows" look of a real built RF dome).

    Construction choices (see report for the underlying trade-off):
      - rings=4 (vs rf_dome's default 3): one additional concentric
        ring-family of face-hub fans from apex to base, i.e. a genuinely
        4-ring-family dense tiling, matching this module's own docstring's
        "hex & tri RF-units" reading of Song Fig. 3's face-hub construction
        pushed to the same mesh-resolution regime as Song's Table 1
        "large" reference structures (92 RF-units / 205 rods): rings=4
        here gives 92 triangular face-hub fans / 152 physical members --
        same order of magnitude, allowing for this module's own
        member-fusion bookkeeping (Song counts assembled *rods*, we count
        fully-fused physical chains).
      - span 18.5 m, rise 3.15 m => rise/span ~=0.170, inside the
        0.15-0.20 plausible built-lattice-dome proportion window.
      - max_group_rings=4 caps the section-group count at 4 ring
        families x 3 orientations = 12 groups (spec's 8-14-group target)
        EVEN THOUGH the mesh itself has 5 ring families (0..4) at rings=4
        -- the outermost ring family is folded into ring family 3's
        section groups (see _hex_nexorade_core's max_group_rings doc).

    All other properties (member fusion, perimeter-only supports, +-e/2
    bearings, straight members, non-singular K) are inherited unchanged
    from rf_dome/_hex_nexorade_core.

    DOF note (see report): at rings=4 this model has restrained-DOF ~3.4k
    (vs rf_dome's rings=3 default ~1.9k) -- slightly above the spec's
    "~3200" soft cap, because in this construction's fixed per-fan node
    budget (6 FE nodes/face-fan) member count and DOF both scale ~linearly
    with face count, so the 150-member floor and the ~3200-DOF ceiling
    cannot both be hit exactly with an unmodified core; rings=4 was chosen
    as the closest feasible balance (reported not hidden)."""
    m = rf_dome(rings=rings, span=span, rise=rise, lam=lam, ecc_ratio=ecc_ratio,
                load_per_node=load_per_node, chirality=chirality,
                lam_per_ring=lam_per_ring, max_group_rings=max_group_rings)
    m.meta['kind'] = 'rf_dome_complex'
    m.meta['variant'] = 'complex'
    m.meta['rise_span_ratio'] = rise / span
    return m


# --------------------------------------------------------------------------
# 5. davinci_bridge -- Leonardo's self-supporting reciprocal bridge (NEW)
# --------------------------------------------------------------------------
def davinci_bridge(n_bays, span, width, rise=None, lam=0.25, ecc_ratio=0.10,
                    load_per_node=-5500.0):
    """Leonardo da Vinci self-supporting reciprocal bridge.

    Simplification (stated per spec): the arch camber is PRESCRIBED as a
    circular-segment profile of given span/rise (straight chords between
    stations) rather than emergent purely from eccentricity stacking. What
    IS modelled faithfully is the reciprocal bearing PATTERN: two parallel
    longitudinal rod chains (left/right rail, `width` apart), one member
    per bay, interleaved with transverse rungs at each interior bay
    station; at each interior station the incoming longitudinal member's
    end rests UNDER the rung (-ecc/2) and the outgoing longitudinal
    member's start rests ON TOP of the rung (+ecc/2), alternating
    over/under exactly like a fan_unit's link-element bearing, but chained
    linearly along the bridge instead of in a closed rotor.

    Engagement variable: `lam` (the spec's "overlap ratio of longitudinal
    members past the rung") is realised here as the ratio controlling the
    vertical bearing eccentricity magnitude (ecc = ecc_ratio*lam/0.25*bay),
    not as an explicit drawn horizontal overlap length past each rung --
    a drawn horizontal overlap stub would be a free, unloaded cantilever
    tail with no mechanical role in this point-contact idealisation, so it
    is omitted; lam still regenerates distinct, non-singular geometry/
    stiffness at each of the required test values.

    Supports: abutment (x=0 and x=span) ends of the first/last bay
    longitudinal members pinned. Load: vertical, at rung connection nodes
    (deck load).
    """
    if n_bays < 3:
        raise ValueError("davinci_bridge needs n_bays >= 3 for at least one interior rung pair")
    if rise is None:
        rise = 0.12 * span
    model = RFModel()
    bay = span / n_bays
    ecc = ecc_ratio * (lam / 0.25) * bay
    xs = np.linspace(0.0, span, n_bays + 1)
    a = span / 2.0
    Rc = (a ** 2 + rise ** 2) / (2 * rise)

    def z_of(x):
        dx = x - a
        val = max(Rc ** 2 - dx ** 2, 0.0)
        return rise - (Rc - np.sqrt(val))

    ys = [-width / 2.0, width / 2.0]
    long_start = {}
    long_end = {}
    for si, y in enumerate(ys):
        for i in range(n_bays):
            x0, x1 = xs[i], xs[i + 1]
            if i == 0:
                n0 = model.add_node((x0, y, z_of(x0)))
                model.support_idx.add(n0)
            else:
                n0 = model.add_node((x0, y, z_of(x0) + ecc / 2.0))
                long_start[(si, i)] = n0
            if i == n_bays - 1:
                n1 = model.add_node((x1, y, z_of(x1)))
                model.support_idx.add(n1)
            else:
                n1 = model.add_node((x1, y, z_of(x1) - ecc / 2.0))
                long_end[(si, i)] = n1
            quarter = min(3, (i * 4) // n_bays)
            grp = si * 4 + quarter
            model.elements.append(dict(i=n0, j=n1, kind='beam', group=grp,
                                        member_id=('long', si, i)))

    for i in range(1, n_bays):
        x = xs[i]
        z = z_of(x)
        left = model.add_node((x, ys[0], z))
        right = model.add_node((x, ys[1], z))
        rung_grp = 8 + (i % 2)
        model.elements.append(dict(i=left, j=right, kind='beam', group=rung_grp,
                                    member_id=('rung', i)))
        model.load_idx.append(left)
        model.load_idx.append(right)
        for si, node_side in ((0, left), (1, right)):
            incoming = long_end[(si, i - 1)]
            outgoing = long_start[(si, i)]
            model.elements.append(dict(i=node_side, j=incoming, kind='link', group=None))
            model.elements.append(dict(i=node_side, j=outgoing, kind='link', group=None))

    model.n_groups = 10
    model.meta = dict(kind='davinci_bridge', n_bays=n_bays, span=span, width=width,
                       rise=rise, lam=lam, ecc=ecc, load_per_node=load_per_node)
    return model


def davinci_bridge_complex(n_bays=12, span=20.0, width=6.0, rise=None, lam=0.25,
                            ecc_ratio=0.10, load_per_node=-5500.0):
    """Richer Leonardo da Vinci self-supporting reciprocal bridge: more bays
    (10-12, vs davinci_bridge's default 6) AND a THIRD longitudinal rail
    (3 parallel rod chains across a wider deck, vs davinci_bridge's 2),
    with transverse rungs at EVERY interior station bridging all 3 rails --
    the natural "notched grid" extension of the 2-rail idiom to a wider
    deck, in the spirit of the RF literature's reading of Leonardo's
    self-supporting bridge sketches (see davinci_bridge's own docstring
    for the engagement-ratio-as-bearing-eccentricity simplification, reused
    unchanged here).

    Construction:
      - 3 rails at y in {-width/2, 0, +width/2} (left/centre/right), each
        built exactly like davinci_bridge's single rail: n_bays longitudinal
        members per rail, abutment-pinned at the two ends, interior joints
        alternating over(+ecc/2)/under(-ecc/2) bearing exactly as before.
      - At every interior station i=1..n_bays-1, a full-width transverse
        rung is realised as TWO straight half-width physical members
        (left-to-centre, centre-to-right) meeting at the centre rail's own
        deck node -- i.e. the single 2-rail "rung" member is naturally
        split by the added centre rail, giving 2 independently-sized rung
        members per station instead of 1 (richer member-size diversity by
        construction, matching the task's aim of more distinct group
        sizes). Every rail's deck node at that station still bears via a
        link element on/under its own incoming/outgoing longitudinal
        member, exactly as in davinci_bridge (generalised to 3 rails
        instead of 2).

    Member groups (target 10-16, spec): longitudinal members are grouped
    by (rail x quarter-span bay-pair) = 3 rails x 4 quarters = 12 groups;
    rung segments are grouped by (station parity x left/right half) = 4
    groups; n_groups = 16 total.

    Member count: 3*n_bays longitudinal + 2*(n_bays-1) rung segments (e.g.
    n_bays=12 => 36+22 = 58 physical members, inside the spec's 40-70
    target).

    Supports: abutment (x=0, x=span) ends of all 3 rails' first/last-bay
    longitudinal members pinned (6 supports total; zero interior supports).
    Load: vertical, at every rail deck node (3 per interior station)."""
    if n_bays < 4:
        raise ValueError("davinci_bridge_complex needs n_bays >= 4 for at least "
                          "one interior rung station")
    if rise is None:
        rise = 0.12 * span
    model = RFModel()
    bay = span / n_bays
    ecc = ecc_ratio * (lam / 0.25) * bay
    xs = np.linspace(0.0, span, n_bays + 1)
    a = span / 2.0
    Rc = (a ** 2 + rise ** 2) / (2 * rise)

    def z_of(x):
        dx = x - a
        val = max(Rc ** 2 - dx ** 2, 0.0)
        return rise - (Rc - np.sqrt(val))

    ys = [-width / 2.0, 0.0, width / 2.0]  # 3 rails: left, centre, right
    n_rails = len(ys)
    long_start = {}
    long_end = {}
    for si, y in enumerate(ys):
        for i in range(n_bays):
            x0, x1 = xs[i], xs[i + 1]
            if i == 0:
                n0 = model.add_node((x0, y, z_of(x0)))
                model.support_idx.add(n0)
            else:
                n0 = model.add_node((x0, y, z_of(x0) + ecc / 2.0))
                long_start[(si, i)] = n0
            if i == n_bays - 1:
                n1 = model.add_node((x1, y, z_of(x1)))
                model.support_idx.add(n1)
            else:
                n1 = model.add_node((x1, y, z_of(x1) - ecc / 2.0))
                long_end[(si, i)] = n1
            quarter = min(3, (i * 4) // n_bays)
            grp = si * 4 + quarter
            model.elements.append(dict(i=n0, j=n1, kind='beam', group=grp,
                                        member_id=('long', si, i)))

    for i in range(1, n_bays):
        x = xs[i]
        z = z_of(x)
        rail_nodes = [model.add_node((x, y, z)) for y in ys]
        parity = i % 2
        for seg_idx in range(n_rails - 1):
            a_node = rail_nodes[seg_idx]
            b_node = rail_nodes[seg_idx + 1]
            rung_grp = 12 + parity * 2 + seg_idx
            model.elements.append(dict(i=a_node, j=b_node, kind='beam',
                                        group=rung_grp,
                                        member_id=('rung', i, seg_idx)))
        for node in rail_nodes:
            model.load_idx.append(node)
        for si, node_side in enumerate(rail_nodes):
            incoming = long_end[(si, i - 1)]
            outgoing = long_start[(si, i)]
            model.elements.append(dict(i=node_side, j=incoming, kind='link', group=None))
            model.elements.append(dict(i=node_side, j=outgoing, kind='link', group=None))

    model.n_groups = 16
    model.meta = dict(kind='davinci_bridge_complex', n_bays=n_bays, span=span,
                       width=width, rise=rise, lam=lam, ecc=ecc,
                       load_per_node=load_per_node, n_rails=n_rails)
    return model


# --------------------------------------------------------------------------
# 5b. davinci_bridge2 -- REDESIGN of davinci_bridge_complex fixing the
#     strain-energy-floor defect (22/58 rung half-segments carrying
#     essentially zero force; see results/bridge_diagnosis.json).
# --------------------------------------------------------------------------
def davinci_bridge2(n_bays=24, span=20.0, width=6.0, rise=None, lam=0.25,
                     ecc_ratio=0.10, load_per_node=-5500.0):
    """Load-path-corrected 3-rail Da Vinci bridge (fixes davinci_bridge_complex's
    genuine strain-energy-floor defect, see results/bridge_diagnosis.json).

    DIAGNOSIS (recap; full detail in bridge_diagnosis.json): in
    davinci_bridge_complex, the deck's vertical service/self-weight load is
    applied ONLY at each rail's own deck node at every interior station.
    Those same nodes ALSO carry two eccentric 'link' bearings onto the
    adjacent longitudinal-rail members (the same +/-ecc/2 notch idiom used
    throughout this project), and it is THAT eccentricity, not the
    transverse rung, that gives the longitudinal rails their moment arm to
    carry vertical load in arch action. The transverse rung half-segments
    (oriented purely along y) then have no applied transverse (y) load and
    no compatibility requirement forcing any relative y-displacement between
    their two ends, so they carry essentially zero axial/bending force --
    a genuine, reproducible (service-load AND self-weight) topological
    modelling artifact of the split-rung construction, not a sizing
    artifact.

    REDESIGN CHOICE: option (b) from the task spec -- ADD the deck load at
    each rung half-segment's OWN mid-span (in addition to, not instead of,
    the existing rail-node loads, which are kept unchanged so the
    longitudinal rails' eccentric-bearing arch action is fully preserved).
    Concretely: each rung half-segment (still 2 independent half-members
    per station, preserving the member-size-diversity feature of the
    original 3-rail design and its 4 rung-station-parity/half groups) is
    now built as TWO physical FE sub-elements (rail-node -> new mid-node,
    mid-node -> rail-node) sharing one member_id, i.e. still exactly ONE
    physical member per half-segment -- and `load_per_node` is applied at
    that new mid-node exactly like every other loaded node in this project
    (frm.add_load(mid_node, fz=load_per_node) via build_frame's existing,
    unmodified load_idx loop). A transverse point load applied at a beam's
    own mid-span is, by elementary statics, IMPOSSIBLE to carry with zero
    shear/moment anywhere along that beam's length regardless of any
    left-right mirror symmetry in the rest of the structure -- so this is
    not a numerical patch but a genuinely different (and, we would argue,
    MORE realistic) load idealisation: a real notched-timber deck actually
    rests on and is supported BY the transverse rungs spanning between
    rails, so the deck's self-weight/live load naturally acts on the
    rungs' own span, not solely at the rail-joint nodes.

    Options considered and rejected (documented per spec):
      (a) fully continuous single-member rung bearing reciprocally on all
          3 rails via notch links instead of split half-members -- rejected
          because it would remove the split-rung member-diversity feature
          that was itself a deliberate design goal of the original 3-rail
          generator, and does not by itself change the LOAD PATH (a
          continuous rung with load applied only at its own rail-bearing
          ends is subject to the exact same zero-transverse-load argument
          that broke the split version).
      (c) staggering the 3 rails so the weave forces engagement -- rejected
          as a much larger geometry change (breaks the straight, single
          arch-profile rail alignment that gives this typology its
          Leonardo-bridge reading) for no clearer physical motivation than
          option (b).
      (d) dropping the centre rail -- rejected: the centre rail's own
          LONGITUDINAL members are not in the sub-floor list (only the
          rungs are); dropping it would remove real structure (and half of
          the campaign's richer 16-group member diversity) to fix a defect
          that is confined entirely to how the rungs are LOADED, not
          whether the centre rail exists.

    Also increases n_bays (24 vs the original's 12) so total physical
    member count clears the redesign's own 60-120 target (118, vs the
    original's 58) while keeping the identical rail/quarter/rung-parity
    grouping formula (still exactly 16 groups).

    n_bays=24 CALIBRATION (empirical, disclosed): an earlier draft of this
    redesign used n_bays=14 (68 members). The V1-V5 verification battery
    (results/bridge2_verification.json) found that, while the mid-node
    load fix (above) cleared the ORIGINAL defect completely under the
    project's own service-load convention (min strain-energy share 8.37%
    of the mean at n_bays=14, comfortably above the 5% floor, vs the
    original's 22/58 members at ~1e-23%), exactly 2 of 68 members (the
    rung half-segments at the station immediately next to each abutment)
    fell marginally BELOW the 5% floor under the SEPARATE self-weight-only
    load case (4.58%) -- a real but tiny elastic-response effect (near-
    support members simply deflect less under a uniform self-weight
    load), not a reproduction of the original zero-force artifact. A
    direct empirical sweep of n_bays in {12,14,...,24} (holding every
    other parameter fixed) showed the self-weight-only minimum share rises
    monotonically with n_bays (4.38% at 12 -> 4.58% at 14 -> 4.95% at 22
    -> 5.02% at 24), crossing the 5% floor at n_bays=24 while the
    service-load share stays comfortably clear (6.31% at n_bays=24, vs
    8.37% at n_bays=14) and the member count (118) remains inside the
    60-120 target. n_bays=24 was therefore adopted as the new default so
    BOTH load cases clear the floor; see bridge2_verification.json for the
    full V3 numbers under both conventions.

    Member groups: unchanged formula from davinci_bridge_complex (12
    rail-x-quarter longitudinal groups + 4 rung-parity-x-half groups = 16).
    Member count: 3*n_bays longitudinal + 2*(n_bays-1) rung half-segments
    (n_bays=24 => 72+46 = 118 physical members).
    Supports: abutment (x=0, x=span) ends of all 3 rails, 6 total; zero
    interior supports (unchanged).
    Load: vertical, at every rail deck node (unchanged, 3/station) PLUS
    every rung half-segment's own new mid-span node (NEW, 2/station).
    """
    if n_bays < 4:
        raise ValueError("davinci_bridge2 needs n_bays >= 4 for at least "
                          "one interior rung station")
    if rise is None:
        rise = 0.12 * span
    model = RFModel()
    bay = span / n_bays
    ecc = ecc_ratio * (lam / 0.25) * bay
    xs = np.linspace(0.0, span, n_bays + 1)
    a = span / 2.0
    Rc = (a ** 2 + rise ** 2) / (2 * rise)

    def z_of(x):
        dx = x - a
        val = max(Rc ** 2 - dx ** 2, 0.0)
        return rise - (Rc - np.sqrt(val))

    ys = [-width / 2.0, 0.0, width / 2.0]  # 3 rails: left, centre, right
    n_rails = len(ys)
    long_start = {}
    long_end = {}
    for si, y in enumerate(ys):
        for i in range(n_bays):
            x0, x1 = xs[i], xs[i + 1]
            if i == 0:
                n0 = model.add_node((x0, y, z_of(x0)))
                model.support_idx.add(n0)
            else:
                n0 = model.add_node((x0, y, z_of(x0) + ecc / 2.0))
                long_start[(si, i)] = n0
            if i == n_bays - 1:
                n1 = model.add_node((x1, y, z_of(x1)))
                model.support_idx.add(n1)
            else:
                n1 = model.add_node((x1, y, z_of(x1) - ecc / 2.0))
                long_end[(si, i)] = n1
            quarter = min(3, (i * 4) // n_bays)
            grp = si * 4 + quarter
            model.elements.append(dict(i=n0, j=n1, kind='beam', group=grp,
                                        member_id=('long', si, i)))

    n_mid_loaded = 0
    for i in range(1, n_bays):
        x = xs[i]
        z = z_of(x)
        rail_nodes = [model.add_node((x, y, z)) for y in ys]
        parity = i % 2
        for node in rail_nodes:
            model.load_idx.append(node)
        for seg_idx in range(n_rails - 1):
            a_node = rail_nodes[seg_idx]
            b_node = rail_nodes[seg_idx + 1]
            ya, yb = ys[seg_idx], ys[seg_idx + 1]
            mid_node = model.add_node((x, 0.5 * (ya + yb), z))
            rung_grp = 12 + parity * 2 + seg_idx
            mid_id = ('rung', i, seg_idx)
            model.elements.append(dict(i=a_node, j=mid_node, kind='beam',
                                        group=rung_grp, member_id=mid_id))
            model.elements.append(dict(i=mid_node, j=b_node, kind='beam',
                                        group=rung_grp, member_id=mid_id))
            # REDESIGN fix: ADD the deck load at the rung half-segment's own
            # mid-span node (rail-node loads above are UNCHANGED/kept) --
            # forces genuine bending/shear in every rung regardless of
            # left-right symmetry (see docstring).
            model.load_idx.append(mid_node)
            n_mid_loaded += 1
        for si, node_side in enumerate(rail_nodes):
            incoming = long_end[(si, i - 1)]
            outgoing = long_start[(si, i)]
            model.elements.append(dict(i=node_side, j=incoming, kind='link', group=None))
            model.elements.append(dict(i=node_side, j=outgoing, kind='link', group=None))

    model.n_groups = 16
    model.meta = dict(kind='davinci_bridge2', n_bays=n_bays, span=span,
                       width=width, rise=rise, lam=lam, ecc=ecc,
                       load_per_node=load_per_node, n_rails=n_rails,
                       load_application='rail_deck_nodes_PLUS_rung_midspans',
                       n_rung_midspan_loaded_nodes=n_mid_loaded,
                       redesign_of='davinci_bridge_complex',
                       redesign_reason='strain_energy_floor defect (22/58 rung '
                                       'half-segments ~0 force); see '
                                       'results/bridge_diagnosis.json')
    return model


def verify_bridge2_bearings(model, label=""):
    """Gate 3 for davinci_bridge2: same longitudinal-bearing check as
    verify_bridge_bearings_complex, generalised for the rung's now-2-segment
    (rail_node -> mid_node -> rail_node) physical-member chain: each rung's
    two OUTER (rail-node) chain ends must each carry exactly 2 link-element
    bearings from their own rail's longitudinal chain, exactly as before --
    the new interior mid-node is a genuine mid-span frame joint (not a
    'bearing' point), so it is correctly excluded from this bearing count."""
    link_partners = {}
    for e in model.elements:
        if e['kind'] == 'link':
            link_partners.setdefault(e['i'], []).append(e['j'])
            link_partners.setdefault(e['j'], []).append(e['i'])

    members = physical_members_v2(model)
    by_id = {m['member_id']: m for m in members}

    n_bays = model.meta['n_bays']
    n_rails = model.meta['n_rails']
    bad = []
    n_interior_long = 0
    for si in range(n_rails):
        for i in range(n_bays):
            mid = ('long', si, i)
            m = by_id[mid]
            ni, nj = m['chain'][0], m['chain'][-1]
            is_abutment_start = ni in model.support_idx
            is_abutment_end = nj in model.support_idx
            if is_abutment_start or is_abutment_end:
                continue
            n_interior_long += 1
            end_bearings = sum(1 for n in (ni, nj) if len(link_partners.get(n, [])) >= 1)
            if end_bearings != 2:
                bad.append((mid, end_bearings))

    n_rung_segments = (n_bays - 1) * (n_rails - 1)
    rung_bad = []
    for i in range(1, n_bays):
        for seg_idx in range(n_rails - 1):
            mid = ('rung', i, seg_idx)
            m = by_id[mid]
            outer_a, outer_b = m['chain'][0], m['chain'][-1]
            for n in (outer_a, outer_b):
                if len(link_partners.get(n, [])) != 2:
                    rung_bad.append((mid, n, len(link_partners.get(n, []))))

    ok = (len(bad) == 0) and (len(rung_bad) == 0)
    print(f"[gate3-bridge2:{label}] interior_longitudinals={n_interior_long} "
          f"malformed_long={len(bad)} n_rung_segments={n_rung_segments} "
          f"malformed_rung_ends={len(rung_bad)} ok={ok}")
    return dict(ok=ok, n_interior_long=n_interior_long, malformed_long=bad,
                n_rung_segments=n_rung_segments, malformed_rung_ends=rung_bad)



# --------------------------------------------------------------------------
# 6. rf_vault -- barrel-vault mapping of the rotated-square nexorade (NEW)
# --------------------------------------------------------------------------
def rf_vault(nx=10, ny=8, cell_x=2.0, span_transverse=14.0, rise=2.6,
             lam_bands=(0.15, 0.2, 0.25, 0.3, 0.35), ecc_ratio=0.07,
             load_per_node=-2200.0, chirality=-1, n_section_groups=10):
    """Reciprocal barrel VAULT: a rectangular nx x ny patch of the SQUARE
    (rotated-square / pinwheel) nexorade pattern -- i.e. exactly
    square_nexorade's per-hub-fan + member-fusion construction (every
    strictly-interior grid vertex is a degree-4 hub-fan; every grid EDGE
    becomes ONE physical member, fused across its two hub-fans if interior,
    pinned/free at its own outer end if not) -- mapped onto a cylindrical
    (single-curvature) shell instead of the flat plane: z(y) is a circular-
    arc profile across the TRANSVERSE width (span_transverse, the vault's
    springing-to-springing chord), extruded unchanged along the LENGTH
    direction (x). Members stay straight chords between their mapped 3D
    endpoints, exactly like rf_dome's spherical-cap mapping of the
    triangular hex_nexorade pattern (see that function's docstring) --
    same idea, single instead of double curvature, square instead of
    triangular base pattern.

    Construction-choice note (square vs triangular base pattern): the
    square/rotated-square pattern (square_nexorade's degree-4 vertex hubs)
    was chosen over the triangular/hex pattern (rf_dome's degree-3 face
    hubs) because a rectangular nx-by-ny grid of degree-4 hubs maps onto a
    barrel-vault's own natural (length x transverse) rectangular topology
    with NO left-over partial cells at the two free (gable) ends -- every
    hub along ix=1 or ix=nx-1 simply loses its outward (E or W) neighbour
    to a FREE (unsupported) longitudinal-end node instead of a second hub,
    with no special-casing needed beyond the direction check below. The
    triangular pattern's alternating up/down face orientation has no
    natural "straight free edge" analogue at a rectangular boundary (its
    hex_nexorade/rf_dome boundary is a ring, not two straight edges), so it
    was judged less robust for this particular free-longitudinal-edge
    requirement; the square pattern reuses square_nexorade's
    already-validated hub/fusion/bearing machinery unchanged (just a
    different pos()/lift() and a direction-dependent support rule), which
    is the most robust path per the spec's own guidance to "pick whichever
    fuses more robustly".

    Supports: ONLY at the two straight LONGITUDINAL edges (the springing
    lines, iy=0 and iy=ny, i.e. constant transverse position at each
    extreme of the arc) -- realised, as in square_nexorade, as the pinned
    OUTER end of every boundary member whose grid neighbour direction was
    the TRANSVERSE (N/S, dy!=0) one. Grid neighbours reached in the
    LENGTH direction (E/W, dx!=0) that fall outside the hub range (ix=0 or
    ix=nx, i.e. the two gable/open ends of the barrel) are built exactly
    the same way (a genuine physical member, still a real 2-segment
    support->engagement->tip chain with its own end bearing) but are
    deliberately left UNSUPPORTED (no pin) -- these are the vault's own
    free longitudinal edges, matching a real barrel vault/shell resting on
    two continuous side walls with open, unsupported ends (c.f. a Nissen
    hut / Quonset-type barrel shell). Zero interior supports (identical
    guarantee to every other v2 generator, re-verified by
    verify_no_interior_supports unchanged).

    Section grouping / lambda-family bands: TWO INDEPENDENT partitions of
    the same transverse index range iy in [0, ny], both keyed purely by
    the hub's own iy (transverse position), NOT by direction:
      - `n_section_groups` (spec: ~8-12) FINE bands -> model.n_groups,
        the section-sizing group id (one CHS section catalog index per
        band, shared by both E/W and N/S members whose hub falls in that
        band) -- a genuine, but coarser-than-the-hex-generators',
        band/ring style member grouping.
      - `lam_bands` (spec: 4-6 elements) COARSE bands -> a distinct
        engagement ratio lambda per band-family, exactly analogous to
        rf_dome's lam_per_ring but indexed by transverse band instead of
        radial ring -- this is the black-box geometry-search variable
        used by CS-VAULT-GEO (PART 2), giving the vault a genuinely
        richer (4-6-dimensional, vs a single dome-family sub-count)
        geometry-variable space per the task's requirement.

    Note (identical in spirit to davinci_bridge's): the arc
    profile z(y) is PRESCRIBED geometrically (circular segment through the
    given span_transverse/rise, straight chords between grid stations),
    exactly like davinci_bridge's/rf_dome's own prescribed-profile
    simplification -- what is modelled faithfully is the reciprocal
    bearing PATTERN (per-hub fan, member fusion, +-e/2 bearings),
    unchanged from square_nexorade.
    """
    if nx < 3 or ny < 3:
        raise ValueError("rf_vault needs nx,ny >= 3 for at least one interior hub")
    n_lambda_bands = len(lam_bands)
    cell_y = span_transverse / ny
    length = nx * cell_x
    a = span_transverse / 2.0
    Rc = (a ** 2 + rise ** 2) / (2 * rise)

    def z_of(y):
        dy = y - a
        val = max(Rc ** 2 - dy ** 2, 0.0)
        return rise - (Rc - np.sqrt(val))

    def pos(ix, iy):
        return (ix * cell_x, iy * cell_y, z_of(iy * cell_y))

    def is_hub(ix, iy):
        return 1 <= ix <= nx - 1 and 1 <= iy <= ny - 1

    def lam_band_of(iy):
        frac = iy / ny
        return min(int(frac * n_lambda_bands), n_lambda_bands - 1)

    def sec_band_of(iy):
        frac = iy / ny
        return min(int(frac * n_section_groups), n_section_groups - 1)

    model = RFModel()
    ecc = ecc_ratio * float(np.sqrt(cell_x * cell_y))
    midg = _MemberIdGen()
    NEI_DIRS = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # cyclic E, N, W, S (N/S = transverse)

    boundary_node = {}

    def boundary_gid(ix, iy):
        key = (ix, iy)
        if key not in boundary_node:
            boundary_node[key] = model.add_node(pos(ix, iy))
        return boundary_node[key]

    hubs = [(ix, iy) for ix in range(1, nx) for iy in range(1, ny)]
    spoke = {}  # (hub_key, neighbor_key) -> dict(lower=gid, tip=gid, group=g)

    for (ix, iy) in hubs:
        neigh = [(ix + dx, iy + dy) for dx, dy in NEI_DIRS]
        support_pts = [pos(*nk) for nk in neigh]
        fan_lam = lam_bands[lam_band_of(iy)]
        fan = pinwheel_fan(support_pts, fan_lam, ecc, chirality=chirality)
        loc2glob = {}

        def gid(li, _map=loc2glob, _fan=fan):
            if li not in _map:
                _map[li] = model.add_node(_fan['nodes'][li])
            return _map[li]

        for (i, j, kind, bid) in fan['elements']:
            if kind == 'link':
                model.elements.append(dict(i=gid(i), j=gid(j), kind='link', group=None))

        group = sec_band_of(iy)
        for k, (s_local, l_local, u_local) in enumerate(_beam_pairs(fan)):
            lower_g = gid(l_local)
            upper_g = gid(u_local)
            neighbor_key = neigh[k]
            dxk, dyk = NEI_DIRS[k]
            if is_hub(*neighbor_key):
                spoke[((ix, iy), neighbor_key)] = dict(lower=lower_g, tip=upper_g, group=group)
            else:
                bnd_g = boundary_gid(*neighbor_key)
                if dyk != 0:
                    # transverse (N/S) direction: this grid edge reaches a
                    # springing line -> pin it (the vault's ONLY supports).
                    model.support_idx.add(bnd_g)
                # else dxk != 0 (E/W, length direction): this grid edge
                # reaches a gable/open end -> a genuine free (unsupported)
                # member end, deliberately left unpinned.
                mid = midg.new()
                model.elements.append(dict(i=bnd_g, j=lower_g, kind='beam', group=group, member_id=mid))
                model.elements.append(dict(i=lower_g, j=upper_g, kind='beam', group=group, member_id=mid))
                model.load_idx.append(upper_g)

    processed = set()
    for (hub_key, nb_key), info_a in list(spoke.items()):
        ek = frozenset({hub_key, nb_key})
        if ek in processed:
            continue
        info_b = spoke.get((nb_key, hub_key))
        if info_b is None:
            continue
        processed.add(ek)
        mid = midg.new()
        model.elements.append(dict(i=info_a['tip'], j=info_a['lower'], kind='beam', group=info_a['group'], member_id=mid))
        model.elements.append(dict(i=info_a['lower'], j=info_b['lower'], kind='beam', group=info_a['group'], member_id=mid))
        model.elements.append(dict(i=info_b['lower'], j=info_b['tip'], kind='beam', group=info_b['group'], member_id=mid))
        model.load_idx.append(info_a['tip'])
        model.load_idx.append(info_b['tip'])

    model.n_groups = n_section_groups
    model.meta = dict(kind='rf_vault', nx=nx, ny=ny, cell_x=cell_x, cell_y=cell_y,
                       span_transverse=span_transverse, rise=rise,
                       rise_span_ratio=rise / span_transverse, length=length,
                       lam_bands=[float(l) for l in lam_bands],
                       n_lambda_bands=n_lambda_bands, n_section_groups=n_section_groups,
                       ecc=ecc, load_per_node=load_per_node,
                       span=span_transverse)
    return model


# --------------------------------------------------------------------------
# 7. rf_vault_hex -- barrel-vault mapping of the HEXAGONAL/TRIANGULAR
#    face-hub nexorade pattern (NEW; upgrades rf_vault's square/rotated-
#    square pattern to the same 3-beam-fan, face-hub construction already
#    used by hex_nexorade/rf_dome, per _hex_nexorade_core's docstring).
# --------------------------------------------------------------------------
def _axial_rect_patch(length, span_transverse, n_rows):
    """A RECTANGULAR PATCH of the exact same true 60-degree axial
    triangular lattice `_hex_axial_nodes`/`_hex_triangles` use for
    `hex_nexorade`/`rf_dome` (x = edge*(q + r/2), y = edge*r*sqrt(3)/2),
    just CLIPPED to a rectangle [0,length] x [0,span_transverse] instead
    of a hexagonal ring region -- i.e. the genuinely equilateral-triangle
    lattice (not a sheared/right-triangle Cartesian criss-cross), so
    interior vertices are surrounded by 6 REGULAR triangular faces and the
    assembled reciprocal weave reads as true hexagonal engagement windows,
    matching `rf_dome`'s own look (see qa_vault_hex QA figures).

    `edge` is RE-DERIVED from `n_rows` and `span_transverse` so that lattice
    row r=n_rows lands EXACTLY on y=span_transverse (row r=0 is always
    y=0) -- rows are lines of constant r, and since y depends only on r
    (not q), every row is perfectly straight/horizontal, so the two
    springing lines (y=0, y=span_transverse) are clean, unbroken straight
    boundaries ideal for a continuous line of supports. The two length-
    direction (x=0, x=length) boundaries, by contrast, are necessarily
    JAGGED/stepped in a sheared axial lattice (x depends on both q and r)
    -- deliberately left that way and simply treated as the vault's free
    (unsupported) gable ends, exactly mirroring `rf_vault`'s own straight-
    springing-line / free-gable-end idiom, just with a jagged instead of
    straight free edge (an stated consequence of using the true
    hex lattice on a rectangle -- see `rf_vault_hex`'s own docstring).

    Returns (coords: dict[(q,r)] -> (x,y), faces: list[(a,b,c)] of (q,r)
    triples, edge_realized: float)."""
    e = span_transverse / (n_rows * np.sqrt(3) / 2.0)
    coords = {}
    for r in range(0, n_rows + 1):
        y = e * r * np.sqrt(3) / 2.0
        q_lo = int(np.floor(-r / 2.0)) - 1
        q_hi = int(np.ceil((length / e) - r / 2.0)) + 1
        for q in range(q_lo, q_hi + 1):
            x = e * (q + r / 2.0)
            if -1e-9 <= x <= length + 1e-9:
                coords[(q, r)] = (x, y)
    faces = []
    for (q, r) in list(coords.keys()):
        a, b, c = (q, r), (q + 1, r), (q, r + 1)
        if b in coords and c in coords:
            faces.append((a, b, c))
        d = (q + 1, r + 1)
        if b in coords and c in coords and d in coords:
            faces.append((b, d, c))
    return coords, faces, e


def _trim_free_end_members(model):
    """Post-build cleanup pass: strip `rf_vault_hex`'s structurally-negligible
    FREE-END gable members (see STEP1 diagnosis,
    results/vault_hex_free_end_diagnosis.json): as a class, the reference
    geometry's 17 free-end members carry ~58x less mean per-member strain
    energy than interior members (0.667 vs 38.8, uniform mid-catalog
    section, service load) and only ~0.23% of the whole structure's total
    strain energy combined -- an unsurprising consequence of their outer
    chain node being, by construction, a genuine dead stub (degree 1, "in
    mid-air") that bears on nothing else in the mesh. Removing them trims
    dead weight/member count without touching the shell's real load path.

    Generic, geometry-index-independent graph cleanup driven purely by
    `physical_members_v2`'s member_id/chain reconstruction and
    `model.support_idx` (works for any RFModel using this module's
    member_id convention, not hardcoded to rf_vault_hex's own fan
    indexing):
      1. A FREE-END member = a chain of length 3 (see
         `verify_vault_bearings`) whose outer node is NOT in
         `model.support_idx` (that outer-node-is-a-support case is the
         DISTINCT pinned-boundary/springing-line member, always kept).
      2. Remove that member's 2 'beam' sub-elements; drop its loaded tip's
         entry from load_idx (no orphaned loads on removed nodes).
      3. Iteratively prune any node that is not a support, carries no load,
         and has zero remaining 'beam'-kind incident elements (i.e. only
         eccentricity 'link' elements, or nothing, still touch it) --
         removing each such node's remaining incident 'link' element(s)
         with it. This is exactly the now-isolated outer boundary node PLUS
         the now-purposeless engagement-node/'link' stub each trimmed
         member leaves at its inner end (the bearing point where the
         trimmed member's tip used to rest against/under its neighbour
         beam -- with the trimmed member gone, that bearing contact is
         gone too and would otherwise be left as a dangling stub). This
         pass never touches a node that is still an endpoint of any KEPT
         beam element, so every remaining physical member's own bearing
         chain -- interior fused members, pinned-boundary members, and any
         neighbouring free-end member's own tip that happens to still be
         real -- is left completely intact.
      4. Remaining nodes are renumbered (dead nodes dropped) and
         elements/support_idx/load_idx remapped accordingly.

    Returns (model, trim_meta) where trim_meta reports what was removed.
    """
    members = physical_members_v2(model)
    support_set = model.support_idx

    free_member_ids = set()
    for m in members:
        chain = m['chain']
        if len(chain) != 3:
            continue
        end_a, end_b = chain[0], chain[-1]
        if (end_a in support_set) or (end_b in support_set):
            continue
        free_member_ids.add(m['member_id'])

    if not free_member_ids:
        return model, dict(n_trimmed=0, trimmed_member_ids=[], n_nodes_removed=0)

    n_nodes_before = len(model.nodes)
    n_elems_before = len(model.elements)

    elems_to_remove = set()
    for ei, e in enumerate(model.elements):
        if e['kind'] == 'beam' and e['member_id'] in free_member_ids:
            elems_to_remove.add(ei)

    tips_to_unload = set()
    for m in members:
        if m['member_id'] in free_member_ids:
            tips_to_unload.add(m['chain'][-1])

    kept_elems = [e for ei, e in enumerate(model.elements) if ei not in elems_to_remove]

    load_count = {}
    for ni in model.load_idx:
        load_count[ni] = load_count.get(ni, 0) + 1
    new_load_idx = []
    for ni in model.load_idx:
        if ni in tips_to_unload and load_count.get(ni, 0) > 0:
            load_count[ni] -= 1
            continue
        new_load_idx.append(ni)
    loaded_nodes = set(new_load_idx)

    def beam_degree(elems):
        deg = {}
        for e in elems:
            if e['kind'] != 'beam':
                continue
            deg[e['i']] = deg.get(e['i'], 0) + 1
            deg[e['j']] = deg.get(e['j'], 0) + 1
        return deg

    changed = True
    while changed:
        changed = False
        beam_deg = beam_degree(kept_elems)
        touched_nodes = set()
        for e in kept_elems:
            touched_nodes.add(e['i']); touched_nodes.add(e['j'])
        removable = set(
            ni for ni in touched_nodes
            if ni not in support_set and ni not in loaded_nodes and beam_deg.get(ni, 0) == 0
        )
        if not removable:
            break
        new_kept = []
        for e in kept_elems:
            if e['i'] in removable or e['j'] in removable:
                changed = True
                continue
            new_kept.append(e)
        kept_elems = new_kept

    referenced = set()
    for e in kept_elems:
        referenced.add(e['i']); referenced.add(e['j'])
    keep_old = sorted(referenced | (support_set & referenced) | (loaded_nodes & referenced))
    # (support/loaded nodes with zero remaining elements cannot occur for
    # this construction -- supports and loaded tips always keep >=1 beam
    # element -- but guard defensively rather than silently dropping one.)
    keep_old = sorted(set(keep_old) | (support_set) | loaded_nodes)
    remap = {old: new for new, old in enumerate(keep_old) if old < n_nodes_before}

    new_nodes = [model.nodes[old] for old in keep_old if old < n_nodes_before]
    new_elems = []
    for e in kept_elems:
        e2 = dict(e)
        e2['i'] = remap[e['i']]
        e2['j'] = remap[e['j']]
        new_elems.append(e2)
    new_support_idx = set(remap[s] for s in support_set if s in remap)
    new_load_idx_remapped = [remap[ni] for ni in new_load_idx if ni in remap]

    model.nodes = new_nodes
    model.elements = new_elems
    model.support_idx = new_support_idx
    model.load_idx = new_load_idx_remapped

    return model, dict(n_trimmed=len(free_member_ids),
                        trimmed_member_ids=sorted(free_member_ids),
                        n_nodes_before=n_nodes_before, n_nodes_after=len(new_nodes),
                        n_nodes_removed=n_nodes_before - len(new_nodes),
                        n_elems_before=n_elems_before, n_elems_after=len(new_elems),
                        n_elems_removed=n_elems_before - len(new_elems))


def rf_vault_hex(n_rows=7, length=17.0, span_transverse=14.0, rise=2.6,
                  lam_bands=(0.15, 0.2, 0.25, 0.3, 0.35), ecc_ratio=0.07,
                  load_per_node=-2200.0, chirality=-1, n_section_groups=10,
                  trim_free_ends=True):
    """Reciprocal barrel VAULT, HEXAGONAL/TRIANGULAR pattern: upgrades
    `rf_vault`'s rotated-SQUARE (degree-4 vertex-hub) base pattern to the
    same 3-beam, FACE-hub nexorade construction used by `hex_nexorade`/
    `rf_dome` (`_hex_nexorade_core`'s "hub = triangular face, 3 supports =
    its 3 edge-midpoints" idiom -- see that function's docstring), mapped
    onto the identical cylindrical single-curvature surface `rf_vault`
    already uses (`z(y)` circular-arc profile across the transverse width,
    extruded unchanged along the length; straight chords between mapped 3D
    endpoints).

    Base tessellation: `_axial_rect_patch(edge, length, span_transverse,
    n_rows)` -- a RECTANGULAR PATCH of the SAME true 60-degree axial
    triangular lattice `hex_nexorade`/`rf_dome` use (not a sheared
    Cartesian criss-cross), giving every strictly-interior grid VERTEX
    degree 6 (six alternating up/down EQUILATERAL triangular fans meeting
    around it) -- i.e. genuine hexagonal engagement WINDOWS in the
    assembled weave, denser than `rf_vault`'s 4-per-vertex square windows
    at a comparable member budget (see module/QA-figure discussion). Every
    grid EDGE becomes exactly one physical member; edges with 2 incident
    faces are FUSED (interior); edges with exactly 1 incident face sit on
    the rectangle's own outer boundary.

    Supports: ONLY at the two straight TRANSVERSE-boundary (springing)
    lines -- the axial lattice's row r=0 and r=n_rows are, by
    construction, perfectly straight lines of constant y (see
    `_axial_rect_patch`'s docstring), so boundary edges connecting two
    same-row (q,0)-(q+1,0) or (q,n_rows)-(q+1,n_rows) vertices are PINNED
    at their outer end, exactly like `rf_vault`'s springing-line
    convention. ALL OTHER boundary edges (the jagged x=0/x=length gable
    ends, an consequence of clipping a sheared axial lattice to a
    rectangle) are built as genuine 2-segment member chains but
    deliberately left UNPINNED (free tip) -- the vault's free gable ends.
    Zero interior supports (identical guarantee to every other v2
    generator).

    Section grouping / lambda-family bands: identical in spirit to
    `rf_vault`'s two independent transverse-position partitions, just
    keyed by a face's mean transverse LATTICE ROW `(r_a+r_b+r_c)/3`
    instead of a single hub's own iy: `n_section_groups` (spec 8-12) fine
    bands -> `model.n_groups`; `lam_bands` (spec 4-6 elements) coarse
    engagement-ratio band-families -> the CS-VAULT-HEX-GEO black-box
    search variable (Part 2).

    Defaults: n_rows=7, length=17.0 m, span_transverse=14.0 m (matching
    `rf_vault`'s own span_transverse and rise exactly; length is shorter
    than `rf_vault`'s 20 m -- a deliberate, stated trade-off: this denser
    3-beam-fan pattern needs MORE lattice rows (n_rows=7 vs `rf_vault`'s
    ny=8 transverse bays are comparable in count, but each hex-lattice row
    carries much more member/DOF weight than a square-grid row) to keep
    both (a) the free-gable-end cantilever flexibility acceptable (see
    below) and (b) restrained DOF under the spec's ~3400 soft cap, so
    `length` was shortened at fixed n_rows=7 until both constraints were
    met simultaneously -- see `results/vault_hex_geometry_stats.json`
    for the realized counts and the gate4 discussion.

    Gate-4 (deflection window) note, stated UPDATED after the
    `trim_free_ends` cleanup (default True) below: with the JAGGED
    x=0/x=length free gable ends left in (trim_free_ends=False), this
    pattern's un-triangulated free tips made the shell noticeably MORE
    flexible near the gable ends than `rf_vault`'s own straight, fully-
    triangulated free edges; at the DEFAULT geometry the reference
    lam=0.25 case still passed gate4 there, while lam=0.15 (smaller
    engagement ratio -> bearing points sit closer to each fan's own
    centre, softening every joint) FAILED gate4 (too-flexible direction,
    d_max above the span/100 upper bound). A per-physical-member strain-
    energy audit (uniform mid-catalog section, service load; see
    `results/vault_hex_free_end_diagnosis.json`) showed those 17 free
    gable members carry, as a class, ~58x less mean strain energy than
    interior members (~0.23% of the whole structure's total strain energy
    combined) -- an unsurprising consequence of their outer end being a
    genuine unsupported dead stub bearing on nothing else in the mesh --
    so `trim_free_ends=True` now strips them by default (see that
    parameter/`_trim_free_end_members`'s own docstring). With them
    removed the lam=0.15 gate4 failure DISAPPEARS (d_max drops back
    inside the span/2000..span/100 window at every swept lambda, and the
    reference lam=0.25 case's own d_max drops too, since the free ends'
    extra flexibility no longer contaminates the whole-structure
    displacement max) -- see `results/vault_hex_geometry_stats.json`'s
    current gate4 block for the exact post-trim numbers, and
    `results/vault_hex_geometry_stats_pre_trim.json` for the untrimmed
    numbers this replaced.

    Note (identical in spirit to `rf_vault`'s): the arc profile
    z(y) is PRESCRIBED geometrically (circular segment through the given
    span_transverse/rise, straight chords between grid stations); what is
    modelled faithfully is the reciprocal bearing PATTERN (per-face-hub
    3-beam fan, member fusion, +-e/2 bearings), unchanged in kind from
    `hex_nexorade`/`rf_dome`, just re-keyed to a rectangular instead of
    hexagonal outer boundary and a cylindrical instead of planar/spherical
    surface map."""
    if n_rows < 2:
        raise ValueError("rf_vault_hex needs n_rows >= 2 for at least one interior-row hub")
    n_lambda_bands = len(lam_bands)
    a = span_transverse / 2.0
    Rc = (a ** 2 + rise ** 2) / (2 * rise)

    def z_of(y):
        dy = y - a
        val = max(Rc ** 2 - dy ** 2, 0.0)
        return rise - (Rc - np.sqrt(val))

    coords, faces, edge_len = _axial_rect_patch(length, span_transverse, n_rows)

    def pos(key):
        x, y = coords[key]
        return (x, y, z_of(y))

    def midpoint(p, q):
        pa, pb = np.array(pos(p)), np.array(pos(q))
        return tuple(0.5 * (pa + pb))

    edge_faces = {}
    for fi, (va, vb, vc) in enumerate(faces):
        for (p, q) in ((va, vb), (vb, vc), (vc, va)):
            edge_faces.setdefault(frozenset({p, q}), []).append(fi)

    def band_of(va, vb, vc, n_bands):
        r_avg = (va[1] + vb[1] + vc[1]) / 3.0
        frac = r_avg / n_rows
        return min(int(frac * n_bands), n_bands - 1)

    model = RFModel()
    ecc = ecc_ratio * edge_len
    midg = _MemberIdGen()

    boundary_node = {}

    def boundary_gid(mid_pt):
        key = tuple(np.round(mid_pt, 9))
        if key not in boundary_node:
            boundary_node[key] = model.add_node(mid_pt)
        return boundary_node[key]

    spoke = {}  # frozenset(edge_verts) -> list of dict(lower, tip, group)

    for fi, (va, vb, vc) in enumerate(faces):
        dir_edges = ((va, vb), (vb, vc), (vc, va))
        support_pts = [midpoint(p, q) for (p, q) in dir_edges]
        lam_band = band_of(va, vb, vc, n_lambda_bands)
        fan_lam = lam_bands[lam_band]
        fan = pinwheel_fan(support_pts, fan_lam, ecc, chirality=chirality)
        loc2glob = {}

        def gid(li, _map=loc2glob, _fan=fan):
            if li not in _map:
                _map[li] = model.add_node(_fan['nodes'][li])
            return _map[li]

        for (i, j, kind, bid) in fan['elements']:
            if kind == 'link':
                model.elements.append(dict(i=gid(i), j=gid(j), kind='link', group=None))

        group = band_of(va, vb, vc, n_section_groups)
        for k, (s_local, l_local, u_local) in enumerate(_beam_pairs(fan)):
            lower_g = gid(l_local)
            upper_g = gid(u_local)
            ek = frozenset(dir_edges[k])
            if len(edge_faces[ek]) == 2:
                spoke.setdefault(ek, []).append(dict(lower=lower_g, tip=upper_g, group=group))
            else:
                p, q = dir_edges[k]
                mid_pt = midpoint(p, q)
                bnd_g = boundary_gid(mid_pt)
                if p[1] == q[1] and p[1] in (0, n_rows):
                    # same-row boundary edge at a springing line -> pin
                    # (the vault's ONLY supports).
                    model.support_idx.add(bnd_g)
                # else: a jagged x=0/x=length gable-end boundary edge ->
                # a genuine free (unsupported) member end.
                mid = midg.new()
                model.elements.append(dict(i=bnd_g, j=lower_g, kind='beam', group=group, member_id=mid))
                model.elements.append(dict(i=lower_g, j=upper_g, kind='beam', group=group, member_id=mid))
                model.load_idx.append(upper_g)

    for ek, infos in spoke.items():
        if len(infos) != 2:
            continue
        info_a, info_b = infos
        mid = midg.new()
        model.elements.append(dict(i=info_a['tip'], j=info_a['lower'], kind='beam', group=info_a['group'], member_id=mid))
        model.elements.append(dict(i=info_a['lower'], j=info_b['lower'], kind='beam', group=info_a['group'], member_id=mid))
        model.elements.append(dict(i=info_b['lower'], j=info_b['tip'], kind='beam', group=info_b['group'], member_id=mid))
        model.load_idx.append(info_a['tip'])
        model.load_idx.append(info_b['tip'])

    xs = [coords[k][0] for k in coords]
    model.n_groups = n_section_groups
    model.meta = dict(kind='rf_vault_hex', n_rows=n_rows, length=length, edge=edge_len,
                       span_transverse=span_transverse, rise=rise,
                       rise_span_ratio=rise / span_transverse,
                       length_realized=max(xs) - min(xs),
                       n_faces=len(faces), n_lattice_nodes=len(coords),
                       lam_bands=[float(l) for l in lam_bands],
                       n_lambda_bands=n_lambda_bands, n_section_groups=n_section_groups,
                       ecc=ecc, load_per_node=load_per_node,
                       span=span_transverse)

    trim_meta = dict(n_trimmed=0, trimmed_member_ids=[], n_nodes_removed=0)
    if trim_free_ends:
        model, trim_meta = _trim_free_end_members(model)
    model.meta['trim_free_ends'] = bool(trim_free_ends)
    model.meta['trim_free_ends_meta'] = trim_meta
    return model


def verify_vault_bearings(model, label=""):
    """Gate 3, vault idiom: rf_vault's physical members fall into THREE
    categories (vs square_nexorade's two), because its two length-
    direction (gable) ends are deliberately left UNPINNED:
      - interior fused member: chain length 4 (tip-lower-lower-tip), not
        touching any support -- identical check to verify_member_bearings.
      - pinned boundary member (springing line): chain length 3
        (pin-lower-tip) -- identical check to verify_member_bearings.
      - free-end boundary member (gable/open end): chain length 3
        (free_tip-lower-tip), but is_boundary is FALSE per
        physical_members_v2 (neither end is a support node) since its
        outer end is a genuine free tip, not a pin -- this is the case
        verify_member_bearings does NOT expect (it treats every length-3
        chain as necessarily pinned), hence this dedicated variant."""
    members = physical_members_v2(model)
    n_interior = n_pinned = n_free = 0
    bad = []
    support_set = model.support_idx
    for m in members:
        L = len(m['chain'])
        end_a, end_b = m['chain'][0], m['chain'][-1]
        pinned = (end_a in support_set) or (end_b in support_set)
        if L == 4 and not pinned:
            n_interior += 1
        elif L == 3 and pinned:
            n_pinned += 1
        elif L == 3 and not pinned:
            n_free += 1
        else:
            bad.append((m['member_id'], L, pinned))
    n_links = sum(1 for e in model.elements if e['kind'] == 'link')
    ok = (len(bad) == 0)
    print(f"[gate3-vault:{label}] n_members={len(members)} interior={n_interior} "
          f"pinned_boundary={n_pinned} free_end={n_free} n_link_elements={n_links} "
          f"malformed={len(bad)} ok={ok}")
    return dict(ok=ok, n_interior=n_interior, n_pinned_boundary=n_pinned,
                n_free_end=n_free, n_links=n_links, malformed=bad)


# --------------------------------------------------------------------------
# physical member reconstruction (v2 replacement for physical_beams())
# --------------------------------------------------------------------------
def physical_members_v2(model):
    """Reconstruct each PHYSICAL member's true node chain from the
    member_id tag on 'beam' elements (v2's chains are 2 sub-elements for a
    boundary member, 3 for an interior fused member -- physical_beams()'s
    fixed pairwise assumption from v1 does not hold here, hence this
    explicit, more general replacement).

    Returns a list of dicts: member_id, group, chain (ordered list of
    GLOBAL node indices, first->last), is_boundary (bool: True if either
    end of the chain is a pinned support node).
    """
    by_id = {}
    for e in model.elements:
        if e['kind'] != 'beam':
            continue
        mid = e['member_id']
        by_id.setdefault(mid, []).append((e['i'], e['j'], e['group']))

    members = []
    for mid, segs in by_id.items():
        # chain the (i,j) pairs into one ordered node list
        nodes_seq = [segs[0][0], segs[0][1]]
        remaining = segs[1:]
        changed = True
        while remaining and changed:
            changed = False
            for idx, (i, j, g) in enumerate(remaining):
                if nodes_seq[-1] == i:
                    nodes_seq.append(j)
                    remaining.pop(idx)
                    changed = True
                    break
                if nodes_seq[0] == j:
                    nodes_seq.insert(0, i)
                    remaining.pop(idx)
                    changed = True
                    break
        if remaining:
            raise ValueError(f"member {mid}: sub-elements do not form a single chain")
        end_a, end_b = nodes_seq[0], nodes_seq[-1]
        is_boundary = (end_a in model.support_idx) or (end_b in model.support_idx)
        members.append(dict(member_id=mid, group=segs[0][2], chain=nodes_seq,
                             is_boundary=is_boundary))
    return members


# --------------------------------------------------------------------------
# Validation gates
# --------------------------------------------------------------------------
def verify_nonsingular(builder_fn, lam_list, uniform_section=None, label=""):
    """Gate 1: restrained K must be non-singular for every lam in lam_list,
    at a uniform mid-catalog section."""
    if uniform_section is None:
        uniform_section = CHS_CATALOG[4]
    results = {}
    for lam in lam_list:
        model = builder_fn(lam)
        gs = {g: uniform_section for g in range(model.n_groups)}
        load_per_node = model.meta.get('load_per_node', -1500.0)
        frm = build_frame(model, gs, load_per_node=load_per_node)
        try:
            frm.solve()
            cond = np.linalg.cond(frm.K[np.ix_(
                [d for d in range(frm.ndof()) if (d // 6, d % 6) not in frm.fixed],
                [d for d in range(frm.ndof()) if (d // 6, d % 6) not in frm.fixed])])
            results[lam] = dict(ok=True, cond=float(cond), n_nodes=len(model.nodes),
                                 n_elems=len(model.elements))
        except np.linalg.LinAlgError as exc:
            results[lam] = dict(ok=False, error=str(exc))
        print(f"[gate1:{label}] lam={lam}: {results[lam]}")
    return results


def verify_no_interior_supports(model, expected_boundary_count=None, label=""):
    """Gate 2: for nexorades/dome/bridge, support_idx must contain ONLY
    boundary/abutment nodes -- i.e. every support node must be the
    outermost end of some member's chain with no fan on the far side
    (already guaranteed by construction: this function re-derives the
    count independently from physical_members_v2 and cross-checks)."""
    members = physical_members_v2(model)
    boundary_end_nodes = set()
    for m in members:
        end_a, end_b = m['chain'][0], m['chain'][-1]
        if end_a in model.support_idx:
            boundary_end_nodes.add(end_a)
        if end_b in model.support_idx:
            boundary_end_nodes.add(end_b)
    interior_supports = model.support_idx - boundary_end_nodes
    ok = len(interior_supports) == 0
    print(f"[gate2:{label}] n_supports={len(model.support_idx)} "
          f"interior_supports={len(interior_supports)} ok={ok}")
    return dict(ok=ok, n_supports=len(model.support_idx),
                n_interior_supports=len(interior_supports))


def verify_member_bearings(model, label=""):
    """Gate 3: every interior member has exactly 2 end-bearings (chain
    length 4: tip-lower-lower-tip) and every boundary member has exactly 1
    end-bearing + 1 pinned support (chain length 3: pin-lower-tip); fans
    form closed circuits (checked via: every hub contributes exactly n
    link elements forming n (upper,lower) pairs at n distinct engagement
    points -- guaranteed by pinwheel_fan and re-counted here)."""
    members = physical_members_v2(model)
    n_interior = sum(1 for m in members if not m['is_boundary'])
    n_boundary = sum(1 for m in members if m['is_boundary'])
    bad = []
    for m in members:
        L = len(m['chain'])
        if m['is_boundary'] and L != 3:
            bad.append((m['member_id'], 'boundary', L))
        if not m['is_boundary'] and L != 4:
            bad.append((m['member_id'], 'interior', L))
    n_links = sum(1 for e in model.elements if e['kind'] == 'link')
    ok = (len(bad) == 0)
    print(f"[gate3:{label}] n_members={len(members)} interior={n_interior} "
          f"boundary={n_boundary} n_link_elements={n_links} malformed={len(bad)} ok={ok}")
    return dict(ok=ok, n_interior=n_interior, n_boundary=n_boundary,
                n_links=n_links, malformed=bad)


def verify_bridge_bearings(model, label=""):
    """Gate 3, bridge idiom: davinci_bridge is not a fan/nexorade grillage
    (no closed n-beam rotors), so its "2 end bearings, closed circuit"
    check is necessarily different from verify_member_bearings(): here
    every INTERIOR longitudinal member (one per bay, per side, excluding
    the two abutment bays) must bear via a link element at BOTH its start
    (on top of the rung behind it) and its end (under the rung ahead of
    it) -- i.e. exactly 2 end bearings, matching the spec's bearing
    definition applied to a linear reciprocal chain instead of a rotor;
    every interior rung must itself receive exactly 2 link-element
    bearings on each side (incoming longitudinal underneath, outgoing
    longitudinal on top) => 4 link elements per interior rung."""
    link_partners = {}
    for e in model.elements:
        if e['kind'] == 'link':
            link_partners.setdefault(e['i'], []).append(e['j'])
            link_partners.setdefault(e['j'], []).append(e['i'])

    beam_by_member = {}
    for e in model.elements:
        if e['kind'] == 'beam':
            beam_by_member.setdefault(e['member_id'], []).append((e['i'], e['j']))

    n_bays = model.meta['n_bays']
    bad = []
    n_interior_long = 0
    for si in (0, 1):
        for i in range(n_bays):
            mid = ('long', si, i)
            (ni, nj), = beam_by_member[mid]
            is_abutment_start = ni in model.support_idx
            is_abutment_end = nj in model.support_idx
            if is_abutment_start or is_abutment_end:
                continue  # abutment bay: one end is a true pin, not a bearing
            n_interior_long += 1
            end_bearings = sum(1 for n in (ni, nj) if len(link_partners.get(n, [])) >= 1)
            if end_bearings != 2:
                bad.append((mid, end_bearings))

    n_rungs = n_bays - 1
    rung_bad = []
    for i in range(1, n_bays):
        mid = ('rung', i)
        (ni, nj), = beam_by_member[mid]
        for n in (ni, nj):
            if len(link_partners.get(n, [])) != 2:
                rung_bad.append((mid, n, len(link_partners.get(n, []))))

    ok = (len(bad) == 0) and (len(rung_bad) == 0)
    print(f"[gate3-bridge:{label}] interior_longitudinals={n_interior_long} "
          f"malformed_long={len(bad)} n_rungs={n_rungs} malformed_rung_ends={len(rung_bad)} ok={ok}")
    return dict(ok=ok, n_interior_long=n_interior_long, malformed_long=bad,
                n_rungs=n_rungs, malformed_rung_ends=rung_bad)


def verify_bridge_bearings_complex(model, label=""):
    """Gate 3 for davinci_bridge_complex: n_rails-generalisation of
    verify_bridge_bearings (that function hardcodes 2 rails / 1 rung
    segment per station; davinci_bridge_complex has model.meta['n_rails']
    rails and (n_rails-1) rung SEGMENTS per interior station). Every
    interior longitudinal member (any rail, excluding the 2 abutment bays)
    must bear via a link element at both ends; every interior rung
    segment's 2 endpoints (rail deck nodes) must each carry exactly 2
    link-element bearings from THEIR OWN rail's longitudinal chain --
    independent of how many rung segments meet at that node, so this check
    is unchanged in form from the 2-rail case, just re-indexed over
    n_rails and (n_rails-1) segments/station."""
    link_partners = {}
    for e in model.elements:
        if e['kind'] == 'link':
            link_partners.setdefault(e['i'], []).append(e['j'])
            link_partners.setdefault(e['j'], []).append(e['i'])

    beam_by_member = {}
    for e in model.elements:
        if e['kind'] == 'beam':
            beam_by_member.setdefault(e['member_id'], []).append((e['i'], e['j']))

    n_bays = model.meta['n_bays']
    n_rails = model.meta['n_rails']
    bad = []
    n_interior_long = 0
    for si in range(n_rails):
        for i in range(n_bays):
            mid = ('long', si, i)
            (ni, nj), = beam_by_member[mid]
            is_abutment_start = ni in model.support_idx
            is_abutment_end = nj in model.support_idx
            if is_abutment_start or is_abutment_end:
                continue  # abutment bay: one end is a true pin, not a bearing
            n_interior_long += 1
            end_bearings = sum(1 for n in (ni, nj) if len(link_partners.get(n, [])) >= 1)
            if end_bearings != 2:
                bad.append((mid, end_bearings))

    n_rung_segments = (n_bays - 1) * (n_rails - 1)
    rung_bad = []
    for i in range(1, n_bays):
        for seg_idx in range(n_rails - 1):
            mid = ('rung', i, seg_idx)
            (ni, nj), = beam_by_member[mid]
            for n in (ni, nj):
                if len(link_partners.get(n, [])) != 2:
                    rung_bad.append((mid, n, len(link_partners.get(n, []))))

    ok = (len(bad) == 0) and (len(rung_bad) == 0)
    print(f"[gate3-bridge-complex:{label}] interior_longitudinals={n_interior_long} "
          f"malformed_long={len(bad)} n_rung_segments={n_rung_segments} "
          f"malformed_rung_ends={len(rung_bad)} ok={ok}")
    return dict(ok=ok, n_interior_long=n_interior_long, malformed_long=bad,
                n_rung_segments=n_rung_segments, malformed_rung_ends=rung_bad)


def verify_deflection(model, uniform_section=None, span=None, label="",
                       lo_frac=2000.0, hi_frac=100.0):
    """Gate 4: d_max within span/2000..span/100 under the model's own
    service load."""
    if uniform_section is None:
        uniform_section = CHS_CATALOG[4]
    gs = {g: uniform_section for g in range(model.n_groups)}
    load_per_node = model.meta.get('load_per_node', -1500.0)
    frm = build_frame(model, gs, load_per_node=load_per_node)
    frm.solve()
    dmax = frm.max_abs_disp()
    if span is None:
        span = model.meta.get('span')
        if span is None:
            nodes = np.array(model.nodes)
            span = float(np.linalg.norm(nodes.max(axis=0) - nodes.min(axis=0)))
    lo, hi = span / lo_frac, span / hi_frac
    ok = lo <= dmax <= hi
    mass = model_mass(model, gs)
    print(f"[gate4:{label}] span={span:.2f}m dmax={dmax:.5f}m target=[{lo:.5f},{hi:.5f}] "
          f"mass={mass:.1f}kg ok={ok}")
    return dict(ok=bool(ok), dmax=float(dmax), lo=lo, hi=hi, span=span, mass=float(mass))


# --------------------------------------------------------------------------
# force-audit-driven pruning (generalizes _trim_free_end_members's orphan-
# cleanup machinery to an ARBITRARY caller-supplied member_id set, rather
# than the fixed "free-end gable class" -- see
# src/vault_hex_force_audit.py, P2/P3 of the rf_vault_hex force-path
# audit protocol, results/vault_hex_force_audit.json). Deterministic,
# geometry-index-independent: works for any RFModel built by this module
# as long as beam elements carry the member_id tag and physical_members_v2
# can reconstruct chains.
# --------------------------------------------------------------------------
def prune_members_by_id(model, member_ids, label="force_audit"):
    """Remove the PHYSICAL members named in `member_ids` (a set/iterable of
    member_id values, as returned by physical_members_v2) plus every
    orphaned link element and dead node this leaves behind, mirroring
    `_trim_free_end_members`'s own removal+renumbering machinery exactly
    (same 4 steps: drop the member's beam sub-elements -> drop its loaded
    end(s)' load_idx entries -> iteratively prune any node left with zero
    remaining 'beam' incidence that is not itself a support or still
    loaded (dragging its own dangling 'link' element(s) down with it) ->
    renumber). Unlike `_trim_free_end_members`, a removed member's TWO
    ends are each independently checked against `model.support_idx` /
    `model.load_idx` (an interior fused member has BOTH ends loaded; a
    boundary member has exactly one end loaded and the other pinned or a
    dead free stub) rather than assuming only the chain's last node is
    ever loaded -- so this is safe to call on interior members too, not
    just boundary ones.

    Returns (new_model, meta) where meta reports exactly what was removed
    (mirrors `_trim_free_end_members`'s trim_meta schema, plus the
    `member_ids` requested and `label` for provenance in the audit JSON).
    """
    member_ids = set(member_ids)
    if not member_ids:
        return model, dict(n_removed=0, removed_member_ids=[], n_nodes_removed=0, label=label)

    members = physical_members_v2(model)
    by_id = {m['member_id']: m for m in members}
    missing = member_ids - set(by_id.keys())
    if missing:
        raise ValueError(f"prune_members_by_id({label}): unknown member_id(s) {sorted(missing)}")

    support_set = model.support_idx
    n_nodes_before = len(model.nodes)
    n_elems_before = len(model.elements)

    elems_to_remove = set()
    for ei, e in enumerate(model.elements):
        if e['kind'] == 'beam' and e['member_id'] in member_ids:
            elems_to_remove.add(ei)

    # each removed member's own end(s) that were carrying a nodal load
    # (support ends carry none; a boundary member's outer 'free' dead-stub
    # end, if any, also carries none -- only ends actually present in
    # load_idx get decremented).
    tips_to_unload = []
    for mid in member_ids:
        m = by_id[mid]
        for end in (m['chain'][0], m['chain'][-1]):
            if end not in support_set:
                tips_to_unload.append(end)

    kept_elems = [e for ei, e in enumerate(model.elements) if ei not in elems_to_remove]

    load_count = {}
    for ni in model.load_idx:
        load_count[ni] = load_count.get(ni, 0) + 1
    to_drop = {}
    for ni in tips_to_unload:
        to_drop[ni] = to_drop.get(ni, 0) + 1
    new_load_idx = []
    for ni in model.load_idx:
        if to_drop.get(ni, 0) > 0:
            to_drop[ni] -= 1
            continue
        new_load_idx.append(ni)
    loaded_nodes = set(new_load_idx)

    def beam_degree(elems):
        deg = {}
        for e in elems:
            if e['kind'] != 'beam':
                continue
            deg[e['i']] = deg.get(e['i'], 0) + 1
            deg[e['j']] = deg.get(e['j'], 0) + 1
        return deg

    changed = True
    while changed:
        changed = False
        beam_deg = beam_degree(kept_elems)
        touched_nodes = set()
        for e in kept_elems:
            touched_nodes.add(e['i']); touched_nodes.add(e['j'])
        removable = set(
            ni for ni in touched_nodes
            if ni not in support_set and ni not in loaded_nodes and beam_deg.get(ni, 0) == 0
        )
        if not removable:
            break
        new_kept = []
        for e in kept_elems:
            if e['i'] in removable or e['j'] in removable:
                changed = True
                continue
            new_kept.append(e)
        kept_elems = new_kept

    referenced = set()
    for e in kept_elems:
        referenced.add(e['i']); referenced.add(e['j'])
    keep_old = sorted(set(referenced) | support_set | loaded_nodes)
    remap = {old: new for new, old in enumerate(keep_old) if old < n_nodes_before}

    new_nodes = [model.nodes[old] for old in keep_old if old < n_nodes_before]
    new_elems = []
    for e in kept_elems:
        e2 = dict(e)
        e2['i'] = remap[e['i']]
        e2['j'] = remap[e['j']]
        new_elems.append(e2)
    new_support_idx = set(remap[s] for s in support_set if s in remap)
    new_load_idx_remapped = [remap[ni] for ni in new_load_idx if ni in remap]

    import copy as _copy
    new_model = _copy.copy(model)
    new_model.nodes = new_nodes
    new_model.elements = new_elems
    new_model.support_idx = new_support_idx
    new_model.load_idx = new_load_idx_remapped
    new_model.meta = dict(model.meta)
    new_model.meta.setdefault('force_audit_prune_history', [])
    new_model.meta['force_audit_prune_history'] = list(model.meta.get('force_audit_prune_history', [])) + [
        dict(label=label, removed_member_ids=sorted(member_ids))]

    return new_model, dict(n_removed=len(member_ids), removed_member_ids=sorted(member_ids),
                            label=label,
                            n_nodes_before=n_nodes_before, n_nodes_after=len(new_nodes),
                            n_nodes_removed=n_nodes_before - len(new_nodes),
                            n_elems_before=n_elems_before, n_elems_after=len(new_elems),
                            n_elems_removed=n_elems_before - len(new_elems))


# ==========================================================================
# rf_vault_hex2 -- REDESIGNED hexagonal reciprocal barrel vault
# ==========================================================================
# Motivation (author's own review of the rf_vault_hex / _hex_nexorade_core
# family used above): that pattern clips the true 60-degree axial lattice to
# a Cartesian rectangle (`_axial_rect_patch`) using PER-ROW q_lo/q_hi bounds
# derived from floor/ceil of a FLOAT ratio (length/edge_realized - r/2).
# Two consequences, both confirmed numerically while building this redesign
# (see src/_dev_hex2_explore.py, kept in this repo as the exploration
# scratch-file this section's construction was derived from):
#   1. Floating-point rounding of that ratio makes the two springing lines'
#      pin COUNT depend on sub-ULP noise -- e.g. n_rows=8, a length chosen
#      to be an exact analytic multiple of the realised edge length, still
#      gave pins_r0=9 vs pins_rn=8 (an odd/even mismatch that is NOT a
#      geometric feature of the lattice, just float(length)/float(edge)
#      landing a hair off the intended integer). This is almost certainly
#      the direct cause of the "7 vs 6 pins" asymmetry the author flagged.
#   2. `trim_free_ends` (the existing fix for the jagged x=0/x=length gable
#      boundary) REMOVES one leg of the boundary hub's 3-beam fan outright,
#      which is exactly what "ragged incomplete fan circuits" describes: the
#      fan's own pinwheel rotor is still internally closed (pinwheel_fan()
#      always returns a closed n-beam rotor), but the WOVEN pattern of
#      outgoing physical members around that hub is left with only 2 of its
#      3 legs actually reaching a neighbour/support -- a genuinely "open"
#      circuit in the assembled weave, and the removed leg's own neighbour-
#      side bearing stub is a dead, non-load-transferring end.
#
# A closure algorithm that discards any face whose fan cannot be FULLY
# resolved (every leg fused-interior or pinned-at-a-springing-line) was
# tried first and rejected: on this sheared axial lattice, EVERY finite
# rectangular OR parallelogram sub-patch has a family of boundary edges
# (the ones connecting q,r to q,r+1 -- the lattice's own natural gable-
# direction "seam") that can never satisfy that rule by construction,
# because they are not row-constant and a strict recursive closure of
# "kill any face touching an unresolvable edge" propagates through the
# lattice's fully-connected diagonal seam graph and unravels the ENTIRE
# patch to zero faces regardless of patch size (verified numerically for
# every (n_rows, length) combination tried -- see _dev_hex2_explore.py).
#
# The fix used here instead mirrors how real built reciprocal-frame roofs
# actually close off a free edge (Baverel/Douthe; Song et al. 2013's own
# built examples all terminate an open weave edge against a continuous
# perimeter/fascia chord, not by leaving individual rotor legs dangling):
#   (a) an INTEGER-EXACT patch generator (`_hex2_q_window` /
#       `_hex2_parallelogram_patch`) that derives every row's q-window from
#       pure integer floor-division of the row index, never from
#       floor/ceil of a floating ratio -- this alone makes the two
#       springing lines' pin counts PROVABLY equal (both = n_cols, for any
#       n_rows/n_cols) and the whole patch PROVABLY mirror-symmetric about
#       mid-length under the exact integer map (q, r) -> (n_cols - q - r,
#       r) (verified for every tested (n_rows, n_cols) pair: 0 mismatches).
#   (b) every triangular face in the patch is KEPT (no faces are ever
#       discarded -- "the full fan fits the window" for every face, since
#       every leg now has a well-defined resolution -- see (c)); this also
#       makes the pattern DENSER than the old jagged-clip + trim pipeline
#       at a comparable row count, matching the spec's "denser" ask.
#   (c) each of the two gable ends' now-genuinely-free rotor legs (whose
#       outer tip does NOT reach a springing line) is connected, in
#       transverse (y) order, to its immediate neighbour by a NEW straight
#       "gable rib" physical member (its own dedicated section-group family)
#       -- i.e. a continuous chevron/fascia chord threading through every
#       gable bearing point from (near) one springing line to the other.
#       This chord is a REAL load-transferring member (every gable rotor
#       leg's outer tip now bears on it, and it in turn is embedded in the
#       row-0/row-n_rows corner hubs' own fused/pinned legs), fixing both
#       the "ragged incomplete fan" complaint (every hub's 3-beam fan is now
#       fully engaged: 1-3 legs each individually fused, pinned, or rib-
#       borne, never dangling) and the "edge members read as non-load-
#       transferring" complaint (the old free stub is gone; the rib is a
#       genuine chord in the load path, confirmed by the V4 arch-action
#       check in qa_vault_hex2.py). ZERO mid-air member ends result by
#       construction (verified by `verify_vault_hex2_bearings` below, not
#       just asserted).
#
# Net result: a rectangular-footprint (not parallelogram) vault, both
# springing lines pinned with IDENTICAL pin counts by construction, exact
# mid-length topological mirror symmetry, EVERY hub's fan fully engaged,
# and no post-hoc trimming step needed at all (`trim_free_ends` from
# rf_vault_hex has no analogue here -- there is nothing left to trim).
def _hex2_q_window(r, n_cols):
    """INTEGER-exact per-row (q_lo, q_hi) window (inclusive) for
    `_hex2_parallelogram_patch`'s row r of a patch with n_cols unit cells
    along the springing (r=0) line. Derived from pure integer floor
    division of r (never from floor/ceil of a float ratio -- see this
    section's module-level docstring for why that float dependency is the
    likely root cause of `rf_vault_hex`'s pin-count asymmetry): row r's
    real-space x-range is forced to land EXACTLY on x=0 (for even r) with
    the row's own natural half-cell-per-two-rows shear otherwise, giving
    n_cols+1 nodes on even rows (touching both x=0 and x=n_cols*edge
    exactly) and n_cols nodes on odd rows (whose extremes sit a half-cell
    short of each end -- the genuine, jaggedness of a true 60-degree
    lattice sliced by a straight vertical line, unavoidable and unrelated
    to the float-rounding bug this replaces)."""
    m = r // 2
    if r % 2 == 0:
        return (-m, n_cols - m)
    return (-m, n_cols - m - 1)


def _hex2_parallelogram_patch(n_rows, n_cols, edge):
    """Integer-exact rectangular-footprint (in real x,y -- NOT a sheared
    parallelogram plan despite the function's name, which refers to the
    per-row q-window construction) patch of the true 60-degree axial
    lattice. Returns (coords: dict[(q,r)]->(x,y), faces: list[(a,b,c)]
    (q,r)-triples, deduplicated). See `_hex2_q_window`/module docstring."""
    coords = {}
    for r in range(n_rows + 1):
        qlo, qhi = _hex2_q_window(r, n_cols)
        for q in range(qlo, qhi + 1):
            x = edge * (q + r / 2.0)
            y = edge * r * np.sqrt(3) / 2.0
            coords[(q, r)] = (x, y)
    faces = {}
    for r in range(n_rows):
        qlo0, qhi0 = _hex2_q_window(r, n_cols)
        qlo1, qhi1 = _hex2_q_window(r + 1, n_cols)
        for q in range(min(qlo0, qlo1) - 1, max(qhi0, qhi1) + 1):
            a, b, c = (q, r), (q + 1, r), (q, r + 1)
            if a in coords and b in coords and c in coords:
                faces[frozenset((a, b, c))] = (a, b, c)
            b2, d, c2 = (q + 1, r), (q + 1, r + 1), (q, r + 1)
            if b2 in coords and d in coords and c2 in coords:
                faces[frozenset((b2, d, c2))] = (b2, d, c2)
    return coords, list(faces.values())


def rf_vault_hex2(n_rows=8, n_cols=8, span_transverse=14.0, rise=2.6,
                   lam_bands=(0.15, 0.2, 0.25, 0.3, 0.35, 0.4), ecc_ratio=0.07,
                   load_per_node=-2200.0, chirality=-1, n_section_groups=12):
    """REDESIGNED hexagonal reciprocal barrel vault -- see this module's
    "rf_vault_hex2" section-header docstring immediately above for the full
    rationale/construction. Additive, does not touch `rf_vault_hex`.

    n_cols (new vs rf_vault_hex's `length`): integer number of unit cells
    along the springing (r=0) line; the realised length is n_cols*edge
    (edge itself derived from n_rows/span_transverse exactly as in
    `rf_vault_hex`). n_section_groups is the TOTAL group count: the last
    group index (n_section_groups-1) is reserved for the two gable "rib"
    chords (one shared family, by mirror symmetry); the remaining
    n_section_groups-1 indices are the usual transverse-row-band fan
    groups (`band_of` on face row-average, as in rf_vault_hex)."""
    if n_rows < 2 or n_rows % 2 != 0:
        raise ValueError("rf_vault_hex2 needs an EVEN n_rows>=2 (guarantees identical "
                          "pin counts on both springing lines and exact q-window "
                          "self-consistency under the mid-length mirror map -- see "
                          "module docstring)")
    if n_cols < 2:
        raise ValueError("rf_vault_hex2 needs n_cols >= 2")
    if n_section_groups < 2:
        raise ValueError("rf_vault_hex2 needs n_section_groups >= 2 (>=1 fan band + 1 rib group)")
    n_lambda_bands = len(lam_bands)
    n_fan_groups = n_section_groups - 1
    RIB_GROUP = n_section_groups - 1

    a = span_transverse / 2.0
    Rc = (a ** 2 + rise ** 2) / (2 * rise)

    def z_of(y):
        dy = y - a
        val = max(Rc ** 2 - dy ** 2, 0.0)
        return rise - (Rc - np.sqrt(val))

    e = span_transverse / (n_rows * np.sqrt(3) / 2.0)
    lat_coords, faces = _hex2_parallelogram_patch(n_rows, n_cols, e)
    length_realized = n_cols * e

    def pos(key):
        x, y = lat_coords[key]
        return (x, y, z_of(y))

    def midpoint(p, q):
        pa, pb = np.array(pos(p)), np.array(pos(q))
        return tuple(0.5 * (pa + pb))

    edge_faces = {}
    for fi, (va, vb, vc) in enumerate(faces):
        for (p, q) in ((va, vb), (vb, vc), (vc, va)):
            edge_faces.setdefault(frozenset({p, q}), []).append(fi)

    def band_of(va, vb, vc, n_bands):
        r_avg = (va[1] + vb[1] + vc[1]) / 3.0
        frac = r_avg / n_rows
        return min(int(frac * n_bands), n_bands - 1)

    model = RFModel()
    ecc = ecc_ratio * e
    midg = _MemberIdGen()

    boundary_node = {}

    def boundary_gid(mid_pt):
        key = tuple(np.round(mid_pt, 9))
        if key not in boundary_node:
            boundary_node[key] = model.add_node(mid_pt)
        return boundary_node[key]

    spoke = {}  # frozenset(edge_verts) -> list of dict(lower, tip, group)
    gable_pending = dict(start=[], end=[])  # 'start'=x<mid, 'end'=x>=mid -> [(y, node_gid)]

    for fi, (va, vb, vc) in enumerate(faces):
        dir_edges = ((va, vb), (vb, vc), (vc, va))
        support_pts = [midpoint(p, q) for (p, q) in dir_edges]
        lam_band = band_of(va, vb, vc, n_lambda_bands)
        fan_lam = lam_bands[lam_band]
        fan = pinwheel_fan(support_pts, fan_lam, ecc, chirality=chirality)
        loc2glob = {}

        def gid(li, _map=loc2glob, _fan=fan):
            if li not in _map:
                _map[li] = model.add_node(_fan['nodes'][li])
            return _map[li]

        for (i, j, kind, bid) in fan['elements']:
            if kind == 'link':
                model.elements.append(dict(i=gid(i), j=gid(j), kind='link', group=None))

        group = band_of(va, vb, vc, n_fan_groups)
        for k, (s_local, l_local, u_local) in enumerate(_beam_pairs(fan)):
            lower_g = gid(l_local)
            upper_g = gid(u_local)
            ek = frozenset(dir_edges[k])
            if len(edge_faces[ek]) == 2:
                spoke.setdefault(ek, []).append(dict(lower=lower_g, tip=upper_g, group=group))
                continue
            p, q = dir_edges[k]
            mid_pt = midpoint(p, q)
            bnd_g = boundary_gid(mid_pt)
            mid = midg.new()
            model.elements.append(dict(i=bnd_g, j=lower_g, kind='beam', group=group, member_id=mid))
            model.elements.append(dict(i=lower_g, j=upper_g, kind='beam', group=group, member_id=mid))
            model.load_idx.append(upper_g)
            if p[1] == q[1] and p[1] in (0, n_rows):
                # same-row boundary edge at a springing line -> pin
                # (rf_vault_hex2's ONLY supports).
                model.support_idx.add(bnd_g)
            else:
                # gable-direction boundary edge -> defer to the rib pass
                # below instead of leaving bnd_g a free (mid-air) tip.
                end_key = 'start' if mid_pt[0] < length_realized / 2.0 else 'end'
                gable_pending[end_key].append((mid_pt[1], bnd_g))

    for ek, infos in spoke.items():
        if len(infos) != 2:
            continue
        info_a, info_b = infos
        mid = midg.new()
        model.elements.append(dict(i=info_a['tip'], j=info_a['lower'], kind='beam', group=info_a['group'], member_id=mid))
        model.elements.append(dict(i=info_a['lower'], j=info_b['lower'], kind='beam', group=info_a['group'], member_id=mid))
        model.elements.append(dict(i=info_b['lower'], j=info_b['tip'], kind='beam', group=info_b['group'], member_id=mid))
        model.load_idx.append(info_a['tip'])
        model.load_idx.append(info_b['tip'])

    n_rib_members = 0
    for end_key in ('start', 'end'):
        pts = sorted(gable_pending[end_key], key=lambda t: t[0])
        for (y_a, g_a), (y_b, g_b) in zip(pts[:-1], pts[1:]):
            mid = midg.new()
            model.elements.append(dict(i=g_a, j=g_b, kind='beam', group=RIB_GROUP, member_id=mid))
            n_rib_members += 1

    model.n_groups = n_section_groups
    model.meta = dict(kind='rf_vault_hex2', n_rows=n_rows, n_cols=n_cols, edge=e,
                       span_transverse=span_transverse, rise=rise,
                       rise_span_ratio=rise / span_transverse,
                       length_realized=length_realized,
                       n_faces=len(faces), n_lattice_nodes=len(lat_coords),
                       lam_bands=[float(l) for l in lam_bands],
                       n_lambda_bands=n_lambda_bands, n_section_groups=n_section_groups,
                       n_fan_groups=n_fan_groups, rib_group=RIB_GROUP,
                       n_gable_start_nodes=len(gable_pending['start']),
                       n_gable_end_nodes=len(gable_pending['end']),
                       n_rib_members=n_rib_members,
                       n_pins_r0=sum(1 for ni in model.support_idx if model.nodes[ni][1] < span_transverse / 2.0),
                       n_pins_rn=sum(1 for ni in model.support_idx if model.nodes[ni][1] >= span_transverse / 2.0),
                       ecc=ecc, load_per_node=load_per_node,
                       span=span_transverse)
    return model


def verify_vault_hex2_bearings(model, label=""):
    """Gate-3 analogue of `verify_vault_bearings` for `rf_vault_hex2`: FOUR
    physical-member classes now (vs rf_vault_hex's three -- ribs are a new
    class, and the old "free_end" class is replaced by "gable_boundary",
    whose outer end now bears on a rib instead of dangling):
      - interior fused member: chain length 4, not touching a support.
      - pinned boundary member (springing line): chain length 3, pinned.
      - gable boundary member: chain length 3, NOT pinned (its outer node
        is on the gable, not a springing line) -- its outer tip MUST now
        have >=1 rib segment attached (checked explicitly below), unlike
        rf_vault_hex's structurally-equivalent-looking "free_end" class
        whose outer tip was a genuine dead stub.
      - gable RIB member: chain length 2 (a single new i-j segment directly
        joining two pre-existing gable bearing nodes), never pinned.
    A node is "mid-air" iff it is NOT a support AND touches NO 'link'
    element (i.e. it is a bare gable/boundary meeting point, not a fan's
    own engagement node -- every engagement node always has exactly one
    'link' element by pinwheel_fan() construction, independent of anything
    checked here) AND its own 'beam'-kind degree is < 2 (meaning it has
    only its own single boundary-member segment and nothing else bears on
    it). ok=True requires zero such nodes AND zero mis-classified chains."""
    members = physical_members_v2(model)
    n_interior = n_pinned = n_gable = n_rib = 0
    bad = []
    support_set = model.support_idx
    beam_deg = {}
    link_touch = set()
    for e in model.elements:
        if e['kind'] == 'link':
            link_touch.add(e['i']); link_touch.add(e['j'])
            continue
        beam_deg[e['i']] = beam_deg.get(e['i'], 0) + 1
        beam_deg[e['j']] = beam_deg.get(e['j'], 0) + 1
    for m in members:
        L = len(m['chain'])
        end_a, end_b = m['chain'][0], m['chain'][-1]
        pinned = (end_a in support_set) or (end_b in support_set)
        if L == 4 and not pinned:
            n_interior += 1
        elif L == 3 and pinned:
            n_pinned += 1
        elif L == 3 and not pinned:
            n_gable += 1
        elif L == 2 and not pinned:
            n_rib += 1
        else:
            bad.append((m['member_id'], L, pinned))
    mid_air = []
    all_nodes_touched = set()
    for e in model.elements:
        all_nodes_touched.add(e['i']); all_nodes_touched.add(e['j'])
    for ni in all_nodes_touched:
        if ni in support_set or ni in link_touch:
            continue
        if beam_deg.get(ni, 0) < 2:
            mid_air.append(ni)
    n_links = sum(1 for e in model.elements if e['kind'] == 'link')
    ok = (len(bad) == 0) and (len(mid_air) == 0)
    print(f"[gate3-vault_hex2:{label}] n_members={len(members)} interior={n_interior} "
          f"pinned_boundary={n_pinned} gable_boundary={n_gable} gable_rib={n_rib} "
          f"n_link_elements={n_links} malformed={len(bad)} mid_air_nodes={len(mid_air)} ok={ok}")
    return dict(ok=ok, n_interior=n_interior, n_pinned_boundary=n_pinned,
                n_gable_boundary=n_gable, n_gable_rib=n_rib, n_links=n_links,
                malformed=bad, mid_air_nodes=mid_air)


# --------------------------------------------------------------------------
# 8. rf_dome2 -- REDESIGNED reciprocal dome with a TRUE CIRCULAR support ring
#
# Why a redesign (see results/dome_diagnosis.json for the measured audit
# of the superseded rf_dome_complex):
#
#   * rf_dome_complex tiles a HEX-BOUNDARY TRIANGULAR lattice, so its
#     boundary is a hexagonal polygon and the pinned nodes (boundary edge
#     MIDPOINTS) fall at 14 DISTINCT cylindrical radii spanning
#     6.178 .. 9.523 m (mean 8.053 m, std 1.036 m, i.e. 12.9% CoV).  That is
#     a ragged lattice boundary, not a support RING: a real RF dome springs
#     from a circular ring beam / circular foundation.
#   * The audit found NO under-utilised members (min per-member strain-energy
#     share 26.5% of the mean under service load, 27.4% under self-weight,
#     0 members below the 5% floor), so the redesign is NOT about removing
#     dead members -- it is about (i) the support ring, (ii) density.
#
# rf_dome2 therefore replaces the hex-boundary triangular lattice with a
# POLAR (ring x sector) triangular lattice:
#
#     apex node at r = 0; rings i = 1..n_ring at r = R*i/n_ring; each ring
#     carrying n_sect nodes at theta_j = 2*pi*j/n_sect.  The apex ring is
#     triangulated with n_sect apex triangles; every annulus between rings
#     i and i+1 is split into n_sect quads, each cut into 2 triangles.
#
# Consequences (all exact, all verified by qa_dome2.py):
#   * The ONLY boundary edges are the n_sect circumferential edges of the
#     outermost ring, so there are exactly n_sect supports, one per sector,
#     and the assembly has EXACT C_{n_sect} rotational symmetry (the same
#     construction is emitted sector by sector, including fan chirality).
#   * Each boundary support point is the midpoint of an outer-ring chord and
#     is SNAPPED radially onto the circle r = span/2 (the nominal springing
#     circle).  Because the spherical cap has z = 0 at r = span/2, all pinned
#     nodes land at exactly r = span/2 and exactly z = 0: radius std = 0 and
#     z std = 0 to machine precision, i.e. a TRUE circular, level springing
#     ring.  (The snap moves the pin outward by R*(1 - cos(pi/n_sect)) along
#     its own radius only; the fan itself is still constructed from the true
#     edge midpoints, so fan geometry/regularity is untouched.)
#   * Fans are COMPLETE everywhere: every triangular face carries one full
#     closed 3-beam pinwheel_fan rotor, every interior edge is ONE fused
#     physical member shared by its two fans, and every boundary edge is a
#     member whose outer end is PINNED.  No partial/truncated fan circuits.
#   * Member/DOF algebra of this fan-per-face construction is exactly
#         E  = n_sect * (3*n_ring - 1)          (physical members)
#         F  = n_sect * (2*n_ring - 1)          (fans / faces)
#         V  = 4*E - B, B = n_sect              (FE nodes; 6 per fan + pins)
#     so members and node count are rigidly proportional (V ~ 4E): the task's
#     "180-280 physical members" and "restrained DOF <= ~3600" targets cannot
#     BOTH be met by any member of this construction class (180 members
#     already implies ~4.2k restrained DOF).  Density was prioritised, since
#     the banded fast path (fastsolve.py) now solves this size in ~0.2 s;
#     the resulting DOF count is reported not hidden.
#
# Defaults: n_sect=12, n_ring=6 -> 204 physical members, 132 fans, 12 pins,
# 12-fold rotational symmetry, 15 section groups (5 retained ring families x
# 3 fan orientations), span 18.5 m / rise 3.15 m -> rise/span = 0.170.
# --------------------------------------------------------------------------
def _polar_triangular_patch(n_sect, n_ring, R):
    """KIEWITT-style geodesic triangulation of a disc of radius R.

    Ring i (i = 1..n_ring) carries n_sect*i nodes at radius R*i/n_ring, so
    the circumferential node pitch stays roughly CONSTANT with radius and
    every face is close to equilateral.  (A naive polar grid with a fixed
    n_sect per ring instead produces increasingly obtuse sliver triangles
    towards the boundary; their three edge-midpoints become near-collinear,
    the face-hub rotor built on them becomes near-degenerate, and the
    assembled model turns into a near-mechanism -- measured d_max 3.34 m
    and compliance 3.2e6 versus 0.011 m / 5.5e3 for the reference dome.
    This is why the Kiewitt subdivision is used.)

    Returns (coords, faces, fmeta); fmeta[k] = (family_key, sector_s) marks
    face k as the sector-s copy of the family_key PROTOTYPE, so rf_dome2
    can build each family's rotor once and place the other n_sect-1 copies
    by exact rigid rotation (exact C_{n_sect} symmetry)."""
    coords = [(0.0, 0.0)]
    offs = [0]                      # offs[i] = index of first node of ring i
    for i in range(1, n_ring + 1):
        offs.append(len(coords))
        r = R * i / float(n_ring)
        for k in range(n_sect * i):
            th = 2.0 * np.pi * k / (n_sect * i)
            coords.append((r * np.cos(th), r * np.sin(th)))

    def nid(i, k):
        if i == 0:
            return 0
        return offs[i] + (k % (n_sect * i))

    def orient(tri):
        (ax, ay), (bx, by), (cx, cy) = (coords[t] for t in tri)
        area2 = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        return tri if area2 >= 0 else (tri[0], tri[2], tri[1])

    faces, fmeta = [], []
    for i in range(0, n_ring):                  # annulus between ring i, i+1
        n_in, n_out = i, i + 1                  # edges per base sector
        for s_ in range(n_sect):
            a = [nid(i, s_ * n_in + t) for t in range(n_in + 1)]
            b = [nid(i + 1, s_ * n_out + t) for t in range(n_out + 1)]
            pp, qq, t_idx = 0, 0, 0
            while pp < n_in or qq < n_out:
                take_inner = (qq >= n_out) or (
                    pp < n_in and (pp + 1) / float(n_in) <= (qq + 1) / float(n_out))
                if take_inner:
                    tri = (a[pp], a[pp + 1], b[qq])
                    pp += 1
                else:
                    tri = (a[pp], b[qq + 1], b[qq])
                    qq += 1
                faces.append(orient(tri))
                fmeta.append(((i, t_idx), s_))
                t_idx += 1
    return coords, faces, fmeta


def _face_ring_polar(fam):
    """Ring family of a face = the annulus index carried in its family key
    (0 = apex ring of triangles, i = annulus between rings i and i+1)."""
    return int(fam[0])


def rf_dome2(n_sect=12, n_ring=6, span=18.5, rise=3.15, lam=0.25, ecc_ratio=0.07,
             load_per_node=-3500.0, chirality=-1, lam_per_ring=None,
             max_group_rings=5):
    """Redesigned reciprocal dome: polar-lattice face-hub nexorade on a
    spherical cap, with a TRUE CIRCULAR, level springing ring of n_sect
    pinned supports.  See the section header above for the full rationale.

    lam_per_ring: optional list of length n_ring, one engagement ratio per
    concentric ring family (default None applies the scalar `lam`
    uniformly) -- the black-box geometry variable of the CS4 campaign.
    max_group_rings: ring families beyond max_group_rings-1 are folded into
    the outermost retained family, so n_groups = 3*max_group_rings."""
    a = span / 2.0
    Rs = (a ** 2 + rise ** 2) / (2 * rise)

    def lift(xy):
        x, y = float(xy[0]), float(xy[1])
        r_xy = min(np.hypot(x, y), a)
        z = rise - (Rs - np.sqrt(max(Rs ** 2 - r_xy ** 2, 0.0)))
        return (x, y, z)

    coords, faces, fmeta = _polar_triangular_patch(n_sect, n_ring, a)
    edge_faces = {}
    for fi, (p, q, s) in enumerate(faces):
        for (u, v) in ((p, q), (q, s), (s, p)):
            edge_faces.setdefault(frozenset({u, v}), []).append(fi)

    def midpoint(u, v):
        pu, pv = np.array(lift(coords[u])), np.array(lift(coords[v]))
        return tuple(0.5 * (pu + pv))

    def snapped_pin(u, v):
        """Boundary support point: the planar chord midpoint pushed radially
        out onto the springing circle r = a, then lifted (z = 0 there)."""
        mx = 0.5 * (coords[u][0] + coords[v][0])
        my = 0.5 * (coords[u][1] + coords[v][1])
        rr = np.hypot(mx, my)
        sc = a / rr if rr > 0 else 1.0
        return lift((mx * sc, my * sc))

    edge_len = a / float(n_ring)   # radial pitch: representative edge scale of
                                   # the near-equilateral Kiewitt subdivision
    ecc = ecc_ratio * edge_len
    model = RFModel()
    midg = _MemberIdGen()
    boundary_node = {}

    def boundary_gid(pt):
        key = tuple(np.round(pt, 9))
        if key not in boundary_node:
            boundary_node[key] = model.add_node(pt)
        return boundary_node[key]

    group_ring_cap = n_ring if max_group_rings is None else int(max_group_rings)

    def _rotz(pts, ang):
        c, s_ = np.cos(ang), np.sin(ang)
        P = np.asarray(pts, dtype=float)
        out = np.empty_like(P)
        out[:, 0] = c * P[:, 0] - s_ * P[:, 1]
        out[:, 1] = s_ * P[:, 0] + c * P[:, 1]
        out[:, 2] = P[:, 2]
        return out

    proto_fans = {}
    spoke = {}
    for fi, face in enumerate(faces):
        p, q, s = face
        dir_edges = ((p, q), (q, s), (s, p))
        support_pts = [midpoint(u, v) for (u, v) in dir_edges]
        fam, sect_j = fmeta[fi]
        fring = _face_ring_polar(fam)
        fan_lam = lam_per_ring[min(fring, len(lam_per_ring) - 1)] if lam_per_ring is not None else lam
        # EXACT rotational symmetry: build each family's rotor once (sector 0)
        # and place every other sector's copy by rigid rotation, so the model
        # is invariant under rotation by 2*pi/n_sect to machine precision
        # (pinwheel_fan's internal SVD local-plane basis is NOT guaranteed to
        # be rotation-equivariant, so re-calling it per sector would inject
        # small, spurious symmetry-breaking noise).
        if fam not in proto_fans:
            proto_fans[fam] = (pinwheel_fan(support_pts, fan_lam, ecc, chirality=chirality),
                               sect_j)
        proto, proto_j = proto_fans[fam]
        d_ang = 2.0 * np.pi * (sect_j - proto_j) / n_sect
        fan = dict(proto)
        fan['nodes'] = _rotz(proto['nodes'], d_ang)
        loc2glob = {}

        def gid(li, _map=loc2glob, _fan=fan):
            if li not in _map:
                _map[li] = model.add_node(_fan['nodes'][li])
            return _map[li]

        for (i2, j2, k2, bid) in fan['elements']:
            if k2 == 'link':
                model.elements.append(dict(i=gid(i2), j=gid(j2), kind='link', group=None))

        group_ring = min(fring, group_ring_cap - 1)
        for k, (s_local, l_local, u_local) in enumerate(_beam_pairs(fan)):
            lower_g, upper_g = gid(l_local), gid(u_local)
            group = group_ring * 3 + k
            ek = frozenset(dir_edges[k])
            if len(edge_faces[ek]) == 2:
                spoke.setdefault(ek, []).append(dict(lower=lower_g, tip=upper_g, group=group))
            else:
                bnd_g = boundary_gid(snapped_pin(*dir_edges[k]))
                model.support_idx.add(bnd_g)
                mid = midg.new()
                model.elements.append(dict(i=bnd_g, j=lower_g, kind='beam', group=group, member_id=mid))
                model.elements.append(dict(i=lower_g, j=upper_g, kind='beam', group=group, member_id=mid))
                model.load_idx.append(upper_g)

    for ek, infos in spoke.items():
        if len(infos) != 2:
            continue
        ia, ib = infos
        mid = midg.new()
        model.elements.append(dict(i=ia['tip'], j=ia['lower'], kind='beam', group=ia['group'], member_id=mid))
        model.elements.append(dict(i=ia['lower'], j=ib['lower'], kind='beam', group=ia['group'], member_id=mid))
        model.elements.append(dict(i=ib['lower'], j=ib['tip'], kind='beam', group=ib['group'], member_id=mid))
        model.load_idx.append(ia['tip'])
        model.load_idx.append(ib['tip'])

    model.n_groups = group_ring_cap * 3
    model.meta = dict(kind='rf_dome2', variant='dome2', n_sect=n_sect, n_ring=n_ring,
                       span=span, rise=rise, Rs=Rs, lam=lam, ecc=ecc, edge=edge_len,
                       n_faces=len(faces), max_group_rings=group_ring_cap,
                       rise_span_ratio=rise / span, load_per_node=load_per_node,
                       support_circle_radius=a, rotational_symmetry_order=n_sect,
                       n_ring_families=n_ring, fix_torsion_at_supports=False)
    return model


def verify_dome2_supports(model, label=""):
    """Gate: all pinned nodes lie on ONE circle (radius std ~ 0) at one
    level (z std ~ 0), one per sector, with C_n rotational symmetry."""
    nodes = np.array(model.nodes)
    supp = sorted(model.support_idx)
    P = nodes[supp]
    cx, cy = float(P[:, 0].mean()), float(P[:, 1].mean())
    r = np.hypot(P[:, 0] - cx, P[:, 1] - cy)
    th = np.sort(np.degrees(np.arctan2(P[:, 1] - cy, P[:, 0] - cx)) % 360.0)
    gaps = np.diff(np.concatenate([th, [th[0] + 360.0]]))
    n_sect, n_ring = model.meta['n_sect'], model.meta['n_ring']
    # Kiewitt subdivision: the outermost ring carries n_sect*n_ring nodes, hence
    # n_sect*n_ring boundary (circumferential) edges and one pin per such edge.
    n_expected = n_sect * n_ring
    ok = bool(r.std() < 1e-9 and P[:, 2].std() < 1e-9 and len(supp) == n_expected
              and gaps.std() < 1e-9)
    print(f"[gate-circle:{label}] n_supports={len(supp)} r_mean={r.mean():.6f} "
          f"r_std={r.std():.3e} z_std={P[:, 2].std():.3e} "
          f"gap_mean={gaps.mean():.4f}deg gap_std={gaps.std():.3e} ok={ok}")
    return dict(ok=ok, n_supports=len(supp), radius_mean_m=float(r.mean()),
                radius_std_m=float(r.std()), radius_min_m=float(r.min()),
                radius_max_m=float(r.max()), z_mean_m=float(P[:, 2].mean()),
                z_std_m=float(P[:, 2].std()), centroid_xy=[cx, cy],
                angular_gap_mean_deg=float(gaps.mean()),
                angular_gap_std_deg=float(gaps.std()),
                n_supports_expected=int(n_expected),
                rotational_symmetry_order=int(n_sect),
                support_ring_angular_symmetry_order=int(len(supp)),
                nominal_support_circle_radius_m=float(model.meta['support_circle_radius']))


__all__ += ["rf_dome2", "verify_dome2_supports"]

import os
import os as _os_bootstrap
ROOT = _os_bootstrap.path.dirname(_os_bootstrap.path.dirname(_os_bootstrap.path.abspath(__file__)))
"""
reciprocal_geom.py -- Reciprocal-frame (nexorade) geometry generators.

Core primitive: `pinwheel_fan()` builds an n-beam mutually-supporting rotor
("fan unit") from n arbitrary support points using the standard tangent-line
("rotation") nexorade construction: each beam's support-to-tip line is
tangent to a common inscribed "engagement circle" of radius r_e = lambda *
R (R = mean distance of support points from the fan centroid, lambda =
engagement-length ratio in (0, 0.5]). All n tangent lines share the same
rotational sense (chirality), producing the characteristic pinwheel /
rotor pattern where consecutive beams cross at "engagement points" X_k.

Physical/eccentric connection modelling: at every engagement point X_k, two
FE nodes are created (one belonging to beam k's tip, one to beam (k+1)'s
interior point) offset by +-eccentricity/2 along the local out-of-plane
normal and connected by a short stiff vertical link element -- this is the
standard master-slave eccentric-connector idealisation of a bearing/bolted
RF joint. Consequently each physical beam is represented as a chain of 2
elements (support -> engaged-on-neighbour node -> own-tip node), and the
next beam is naturally split into sub-elements at the point where the
current beam rests on it, matching the "shared node" splitting requested
in the spec.

Four generators are provided, all built from the same primitive:
  - fan_unit(n_beams, R, lam, ...)              : single rotor, pinned ends
  - square_grillage(nx, ny, cell, lam, ...)      : grid of 4-beam fans
                                                    (rotated-square / Ito
                                                    inspired), sharing grid
                                                    corner nodes between
                                                    adjacent cells
  - hex_grillage(rings, edge, lam, ...)          : triangular tiling of
                                                    3-beam fans sharing
                                                    corner nodes
  - dome(rings, span, rise, lam_per_ring, ...)   : hex_grillage triangulation
                                                    lifted onto a spherical
                                                    cap (paraboloid-exact
                                                    sphere mapping), beams
                                                    naturally tilted; each
                                                    ring "family" may take
                                                    its own engagement ratio

All generators return an RFModel (simple dict) with nodes (N,3), elements
(list of dicts: i, j, kind in {'beam','link'}, group), support node
indices, load node indices, and bookkeeping metadata (t_realized engagement
fractions etc. for sanity reporting). A separate `build_frame()` turns an
RFModel + a group->section map into a ready-to-solve Frame3D.

Note: this is a *reciprocal-inspired* parametric family calibrated
to satisfy the standard nexorade defining property (n mutually-supporting
beams, none passing through a common point, engagement ratio lambda
controlling overlap length) and to produce FE-checkable, non-mechanism
structures under gravity loading. It is not digitised from a specific
built precedent; grillage/hex/dome variants connect fans by sharing
support *nodes* at the underlying grid rather than literally sharing full
beam members across every cell boundary (documented explicitly here and in
results/summary.md as a modelling simplification).
"""
import numpy as np
from frame3d import Frame3D
from sections import E_STEEL, G_STEEL, RHO_STEEL, CHS_CATALOG, N_SECTIONS

LINK_SECTION = dict(A=3e-3, Iy=3e-6, Iz=3e-6, J=6e-6, mass_per_len=RHO_STEEL * 3e-3)


# --------------------------------------------------------------------------
# Core primitive
# --------------------------------------------------------------------------
def _local_basis(support_pts):
    """Best-fit-plane orthonormal basis (e1, e2, e3=normal) via SVD, and
    centroid. e3 is oriented to have a non-negative global-Z component
    where possible (falls back to +X) so the eccentricity offset is a
    consistent 'up'."""
    pts = np.asarray(support_pts, dtype=float)
    c = pts.mean(axis=0)
    q = pts - c
    if pts.shape[0] >= 3:
        _, _, Vt = np.linalg.svd(q, full_matrices=False)
        e1, e2, e3 = Vt[0], Vt[1], Vt[2]
    else:
        e1 = q[1] - q[0]
        e1 = e1 / np.linalg.norm(e1)
        tmp = np.array([0, 0, 1.0]) if abs(e1[2]) < 0.9 else np.array([1.0, 0, 0])
        e3 = np.cross(e1, tmp)
        e3 = e3 / np.linalg.norm(e3)
        e2 = np.cross(e3, e1)
    if e3[2] < 0:
        e3 = -e3
        e2 = -e2
    # ensure right-handed
    if np.dot(np.cross(e1, e2), e3) < 0:
        e2 = -e2
    return c, e1, e2, e3


def pinwheel_fan(support_pts, lam, ecc, chirality=-1):
    """Build one n-beam mutually-supporting rotor from n support points.

    Returns a dict with LOCAL 0-based node list/elements:
      nodes: (M,3) array
      elements: list of (i, j, kind, beam_id)   kind in {'beam','link'}
      support_idx: list of n indices (S_k nodes)
      load_idx: list of n indices (X_k 'upper' engagement nodes)
      t_realized: list of n realised engagement fractions (for reporting)
      R_mean: mean support radius (characteristic fan size)
    """
    n = len(support_pts)
    assert n >= 3
    c, e1, e2, e3 = _local_basis(support_pts)
    S3 = np.asarray(support_pts, dtype=float)
    p2 = np.array([[(S3[k] - c) @ e1, (S3[k] - c) @ e2] for k in range(n)])

    R = np.linalg.norm(p2, axis=1)
    R_mean = R.mean()
    r_e = lam * R_mean
    theta = np.arctan2(p2[:, 1], p2[:, 0])
    alpha = np.arccos(np.clip(r_e / R, -1, 1))
    tpt_theta = theta - chirality * alpha
    T = r_e * np.stack([np.cos(tpt_theta), np.sin(tpt_theta)], axis=1)
    D = T - p2
    D = D / np.linalg.norm(D, axis=1, keepdims=True)

    def line_intersect(p_a, d_a, p_b, d_b):
        A = np.array([[d_a[0], -d_b[0]], [d_a[1], -d_b[1]]])
        rhs = p_b - p_a
        s, _ = np.linalg.solve(A, rhs)
        return p_a + s * d_a

    X = np.zeros((n, 2))
    for k in range(n):
        kp1 = (k + 1) % n
        X[k] = line_intersect(p2[k], D[k], p2[kp1], D[kp1])

    t_realized = []
    for k in range(n):
        kp1 = (k + 1) % n
        len_beam_kp1 = np.linalg.norm(X[kp1] - p2[kp1])
        len_to_conn = np.linalg.norm(X[k] - p2[kp1])
        t_realized.append(float(len_to_conn / max(len_beam_kp1, 1e-12)))

    def to3(pt2):
        return c + pt2[0] * e1 + pt2[1] * e2

    nodes = []
    elements = []
    support_idx = []
    load_idx = []

    # support nodes
    for k in range(n):
        nodes.append(to3(p2[k]))
        support_idx.append(len(nodes) - 1)

    # engagement node pairs (upper=tip of beam k, lower=interior of beam k+1)
    upper_idx = [None] * n
    lower_idx = [None] * n
    for k in range(n):
        base = to3(X[k])
        nodes.append(base + (ecc / 2.0) * e3)
        upper_idx[k] = len(nodes) - 1
        nodes.append(base - (ecc / 2.0) * e3)
        lower_idx[k] = len(nodes) - 1
        elements.append((upper_idx[k], lower_idx[k], 'link', None))

    # beam elements: beam k = S_k -> lower[k-1] -> upper[k]
    for k in range(n):
        km1 = (k - 1) % n
        elements.append((support_idx[k], lower_idx[km1], 'beam', k))
        elements.append((lower_idx[km1], upper_idx[k], 'beam', k))
        load_idx.append(upper_idx[k])

    return dict(nodes=nodes, elements=elements, support_idx=support_idx,
                load_idx=load_idx, t_realized=t_realized, R_mean=R_mean,
                n=n)


# --------------------------------------------------------------------------
# RFModel assembly helpers
# --------------------------------------------------------------------------
class RFModel:
    def __init__(self):
        self.nodes = []       # list of (x,y,z)
        self.elements = []    # list of dict(i,j,kind,group)
        self.support_idx = set()
        self.load_idx = []
        self.meta = {}
        self.n_groups = 0

    def add_node(self, xyz):
        self.nodes.append(tuple(xyz))
        return len(self.nodes) - 1

    def merge_fan(self, fan, node_id_map, group_fn):
        """Merge a local pinwheel_fan() dict into this model.
        node_id_map: dict local_idx -> global_idx for the n support nodes
        (already registered); other (engagement/link) nodes are new.
        group_fn(local_beam_id) -> group id (int) for beam elements.
        """
        n = fan['n']
        local_to_global = {}
        for k in range(n):
            local_to_global[fan['support_idx'][k]] = node_id_map[k]
        # add remaining (engagement) nodes as new global nodes
        for li in range(len(fan['nodes'])):
            if li in local_to_global:
                continue
            gi = self.add_node(fan['nodes'][li])
            local_to_global[li] = gi
        for (i, j, kind, beam_id) in fan['elements']:
            gi, gj = local_to_global[i], local_to_global[j]
            grp = group_fn(beam_id) if kind == 'beam' else None
            self.elements.append(dict(i=gi, j=gj, kind=kind, group=grp))
        for li in fan['load_idx']:
            self.load_idx.append(local_to_global[li])
        return local_to_global


def build_frame(model: RFModel, group_section, link_section=None,
                 load_per_node=-1000.0, E=E_STEEL, G=G_STEEL,
                 fix_torsion_at_supports=True):
    """Build a solvable Frame3D from an RFModel + a group->CHS-props map.

    group_section: dict group_id -> section-props dict (A, Iy, Iz, J)
    """
    if link_section is None:
        link_section = LINK_SECTION
    frm = Frame3D()
    for xyz in model.nodes:
        frm.add_node(*xyz)
    for e in model.elements:
        if e['kind'] == 'link':
            sp = link_section
        else:
            sp = group_section[e['group']]
        frm.add_element(e['i'], e['j'], E, G, sp['A'], sp['Iy'], sp['Iz'], sp['J'])
    # Per-model override (ADDITIVE, backward compatible: no pre-existing
    # generator sets this meta key, so every previously-built model keeps the
    # (0,1,2,3) default exactly).  rf_dome2 sets it False because fixing the
    # GLOBAL-x rotation dof (dof 3) at every support is not invariant under
    # rotation about the dome axis, and so would artificially break the
    # C_n symmetry of an otherwise exactly axisymmetric model; a genuine PIN
    # (three translational restraints) is both the correct idealisation of a
    # springing ring bearing on a ring beam and the only rotation-equivariant
    # support condition.
    _meta = getattr(model, 'meta', None) or {}
    fix_t = _meta.get('fix_torsion_at_supports', fix_torsion_at_supports)
    for ni in model.support_idx:
        dofs = (0, 1, 2, 3) if fix_t else (0, 1, 2)
        frm.fix(ni, dofs=dofs)
    for ni in model.load_idx:
        frm.add_load(ni, fz=load_per_node)
    return frm


def model_mass(model: RFModel, group_section, link_section=None):
    if link_section is None:
        link_section = LINK_SECTION
    nodes = np.array(model.nodes)
    total = 0.0
    for e in model.elements:
        L = np.linalg.norm(nodes[e['i']] - nodes[e['j']])
        if e['kind'] == 'link':
            m = link_section['mass_per_len']
        else:
            m = group_section[e['group']]['mass_per_len']
        total += m * L
    return total


def true_node_positions(model: RFModel):
    """Return an (N,3) array of 'true' (non-eccentric) node positions for
    PLOTTING/visualisation.

    Every engagement point X_k is realised in the FE mesh as a pair of
    nodes offset by +-eccentricity/2 along the local normal (the
    master-slave eccentric-connector idealisation of the bearing joint,
    see module docstring) and joined by a short stiff 'link' element. For
    drawing purposes both nodes of every such pair should collapse back to
    their single shared engagement point (the link's midpoint); every
    other node (supports) is already a true, non-offset point and is
    returned unchanged. This is what lets each physical beam be drawn as
    ONE straight rod with a dot at the true bearing point, matching the
    standard reciprocal-frame drawing convention (e.g. Song et al. 2013;
    Larsen), instead of the kinked FE polyline that appears if the
    eccentric sub-elements are plotted element-by-element.
    """
    nodes = np.array(model.nodes, dtype=float)
    true_pos = nodes.copy()
    for e in model.elements:
        if e['kind'] == 'link':
            mid = 0.5 * (nodes[e['i']] + nodes[e['j']])
            true_pos[e['i']] = mid
            true_pos[e['j']] = mid
    return true_pos


def physical_beams(model: RFModel):
    """Reconstruct each PHYSICAL beam's two true axis endpoints (support
    end + tip end) and section group, for plotting one straight line per
    real beam instead of its FE sub-elements.

    Every physical beam is emitted by `pinwheel_fan()` -- the sole source
    of 'beam'-kind elements in every generator -- as exactly two
    consecutive FE sub-elements chained support -> interior (eccentric)
    node -> tip (eccentric) node (the beam is split where it bears
    on/under its neighbour). `RFModel.merge_fan()` appends each fan's
    element list intact and in original order, so filtering the *global*
    `model.elements` down to kind=='beam' and reading them off two at a
    time exactly reconstructs the original per-fan beam pairing,
    regardless of how many fans/groups were merged into the model.

    Returns a list of dicts with GLOBAL node indices:
      support_idx : the beam's outer (pinned/shared) end -- already a
                    true point (no eccentric offset)
      interior_idx: the eccentric-offset node where this beam bears UNDER
                    its neighbour (an interior point on the beam's own
                    axis, not an endpoint)
      tip_idx     : the eccentric-offset node at the beam's own tip, where
                    the NEXT beam bears on top of it
      group       : section group id for this beam
    Use `true_node_positions(model)` to convert interior_idx/tip_idx (and,
    harmlessly, support_idx) into their true, non-offset 3D coordinates:
    the beam's drawn axis is [true_pos[support_idx], true_pos[tip_idx]].
    """
    beam_elems = [e for e in model.elements if e['kind'] == 'beam']
    if len(beam_elems) % 2 != 0:
        raise ValueError("unexpected odd number of 'beam' elements; each "
                          "physical beam should contribute exactly 2 "
                          "(support->interior, interior->tip)")
    beams = []
    for a, b in zip(beam_elems[0::2], beam_elems[1::2]):
        if a['j'] != b['i']:
            raise ValueError("beam sub-elements are not chained as "
                              "support->interior->tip; physical_beams() "
                              "assumptions no longer match the generator")
        beams.append(dict(support_idx=a['i'], interior_idx=a['j'],
                           tip_idx=b['j'], group=b['group']))
    return beams


# --------------------------------------------------------------------------
# Generators
# --------------------------------------------------------------------------
def fan_unit(n_beams, R, lam, ecc_ratio=0.06, single_group=True):
    """Standalone n-beam fan/rotor. Pinned at outer (support) ends, vertical
    load at each engagement (inner) node."""
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

    model.merge_fan(fan, node_id_map, group_fn)
    model.n_groups = 1 if single_group else n_beams
    model.meta = dict(kind='fan_unit', n_beams=n_beams, R=R, lam=lam,
                       t_realized=fan['t_realized'], ecc=ecc)
    return model


def square_grillage(nx, ny, cell, lam, ecc_ratio=0.06, n_rings=3):
    """nx x ny grid of 4-beam rotated-square fans sharing grid corner
    nodes with neighbouring cells; perimeter corner nodes pinned."""
    model = RFModel()
    grid_node = {}

    def get_corner(ix, iy):
        key = (ix, iy)
        if key not in grid_node:
            gi = model.add_node((ix * cell, iy * cell, 0.0))
            grid_node[key] = gi
        return grid_node[key]

    ecc = ecc_ratio * cell
    ring_max = min(nx, ny) // 2

    for ix in range(nx):
        for iy in range(ny):
            corners_ij = [(ix, iy), (ix + 1, iy), (ix + 1, iy + 1), (ix, iy + 1)]
            gids = [get_corner(*c) for c in corners_ij]
            support_pts = [model.nodes[g] for g in gids]
            fan = pinwheel_fan(support_pts, lam, ecc)
            node_id_map = {k: gids[k] for k in range(4)}
            ring = min(ix, iy, nx - 1 - ix, ny - 1 - iy)
            ring = min(ring, n_rings - 1)

            def group_fn(beam_id, ring=ring):
                return ring * 4 + beam_id

            model.merge_fan(fan, node_id_map, group_fn)

    for ix in range(nx + 1):
        for iy in range(ny + 1):
            if ix in (0, nx) or iy in (0, ny):
                model.support_idx.add(grid_node[(ix, iy)])

    model.n_groups = n_rings * 4
    model.meta = dict(kind='square_grillage', nx=nx, ny=ny, cell=cell, lam=lam,
                       ecc=ecc, n_rings=n_rings)
    return model


def _hex_axial_nodes(rings, edge):
    """Triangular lattice within a hex region of given ring radius, axial
    coords (q,r), s=-q-r, region max(|q|,|r|,|s|)<=rings."""
    coords = {}
    for q in range(-rings, rings + 1):
        for r in range(-rings, rings + 1):
            s = -q - r
            if max(abs(q), abs(r), abs(s)) <= rings:
                x = edge * (q + r / 2.0)
                y = edge * (r * np.sqrt(3) / 2.0)
                coords[(q, r)] = (x, y)
    return coords


def _hex_triangles(coords, rings):
    tris = []
    for (q, r) in list(coords.keys()):
        a, b, c = (q, r), (q + 1, r), (q, r + 1)
        if b in coords and c in coords:
            tris.append((a, b, c))
        d = (q + 1, r + 1)
        if b in coords and c in coords and d in coords:
            tris.append((b, d, c))
    return tris


def hex_grillage(rings, edge, lam, ecc_ratio=0.06):
    """Triangular tiling of 3-beam fans over a hexagonal domain, sharing
    corner nodes between adjacent triangles; boundary-ring corners pinned."""
    coords = _hex_axial_nodes(rings, edge)
    tris = _hex_triangles(coords, rings)

    model = RFModel()
    node_id = {}
    for key, (x, y) in coords.items():
        gi = model.add_node((x, y, 0.0))
        node_id[key] = gi

    ecc = ecc_ratio * edge
    for (a, b, c) in tris:
        gids = [node_id[a], node_id[b], node_id[c]]
        support_pts = [model.nodes[g] for g in gids]
        fan = pinwheel_fan(support_pts, lam, ecc)
        node_id_map = {k: gids[k] for k in range(3)}
        ring = max(max(abs(x[0]), abs(x[1]), abs(x[0] + x[1])) for x in (a, b, c))
        ring = min(int(ring), rings)

        def group_fn(beam_id, ring=ring):
            return ring * 3 + beam_id

        model.merge_fan(fan, node_id_map, group_fn)

    for key, gi in node_id.items():
        q, r = key
        s = -q - r
        if max(abs(q), abs(r), abs(s)) == rings:
            model.support_idx.add(gi)

    model.n_groups = (rings + 1) * 3
    model.meta = dict(kind='hex_grillage', rings=rings, edge=edge, lam=lam, ecc=ecc)
    return model


def dome(rings, span, rise, lam_per_ring, edge=None, ecc_ratio=0.06):
    """Reciprocal dome: hex triangular grid lifted onto a spherical cap of
    given base span and rise. lam_per_ring: list of length (rings+1),
    engagement ratio per ring 'family' (ring 0=centre .. rings=boundary)."""
    if edge is None:
        edge = (span / 2.0) / max(rings, 1)
    coords = _hex_axial_nodes(rings, edge)
    tris = _hex_triangles(coords, rings)

    a = span / 2.0
    f = rise
    Rs = (a**2 + f**2) / (2 * f)

    def lift(x, y):
        r_xy = np.hypot(x, y)
        r_xy = min(r_xy, a)
        z = f - (Rs - np.sqrt(max(Rs**2 - r_xy**2, 0.0)))
        # rescale (x,y) radially so outermost ring lands exactly at r=a
        return x, y, z

    # rescale planar coords so the outer ring sits exactly at radius a
    max_r = max(np.hypot(x, y) for x, y in coords.values())
    scale = a / max_r if max_r > 0 else 1.0

    model = RFModel()
    node_id = {}
    node_ring = {}
    for key, (x, y) in coords.items():
        xs, ys = x * scale, y * scale
        xx, yy, zz = lift(xs, ys)
        gi = model.add_node((xx, yy, zz))
        node_id[key] = gi
        q, r = key
        s = -q - r
        node_ring[key] = max(abs(q), abs(r), abs(s))

    ecc = ecc_ratio * edge
    for (ca, cb, cc) in tris:
        gids = [node_id[ca], node_id[cb], node_id[cc]]
        support_pts = [model.nodes[g] for g in gids]
        ring = max(node_ring[ca], node_ring[cb], node_ring[cc])
        ring = min(ring, len(lam_per_ring) - 1)
        lam = lam_per_ring[ring]
        fan = pinwheel_fan(support_pts, lam, ecc)
        node_id_map = {k: gids[k] for k in range(3)}

        def group_fn(beam_id, ring=ring):
            return ring * 3 + beam_id

        model.merge_fan(fan, node_id_map, group_fn)

    for key, gi in node_id.items():
        if node_ring[key] == rings:
            model.support_idx.add(gi)

    model.n_groups = (rings + 1) * 3
    model.meta = dict(kind='dome', rings=rings, span=span, rise=rise,
                       lam_per_ring=lam_per_ring, edge=edge, ecc=ecc, Rs=Rs)
    return model


# --------------------------------------------------------------------------
# Verification helper
# --------------------------------------------------------------------------
def verify_model(model: RFModel, uniform_section=None, span_estimate=None,
                  load_per_node=-1000.0, label=""):
    """Build FE model with a uniform mid-catalog section, solve, and sanity
    check: finite displacements, span/5000..span/50 magnitude, not singular."""
    if uniform_section is None:
        uniform_section = CHS_CATALOG[4]  # mid-catalog
    gs = {g: uniform_section for g in range(model.n_groups)}
    frm = build_frame(model, gs, load_per_node=load_per_node)
    U, R = frm.solve()
    dmax = frm.max_abs_disp()
    if span_estimate is None:
        nodes = np.array(model.nodes)
        span_estimate = np.linalg.norm(nodes.max(axis=0) - nodes.min(axis=0))
    lo, hi = span_estimate / 5000.0, span_estimate / 50.0
    ok_finite = np.all(np.isfinite(U))
    ok_range = lo <= dmax <= hi
    mass = model_mass(model, gs)
    print(f"[verify:{label}] nodes={len(model.nodes)} elems={len(model.elements)} "
          f"groups={model.n_groups} span~={span_estimate:.3f}m dmax={dmax:.5f}m "
          f"(target {lo:.5f}-{hi:.5f}) mass={mass:.1f}kg finite={ok_finite} "
          f"in_range={ok_range}")
    return dict(dmax=dmax, mass=mass, ok_finite=bool(ok_finite),
                ok_range=bool(ok_range), span_estimate=span_estimate)


if __name__ == "__main__":
    print("=== fan_unit (3-beam) ===")
    m3 = fan_unit(3, R=3.0, lam=0.25)
    verify_model(m3, label="fan3")
    print("t_realized:", m3.meta['t_realized'])

    print("=== fan_unit (4-beam) ===")
    m4 = fan_unit(4, R=3.0, lam=0.25)
    verify_model(m4, label="fan4")
    print("t_realized:", m4.meta['t_realized'])

    print("=== square_grillage 4x4 ===")
    msg = square_grillage(4, 4, cell=3.0, lam=0.25)
    verify_model(msg, label="square_grillage")

    print("=== hex_grillage rings=2 ===")
    mhg = hex_grillage(2, edge=3.0, lam=0.25)
    verify_model(mhg, label="hex_grillage")

    print("=== dome rings=2 ===")
    mdo = dome(2, span=16.0, rise=3.0, lam_per_ring=[0.25, 0.25, 0.25])
    # domes are much stiffer (double curvature / arching action) than flat
    # grillages of similar span, so a proportionally larger nodal load is
    # needed to bring displacements into the same span/5000..span/50
    # sanity window used for the flat typologies.
    verify_model(mdo, label="dome", load_per_node=-4000.0)

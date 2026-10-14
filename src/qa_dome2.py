import os
import os as _os_bootstrap
ROOT = _os_bootstrap.path.dirname(_os_bootstrap.path.dirname(_os_bootstrap.path.abspath(__file__)))
"""qa_dome2.py -- physics verification battery (V1-V6) for the REDESIGNED
reciprocal dome `rf_dome2` (polar-lattice face-hub nexorade on a spherical
cap with a TRUE CIRCULAR, level springing ring).  Same structure as
qa_vault_hex2.py, with the two typology-specific substitutions the task
requires:

  V1 global equilibrium:     |sum(R)+sum(F)| < 1e-6*|sum(F)|, 3 components.
  V2 ROTATIONAL symmetry:    under the AXISYMMETRIC service load, the
                              displacement field must be invariant under the
                              dome's own rotational symmetry order n_sect:
                              u(Q x) = Q u(x) with Q = rotation by
                              2*pi/n_sect.  The node permutation induced by
                              Q is derived independently from the built
                              coordinates (not assumed).  Max deviation is
                              reported as a % of the peak displacement.
  V3 member energy floor:    every PHYSICAL member's strain-energy share of
                              the mean >= 5%; up to 3 (n_sect, n_ring)
                              attempts if violated.
  V4 arch action:            self-weight case: (a) the horizontal reaction
                              at the springing ring must point INWARD
                              (radial component of the support reaction on
                              the structure < 0); (b) meridional members
                              must be net in COMPRESSION.
  V5 standard gates:         gate1 (nonsingular, lam in {0.15,0.25,0.4}),
                              gate2 (zero interior supports), gate3
                              (member bearings), gate4 (deflection window)
                              + closed-3-cycle fan check + TRUE-CIRCLE
                              support-ring check (the dome analogue of the
                              vault's "equal pins on both springing lines").
  V6 QA figures:             qa_dome2_3d.png, qa_dome2_plan.png.

Outputs: results/dome2_verification.json, figures_v2/qa_dome2_*.png
"""
import json
import time
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

from sections import CHS_CATALOG, E_STEEL, G_STEEL
from reciprocal_geom import build_frame, model_mass, LINK_SECTION
from frame3d import Frame3D
from fastsolve import fast_solve, cond_spd_estimate
from reciprocal_geom_v2 import (
    rf_dome2, physical_members_v2, true_node_positions,
    verify_nonsingular, verify_no_interior_supports, verify_member_bearings,
    verify_deflection, verify_dome2_supports,
)

OUT_FIG = os.path.join(ROOT, "figures")
OUT_RES = os.path.join(ROOT, "results")
VERIF_PATH = f"{OUT_RES}/dome2_verification.json"
STATS_PATH = f"{OUT_RES}/dome2_geometry_stats.json"

MID_SECTION = CHS_CATALOG[4]
LAM_LIST = [0.15, 0.25, 0.4]
LAM_REF = 0.25
SPAN, RISE = 18.5, 3.15
MAX_GROUP_RINGS = 5   # ring families beyond this fold into the outermost retained family
G_GRAV = 9.81
V3_ATTEMPTS = [(7, 4), (8, 4), (6, 5)]
CKPT = os.path.join(ROOT, "_scratch_qa", "qa_dome2_ckpt.json")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def make_model(n_sect, n_ring, lam_uniform=LAM_REF):
    return rf_dome2(n_sect=n_sect, n_ring=n_ring, span=SPAN, rise=RISE,
                    lam=lam_uniform, max_group_rings=min(n_ring, MAX_GROUP_RINGS))


def uniform_gs(model, section=MID_SECTION):
    return {g: section for g in range(model.n_groups)}


def axial_sign_convention():
    """Calibrate the local-axial-force sign with BOTH a known-tension and a
    known-compression reference bar.

    IMPORTANT CORRECTION.  The pre-existing scripts in this project
    (r6_uls_v2.py, cs_complex_uls.py, vault_hex_posthoc.py, qa_vault_hex2.py)
    calibrate only the TENSION reference and then classify
        is_compression = (sign(Ni) == sign(axial_sign_of_TENSION_bar)),
    which flags members whose axial force has the SAME sign as a bar known to
    be in TENSION -- i.e. that flag actually marks TENSION, not compression.
    Measured here directly: a bar pulled apart gives Ni = -1000 N, a bar
    pushed together gives Ni = +1000 N.  So Ni < 0 is tension and Ni > 0 is
    compression, and the correct test is sign(Ni) == sign(N_compression_ref).
    This module uses the corrected test and records both reference values."""
    def bar(fx):
        frm = Frame3D()
        frm.add_node(0, 0, 0)
        frm.add_node(1, 0, 0)
        frm.add_element(0, 1, E_STEEL, G_STEEL, 1e-3, 1e-6, 1e-6, 2e-6, ref_vec=[0, 0, 1])
        frm.fix(0, dofs=(0, 1, 2, 3, 4, 5))
        frm.add_load(1, fx=fx)
        frm.solve(check_cond=True)
        return float(frm.element_end_forces_local(0)[0])
    return bar(+1000.0), bar(-1000.0)


def build_selfweight_frame(model, gs):
    frm = build_frame(model, gs, load_per_node=0.0)
    nodes = np.array(model.nodes)
    for e in model.elements:
        L = float(np.linalg.norm(nodes[e['i']] - nodes[e['j']]))
        sp = gs[e['group']] if e['kind'] == 'beam' else LINK_SECTION
        w_each = -sp['mass_per_len'] * L * G_GRAV / 2.0
        frm.add_load(e['i'], fz=w_each)
        frm.add_load(e['j'], fz=w_each)
    return frm


def member_strain_energies(model, frm):
    members = physical_members_v2(model)
    by_id = defaultdict(list)
    for ei, e in enumerate(model.elements):
        if e['kind'] == 'beam':
            by_id[e['member_id']].append(ei)
    se = np.array([sum(frm.element_strain_energy(ei) for ei in by_id[m['member_id']])
                   for m in members])
    return members, se


# ---------------------------------------------------------------- V1
def check_equilibrium(model, frm):
    n = len(model.nodes)
    Rm = frm.R.reshape(n, 6)[:, 0:3]
    Fm = frm.load_vector().reshape(n, 6)[:, 0:3]
    sumR, sumF = Rm.sum(axis=0), Fm.sum(axis=0)
    resid = sumR + sumF
    normF = float(np.linalg.norm(sumF))
    ratio = float(np.linalg.norm(resid) / normF) if normF > 0 else float(np.linalg.norm(resid))
    comp = [float(abs(resid[c]) / normF) if normF > 0 else float(abs(resid[c])) for c in range(3)]
    return dict(sumR=[float(v) for v in sumR], sumF=[float(v) for v in sumF],
                resid=[float(v) for v in resid], normF=normF, ratio_overall=ratio,
                ratio_per_component_xyz=comp, ok=bool(ratio < 1e-6))


# ---------------------------------------------------------------- V2
def check_rotational_symmetry(model, frm):
    """Under the axisymmetric service load the displacement field must
    satisfy u(Q x) = Q u(x) for Q = rotation by 2*pi/n_sect about the dome
    axis.  The permutation sigma with x_sigma(i) = Q x_i is derived here
    directly from the built coordinates (round-to-1e-7 hash lookup); every
    node must have a partner or the model is not C_n symmetric at all."""
    n_sect = model.meta['n_sect']
    ang = 2.0 * np.pi / n_sect
    c, s = np.cos(ang), np.sin(ang)
    Q = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    P = np.array(model.nodes, dtype=float)
    n = len(P)
    U = frm.U.reshape(n, 6)[:, 0:3]

    key = {}
    for i, p in enumerate(P):
        key.setdefault(tuple(np.round(p, 7)), i)
    perm = np.full(n, -1, dtype=int)
    n_unmatched = 0
    for i in range(n):
        j = key.get(tuple(np.round(Q @ P[i], 7)))
        if j is None:
            n_unmatched += 1
        else:
            perm[i] = j
    matched = perm >= 0

    # geometric residual of the permutation (should be ~1e-15)
    geo_res = float(np.abs(P[perm[matched]] - (P[matched] @ Q.T)).max()) if matched.any() else float('nan')

    dev = np.linalg.norm(U[perm[matched]] - (U[matched] @ Q.T), axis=1)
    scale = float(np.linalg.norm(U, axis=1).max())
    max_dev = float(dev.max()) if len(dev) else float('nan')
    worst = int(np.argmax(dev)) if len(dev) else -1
    return dict(rotational_symmetry_order=int(n_sect), rotation_deg=360.0 / n_sect,
                n_nodes=n, n_nodes_matched=int(matched.sum()), n_unmatched=int(n_unmatched),
                geometric_permutation_residual_m=geo_res,
                disp_scale_m=scale, max_deviation_abs_m=max_dev,
                max_deviation_pct=float(100.0 * max_dev / scale) if scale > 0 else float('nan'),
                mean_deviation_pct=float(100.0 * dev.mean() / scale) if scale > 0 else float('nan'),
                worst_node_xyz=[float(v) for v in P[matched][worst]] if worst >= 0 else None,
                ok=bool(n_unmatched == 0 and scale > 0 and 100.0 * max_dev / scale < 1.0),
                note="The service load case is axisymmetric (equal vertical load at every "
                     "load node, and the load-node SET is itself C_n invariant), and the "
                     "geometry is C_n invariant by construction (each fan family is built "
                     "once at sector 0 and placed by exact rigid rotation -- see rf_dome2). "
                     "This is therefore an EXACT symmetry test on the whole mesh: every one "
                     "of the n_nodes has a rotational partner, so the deviation reported is "
                     "pure solver round-off, not a geometric artefact. Contrast the vault's "
                     "V2, where only the lattice-derived subset admits an exact mirror.")


# ---------------------------------------------------------------- V3
def check_member_energy_floor(model, frm):
    members, se = member_strain_energies(model, frm)
    mean_se = float(se.mean())
    shares = 100.0 * se / mean_se
    worst = members[int(np.argmin(shares))]
    return dict(n_members=len(se), mean_se_J=mean_se, min_share_pct=float(shares.min()),
                worst_member_id=int(worst['member_id']), worst_member_group=int(worst['group']),
                n_below_5pct=int((shares < 5.0).sum()),
                percentiles_pct={f'p{p}': float(np.percentile(shares, p)) for p in (5, 25, 50, 75, 95)},
                ok=bool(shares.min() >= 5.0))


# ---------------------------------------------------------------- V4
def check_arch_action(model, gs, compression_sign):
    frm = build_selfweight_frame(model, gs)
    fast_solve(frm)
    nodes = np.array(model.nodes)
    n = len(nodes)
    Rm = frm.R.reshape(n, 6)[:, 0:3]

    supp = sorted(model.support_idx)
    P = nodes[supp]
    rad = P[:, 0:2] / np.linalg.norm(P[:, 0:2], axis=1)[:, None]
    R_radial = np.einsum('ij,ij->i', Rm[supp][:, 0:2], rad)
    inward = bool(R_radial.mean() < 0 and (R_radial < 0).all())

    # meridional members: chord direction predominantly radial in plan
    true_pos = true_node_positions(model)
    members = physical_members_v2(model)
    by_id = defaultdict(list)
    for ei, e in enumerate(model.elements):
        if e['kind'] == 'beam':
            by_id[e['member_id']].append(ei)

    n_merid, n_comp, axials = 0, 0, []
    for m in members:
        pts = true_pos[m['chain']]
        p0, p1 = pts[0], pts[-1]
        mid = 0.5 * (p0 + p1)
        rmid = np.hypot(mid[0], mid[1])
        if rmid < 1e-9:
            continue
        rhat = np.array([mid[0] / rmid, mid[1] / rmid])
        that = np.array([-rhat[1], rhat[0]])
        d = (p1 - p0)[0:2]
        if np.linalg.norm(d) < 1e-12 or abs(d @ rhat) <= abs(d @ that):
            continue
        n_merid += 1
        Ns = [float(frm.element_end_forces_local(ei)[0]) for ei in by_id[m['member_id']]]
        Ni = float(np.mean(Ns))
        axials.append(Ni)
        if Ni != 0 and np.sign(Ni) == np.sign(compression_sign):
            n_comp += 1

    frac = n_comp / n_merid if n_merid else float('nan')
    return dict(n_supports=len(supp),
                mean_radial_reaction_N=float(R_radial.mean()),
                min_radial_reaction_N=float(R_radial.min()),
                max_radial_reaction_N=float(R_radial.max()),
                all_supports_thrust_inward=inward,
                n_meridional_members=n_merid, n_meridional_in_compression=n_comp,
                frac_meridional_in_compression=float(frac),
                mean_meridional_axial_N=float(np.mean(axials)) if axials else None,
                compliance_selfweight=float(frm.compliance()),
                dmax_selfweight_m=float(frm.max_abs_disp()),
                ok=bool(inward and frac >= 0.5),
                note="Arch action for a dome = (a) the springing ring must be pushed INWARD "
                     "by its supports to restrain the outward spread of the cap under "
                     "gravity (radial component of the support reaction ON the structure "
                     "negative at EVERY support), and (b) the meridional (radially-running) "
                     "members must be net in COMPRESSION under the same axial_sign "
                     "convention, CORRECTED here (see axial_sign_convention: a bar in known "
                     "compression gives Ni>0; the pre-existing scripts' is_compression flag "
                     "tested against the TENSION reference and therefore actually flagged "
                     "tension). Both conditions are required for ok; the compression fraction "
                     "is reported exactly as measured.")


# ---------------------------------------------------------------- V5
def check_closed_3cycle(model):
    link_edges = [(e['i'], e['j']) for e in model.elements if e['kind'] == 'link']
    expected = 3 * model.meta['n_faces']
    beam_deg = defaultdict(int)
    for e in model.elements:
        if e['kind'] == 'beam':
            beam_deg[e['i']] += 1
            beam_deg[e['j']] += 1
    touch = defaultdict(int)
    for i, j in link_edges:
        touch[i] += 1
        touch[j] += 1
    bad = sum(1 for (i, j) in link_edges
              if sorted((beam_deg.get(i, 0), beam_deg.get(j, 0))) != [1, 2])
    multi = sum(1 for v in touch.values() if v != 1)
    ok = (len(link_edges) == expected) and bad == 0 and multi == 0
    return dict(n_link_elements=len(link_edges), expected_link_elements=expected,
                n_faces=model.meta['n_faces'], bad_bearing_pairs=bad,
                n_link_nodes_multi_touched=multi, ok=bool(ok),
                note="Every face carries one COMPLETE closed 3-beam pinwheel rotor: the "
                     "alternating beam/link circuit is verified by (1) exactly 3*n_faces "
                     "link elements, (2) every link node touched by exactly one link, and "
                     "(3) every link pair having beam-degree multiset {1,2} (one beam tip "
                     "bearing on one interior point of the next beam).")


# ---------------------------------------------------------------- V6 figures
def classify_member(m, support_set):
    ends = (m['chain'][0], m['chain'][-1])
    pinned = (ends[0] in support_set) or (ends[1] in support_set)
    return 'pinned (springing)' if pinned else 'interior (fused)'


def plot_qa_figures(model, prefix='dome2'):
    true_pos = true_node_positions(model)
    members = physical_members_v2(model)
    support_set = model.support_idx
    S = np.array([model.nodes[i] for i in sorted(support_set)])
    colors = {'interior (fused)': '#1f77b4', 'pinned (springing)': '#2ca02c'}

    for mode in ('3d', 'plan'):
        fig = plt.figure(figsize=(10, 8.2) if mode == '3d' else (9, 9))
        ax = fig.add_subplot(111, projection='3d') if mode == '3d' else fig.add_subplot(111)
        seen = set()
        for m in members:
            cls = classify_member(m, support_set)
            p0, p1 = true_pos[m['chain'][0]], true_pos[m['chain'][-1]]
            lab = cls if cls not in seen else None
            seen.add(cls)
            if mode == '3d':
                ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
                        color=colors[cls], lw=2.4, label=lab, solid_capstyle='round')
            else:
                ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=colors[cls], lw=2.2,
                        label=lab, solid_capstyle='round')

        # circular support ring, drawn as an explicit circle + pin markers
        Rc = model.meta['support_circle_radius']
        th = np.linspace(0, 2 * np.pi, 400)
        if mode == '3d':
            ax.plot(Rc * np.cos(th), Rc * np.sin(th), np.zeros_like(th), color='crimson',
                    lw=2.6, ls='--', label=f'support circle r={Rc:.2f} m')
            ax.scatter(S[:, 0], S[:, 1], S[:, 2], color='crimson', marker='^', s=150,
                       edgecolor='k', linewidth=1.0, depthshade=False,
                       label=f'pinned support (n={len(S)})')
            ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]'); ax.set_zlabel('z [m]')
            ax.view_init(elev=26, azim=-58)
            ax.set_box_aspect((1, 1, 0.42))
        else:
            ax.plot(Rc * np.cos(th), Rc * np.sin(th), color='crimson', lw=2.6, ls='--',
                    label=f'support circle r={Rc:.2f} m')
            ax.scatter(S[:, 0], S[:, 1], color='crimson', marker='^', s=170,
                       edgecolor='k', linewidth=1.0, zorder=5,
                       label=f'pinned support (n={len(S)})')
            ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
            ax.set_aspect('equal')
            ax.grid(alpha=0.25)
        ax.legend(loc='upper right', fontsize=9, framealpha=0.92)
        fig.tight_layout()
        out = f"{OUT_FIG}/qa_{prefix}_{mode}.png"
        fig.savefig(out, dpi=155)
        plt.close(fig)
        log(f"wrote {out}")


# ---------------------------------------------------------------- main
def _load(path, default):
    import os
    if os.path.exists(path):
        try:
            with open(path) as f:
                d = json.load(f)
                if d:
                    return d
        except Exception:
            pass
    return default


def _save(path, obj):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)


STAGES = ['geom', 'v1v2', 'v4', 'gates', 'figs']


def main():
    """Staged + checkpointed so each stage fits inside one short tool call.
    Re-invoke until it prints ALL DONE."""
    t0 = time.time()
    out = _load(VERIF_PATH, {})
    done = set(out.get('_stages_done', []))

    tension_ref, compression_ref = axial_sign_convention()
    nsel = out.get('construction_params', {}).get('n_sect', V3_ATTEMPTS[0][0])
    rsel = out.get('construction_params', {}).get('n_ring', V3_ATTEMPTS[0][1])

    def build():
        m = make_model(nsel, rsel)
        g = uniform_gs(m)
        f = build_frame(m, g, load_per_node=m.meta['load_per_node'])
        fast_solve(f)
        return m, g, f

    if 'geom' not in done:
        attempts, sel = [], None
        for (n_sect, n_ring) in V3_ATTEMPTS:
            log(f"V3 attempt n_sect={n_sect} n_ring={n_ring}...")
            m = make_model(n_sect, n_ring)
            g = uniform_gs(m)
            f = build_frame(m, g, load_per_node=m.meta['load_per_node'])
            fast_solve(f)
            r = check_member_energy_floor(m, f)
            r.update(n_sect=n_sect, n_ring=n_ring,
                     n_physical_members=len(physical_members_v2(m)),
                     ndof_restrained=6 * len(m.nodes) - len(f.fixed))
            attempts.append(r)
            log(f"  -> members={r['n_physical_members']} min_share={r['min_share_pct']:.3f}% "
                f"n<5%={r['n_below_5pct']} ok={r['ok']}")
            if sel is None or r['ok']:
                sel = r
            if r['ok']:
                break
        nsel, rsel = sel['n_sect'], sel['n_ring']
        model, gs, frm = build()
        t_f0 = time.time()
        f_t = build_frame(model, gs, load_per_node=model.meta['load_per_node'])
        fast_solve(f_t)
        t_fast = time.time() - t_f0
        n_nodes = len(model.nodes)
        out.update(
            typology='rf_dome2', supersedes='rf_dome_complex',
            construction_params=dict(n_sect=nsel, n_ring=rsel, span=SPAN, rise=RISE,
                                     max_group_rings=min(rsel, MAX_GROUP_RINGS),
                                     lam_ref=LAM_REF, mid_section_D_mm=MID_SECTION['D_mm']),
            counts=dict(n_sect=nsel, n_ring=rsel, span=SPAN, rise=RISE,
                        n_nodes=n_nodes, n_fe_elements=len(model.elements),
                        n_beam_elements=sum(1 for e in model.elements if e['kind'] == 'beam'),
                        n_link_elements=sum(1 for e in model.elements if e['kind'] == 'link'),
                        n_physical_members=len(physical_members_v2(model)),
                        n_faces=model.meta['n_faces'], n_supports=len(model.support_idx),
                        n_groups=model.n_groups, n_ring_families=model.meta['n_ring_families'],
                        rotational_symmetry_order=model.meta['rotational_symmetry_order'],
                        support_circle_radius_m=model.meta['support_circle_radius'],
                        ndof_total=6 * n_nodes, n_fixed_dof=len(frm.fixed),
                        ndof_restrained=6 * n_nodes - len(frm.fixed),
                        support_condition='pinned (3 translational restraints per support; '
                                          'see reciprocal_geom.build_frame per-model override)',
                        mass_kg_mid_section=float(model_mass(model, gs)),
                        rise_span_ratio=RISE / SPAN,
                        fast_path_solve_time_s=float(t_fast),
                        compliance_service=float(frm.compliance()),
                        dmax_service_m=float(frm.max_abs_disp())),
            axial_sign_convention=dict(
                Ni_known_tension_bar=tension_ref, Ni_known_compression_bar=compression_ref,
                rule="Ni < 0 => TENSION, Ni > 0 => COMPRESSION",
                note="Corrects the inverted is_compression test used by the pre-existing "
                     "post-hoc scripts (which compared against the TENSION reference)."),
            V3_member_energy_floor=sel, V3_attempts=attempts,
            V3_ok_reached_first_try=bool(attempts[0]['ok']))
        done.add('geom')
        out['_stages_done'] = sorted(done)
        _save(VERIF_PATH, out)
        log("stage geom done")

    if 'v1v2' not in done:
        model, gs, frm = build()
        out['V1_equilibrium'] = check_equilibrium(model, frm)
        out['V2_rotational_symmetry'] = check_rotational_symmetry(model, frm)
        log(f"  V1 ratio={out['V1_equilibrium']['ratio_overall']:.3e} "
            f"V2 max_dev={out['V2_rotational_symmetry']['max_deviation_pct']:.3e}%")
        done.add('v1v2')
        out['_stages_done'] = sorted(done)
        _save(VERIF_PATH, out)
        log("stage v1v2 done")

    if 'v4' not in done:
        model, gs, _ = build()
        out['V4_arch_action'] = check_arch_action(model, gs, compression_ref)
        v4 = out['V4_arch_action']
        log(f"  V4 mean_radial_R={v4['mean_radial_reaction_N']:.1f}N inward={v4['all_supports_thrust_inward']} "
            f"merid_compr={v4['n_meridional_in_compression']}/{v4['n_meridional_members']} ok={v4['ok']}")
        done.add('v4')
        out['_stages_done'] = sorted(done)
        _save(VERIF_PATH, out)
        log("stage v4 done")

    if 'gates' not in done:
        model, gs, _ = build()
        gates = out.get('V5_gates', {})
        if 'gate1_nonsingular' not in gates:
            # Same gate semantics as reciprocal_geom_v2.verify_nonsingular (restrained
            # K must be non-singular at every lam), but the 2-norm condition number is
            # obtained from the SPD power/inverse-iteration estimator in fastsolve
            # rather than np.linalg.cond's full SVD, which does not fit the per-call
            # wall-clock budget at ~4.1k DOF.  Agreement of the two on the reference
            # dome is recorded in banded_fastpath_validation.json.
            g1 = {}
            for lam in LAM_LIST:
                mm = make_model(nsel, rsel, lam)
                gg = uniform_gs(mm)
                ff = build_frame(mm, gg, load_per_node=mm.meta['load_per_node'])
                est = cond_spd_estimate(ff)
                g1[str(lam)] = dict(ok=bool(not est['singular']), cond=est['cond'],
                                    lambda_min=est['lambda_min'], lambda_max=est['lambda_max'],
                                    cond_method='SPD power/inverse iteration (lambda_max/lambda_min)',
                                    n_nodes=len(mm.nodes), n_elems=len(mm.elements))
                log(f"[gate1:rf_dome2] lam={lam}: cond={est['cond']:.4e} ok={not est['singular']}")
            gates['gate1_nonsingular'] = g1
            out['V5_gates'] = gates
            _save(VERIF_PATH, out)
        gates.pop('circular_support_ring', None)
        gates['gate2_zero_interior_supports'] = verify_no_interior_supports(model, label='rf_dome2')
        gates['gate3_bearings'] = verify_member_bearings(model, label='rf_dome2')
        gates['gate4_deflection'] = verify_deflection(model, uniform_section=MID_SECTION,
                                                      span=SPAN, label='rf_dome2')
        gates['closed_3cycle_fan_check'] = check_closed_3cycle(model)
        gates['circular_support_ring'] = verify_dome2_supports(model, label='rf_dome2')
        gates['all_gates_ok'] = bool(
            all(v['ok'] for v in gates['gate1_nonsingular'].values()) and
            gates['gate2_zero_interior_supports']['ok'] and gates['gate3_bearings']['ok'] and
            gates['gate4_deflection']['ok'] and gates['closed_3cycle_fan_check']['ok'] and
            gates['circular_support_ring']['ok'])
        out['V5_gates'] = gates
        done.add('gates')
        out['_stages_done'] = sorted(done)
        _save(VERIF_PATH, out)
        log("stage gates done")

    if 'figs' not in done:
        model, gs, _ = build()
        plot_qa_figures(model)
        out['V6_figures'] = ['qa_dome2_3d.png', 'qa_dome2_plan.png']
        _save(STATS_PATH, dict(rf_dome2=dict(
            counts=out['counts'], gates=out.get('V5_gates', {}),
            construction_params=out['construction_params'],
            v1_v6_verification_battery="see results/dome2_verification.json")))
        log(f"WROTE {STATS_PATH}")
        done.add('figs')
        out['_stages_done'] = sorted(done)
        _save(VERIF_PATH, out)
        log("stage figs done")

    out['elapsed_last_call_s'] = time.time() - t0
    _save(VERIF_PATH, out)
    log(f"WROTE {VERIF_PATH}")
    if all(st in done for st in STAGES):
        print("ALL DONE")


if __name__ == '__main__':
    main()

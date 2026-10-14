"""r6_cs3_posthoc_update.py -- PHANTOM-GROUP FIX follow-up: refresh ONLY the
CS3 (hex_nexorade) block of results/r6_uls_v2.json against the corrected
9-group / 72-qubit geometry (results/cs3_v2.json, re-run after fixing the
always-empty ring-family-0 defect documented in
results/group_index_audit.json). CS2/CS4/CS6 blocks are untouched (their
generators were not affected -- see group_index_audit.json's "only CS3
affected" finding) and are copied through byte-for-byte from the existing
results/r6_uls_v2.json (already backed up as
results/r6_uls_v2_phantomgroups.json).

Upgrades the CS3 entry beyond r6_uls_v2.py's original schema (stress +
buckling only) to match the richer bridge2_posthoc.py / vault_hex_posthoc.py
convention used elsewhere in this project:
  - `legacy_convention`    -- byte-compatible with r6_uls_v2.py's own
                              is_compression test (sign(Ni)==sign(axial_sign)
                              where axial_sign is Ni for a KNOWN-TENSION
                              calibration bar => this flag is actually TRUE
                              for TENSION, an inherited inversion, disclosed
                              not hidden -- same finding as Section 39.5).
  - `corrected_convention` -- Ni > 0 => COMPRESSION (verified rule, same as
                              dome2_posthoc.py / bridge2_posthoc.py).
  - serviceability: d_max vs span/250 (span = model.meta['span'] = 13.2 m
                              for hex_nexorade(3, 2.2, 0.25)).
  - link-tension fraction over the 153 excluded eccentricity/bearing LINK
    elements, corrected rule (Ni<0 => tension).

Recovery is IDENTICAL in method to r6_uls_v2.py's recover_cs3(): re-reads the
already-stored idx_history/comp_hist from cs3_v2.json's sa_states (no
re-optimization), rebuilds the frame at that idx, and asserts the recomputed
compliance matches the stored best_compliance value
(reproducibility_match=True required).
"""
import json
import sys
import time
import numpy as np
import os as _os_bootstrap
ROOT = _os_bootstrap.path.dirname(_os_bootstrap.path.dirname(_os_bootstrap.path.abspath(__file__)))

sys.path.insert(0, _os_bootstrap.path.join(ROOT, "code"))
from reciprocal_geom_v2 import hex_nexorade
from reciprocal_geom import build_frame, model_mass
from sections import CHS_CATALOG, E_STEEL

RESULTS = _os_bootstrap.path.join(ROOT, "results")
OUTFILE = f"{RESULTS}/r6_uls_v2.json"
FY = 355e6  # Pa, S355 steel


def axial_sign_convention():
    from frame3d import Frame3D
    frm = Frame3D()
    frm.add_node(0, 0, 0)
    frm.add_node(1, 0, 0)
    frm.add_element(0, 1, 210e9, 80e9, 1e-3, 1e-6, 1e-6, 2e-6, ref_vec=[0, 0, 1])
    frm.fix(0, dofs=(0, 1, 2, 3, 4, 5))
    frm.add_load(1, fx=+1000.0)
    frm.solve()
    f0 = frm.element_end_forces_local(0)
    return float(f0[0])


def recover_cs3():
    d = json.load(open(f"{RESULTS}/cs3_v2.json"))
    sa_states = d['sa_states']
    best_seed, best_C, best_idx = None, np.inf, None
    for s, st in sa_states.items():
        C = np.array(st['comp_hist'])
        bi = int(np.argmin(C))
        if C[bi] < best_C:
            best_C = float(C[bi]); best_seed = s; best_idx = [int(x) for x in st['idx_hist'][bi]]
    model = hex_nexorade(3, 2.2, 0.25)
    assert model.n_groups == 9, f"expected corrected 9-group hex_nexorade, got {model.n_groups}"
    stored_best = d['methods']['sa_qubo'][best_seed]['best_compliance']
    match = abs(best_C - stored_best) < 1e-6 * max(1.0, abs(stored_best))
    return model, {k: CHS_CATALOG[best_idx[k]] for k in range(model.n_groups)}, d['load_per_node'], best_seed, match


def _dist(arr):
    if len(arr) == 0:
        return dict(n=0)
    return dict(n=int(len(arr)), mean=float(arr.mean()), median=float(np.median(arr)),
                p90=float(np.percentile(arr, 90)), max=float(arr.max()),
                min=float(arr.min()), n_exceed_1=int(np.sum(arr > 1.0)))


def uls_checks(model, group_section, load_per_node, axial_sign, span, convention):
    """convention: 'legacy' (sign(Ni)==sign(axial_sign) => compression,
    inherited inversion, TRUE for tension) or 'corrected' (Ni>0 =>
    compression, verified rule)."""
    frm = build_frame(model, group_section, load_per_node=load_per_node)
    frm.solve()

    per_member, link_axials = [], []
    for ei, e in enumerate(model.elements):
        if e['kind'] == 'link':
            link_axials.append(float(frm.element_end_forces_local(ei)[0]))
            continue
        grp = e['group']
        sp = group_section[grp]
        A, Iy, Iz, D = sp['A'], sp['Iy'], sp['Iz'], sp['D']
        c = D / 2.0
        L = np.linalg.norm(np.array(model.nodes[e['i']]) - np.array(model.nodes[e['j']]))

        f = frm.element_end_forces_local(ei)
        Ni, Myi, Mzi = f[0], f[4], f[5]
        Nj, Myj, Mzj = f[6], f[10], f[11]

        N_abs = max(abs(Ni), abs(Nj))
        My_abs = max(abs(Myi), abs(Myj))
        Mz_abs = max(abs(Mzi), abs(Mzj))

        sigma = N_abs / A + My_abs * c / Iy + Mz_abs * c / Iz
        util_stress = sigma / FY

        if convention == 'legacy':
            is_compression = (np.sign(Ni) == np.sign(axial_sign)) if Ni != 0 else False
        else:
            is_compression = bool(Ni > 0.0)

        if is_compression and L > 0:
            N_comp = abs(Ni)
            P_cr = np.pi ** 2 * E_STEEL * Iy / L ** 2
            util_buck = N_comp / P_cr
        else:
            util_buck = 0.0

        per_member.append(dict(elem_idx=ei, group=grp, L=float(L), D_mm=sp['D_mm'],
                                N=float(Ni), My=float(My_abs), Mz=float(Mz_abs),
                                util_stress=float(util_stress),
                                is_compression=bool(is_compression),
                                util_buckling=float(util_buck)))

    stress_utils = np.array([m['util_stress'] for m in per_member])
    buck_utils_comp = np.array([m['util_buckling'] for m in per_member if m['is_compression']])
    link_axials = np.array(link_axials)
    if convention == 'legacy':
        n_link_tension = int(np.sum((np.sign(link_axials) == np.sign(axial_sign)) & (link_axials != 0)))
    else:
        n_link_tension = int(np.sum(link_axials < 0.0))
    n_link = len(link_axials)

    worst_stress = sorted(per_member, key=lambda m: -m['util_stress'])[:5]
    worst_buck = sorted([m for m in per_member if m['is_compression']],
                         key=lambda m: -m['util_buckling'])[:5]

    d_max = frm.max_abs_disp()
    span_limit = span / 250.0

    return dict(
        n_beam_members=len(per_member), n_link_excluded=n_link,
        n_compression_members=int(len(buck_utils_comp)),
        n_tension_members=int(len(per_member) - len(buck_utils_comp)),
        stress_utilization=_dist(stress_utils),
        buckling_utilization_compression_members=_dist(buck_utils_comp),
        worst_5_stress=worst_stress, worst_5_buckling=worst_buck,
        compliance=float(frm.compliance()), mass=float(model_mass(model, group_section)),
        d_max=float(d_max), span=float(span), span_over_250=float(span_limit),
        d_max_over_span_over_250=float(d_max / span_limit),
        serviceability_ok=bool(d_max <= span_limit),
        link_tension_fraction=float(n_link_tension / n_link) if n_link > 0 else float('nan'),
        n_link_tension=n_link_tension, n_link_total=n_link,
    )


def main():
    t0 = time.time()
    axial_sign = axial_sign_convention()
    print(f"axial sign convention (Ni for known TENSION) = {axial_sign:.4f}")

    model, gs, lpn, seed, match = recover_cs3()
    assert match, "CS3 reproducibility_match FAILED -- recomputed best_compliance does not match cs3_v2.json"
    span = model.meta['span']

    legacy = uls_checks(model, gs, lpn, axial_sign, span, 'legacy')
    corrected = uls_checks(model, gs, lpn, axial_sign, span, 'corrected')
    for blk in (legacy, corrected):
        blk['seed'] = seed
        blk['reproducibility_match'] = bool(match)

    new_cs3 = dict(
        note_phantom_group_fix=(
            "Re-computed 2026-07-21 after fixing hex_nexorade's always-empty "
            "ring-family-0 defect (results/group_index_audit.json): "
            "n_groups 12->9, sizing QUBO 96->72 qubits, geometry (240 beam + "
            "153 link elements) bit-for-bit unchanged. Pre-fix numbers "
            "preserved verbatim in results/r6_uls_v2_phantomgroups.json."),
        note_axial_sign=(
            "Reported under TWO compression conventions, same disclosure as "
            "Section 39.5 / bridge2_posthoc.py: `legacy_convention` reproduces "
            "r6_uls_v2.py's own is_compression test byte-for-byte (inverted -- "
            "TRUE for tension, since axial_sign is Ni measured on a KNOWN-"
            "TENSION calibration bar and equals sign(Ni)==sign(axial_sign)); "
            "`corrected_convention` uses the verified rule Ni>0 => COMPRESSION. "
            "Stress utilisation, compliance, mass and d_max are identical in "
            "both blocks; only the compression/tension split, buckling table, "
            "and link-tension fraction differ."),
        legacy_convention=legacy, corrected_convention=corrected,
        seed=seed, reproducibility_match=bool(match),
        # top-level convenience mirror of the corrected block, so any old
        # caller reading cs3.stress_utilization / .buckling_utilization_*
        # directly (as r6_uls_v2.py's own schema does for cs2/cs4/cs6) still
        # finds sensible (CORRECTED-convention) numbers at the old paths:
        n_beam_members=corrected['n_beam_members'],
        n_link_excluded=corrected['n_link_excluded'],
        n_compression_members=corrected['n_compression_members'],
        n_tension_members=corrected['n_tension_members'],
        stress_utilization=corrected['stress_utilization'],
        buckling_utilization_compression_members=corrected['buckling_utilization_compression_members'],
        worst_5_stress=corrected['worst_5_stress'],
        worst_5_buckling=corrected['worst_5_buckling'],
        compliance=corrected['compliance'], mass=corrected['mass'],
        d_max=corrected['d_max'], span=corrected['span'],
        span_over_250=corrected['span_over_250'],
        d_max_over_span_over_250=corrected['d_max_over_span_over_250'],
        serviceability_ok=corrected['serviceability_ok'],
        link_tension_fraction=corrected['link_tension_fraction'],
        n_link_tension=corrected['n_link_tension'], n_link_total=corrected['n_link_total'],
    )

    full = json.load(open(OUTFILE))
    full['cs3'] = new_cs3
    full.setdefault('note_cs3_update', new_cs3['note_phantom_group_fix'])
    with open(OUTFILE, 'w') as f:
        json.dump(full, f, indent=2)

    print(f"[cs3] seed={seed} match={match} n_beams={corrected['n_beam_members']} "
          f"stress_max={corrected['stress_utilization']['max']:.4f} "
          f"| legacy: n_comp={legacy['n_compression_members']} "
          f"buck_max={legacy['buckling_utilization_compression_members'].get('max')} "
          f"link_tens={legacy['link_tension_fraction']:.3f} "
          f"| corrected: n_comp={corrected['n_compression_members']} "
          f"buck_max={corrected['buckling_utilization_compression_members'].get('max')} "
          f"link_tens={corrected['link_tension_fraction']:.3f} "
          f"| d_max/span250={corrected['d_max_over_span_over_250']:.4f} "
          f"({time.time()-t0:.1f}s elapsed)")
    print("WROTE", OUTFILE)
    print("ALL DONE")


if __name__ == "__main__":
    main()

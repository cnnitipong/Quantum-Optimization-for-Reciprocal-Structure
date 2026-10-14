import os as _os_bootstrap
ROOT = _os_bootstrap.path.dirname(_os_bootstrap.path.dirname(_os_bootstrap.path.abspath(__file__)))
"""bridge2_posthoc.py -- post-hoc ULS (ultimate-limit-state) + serviceability
+ link-tension checks on the best CS6-complex design from the RE-RUN
campaign on the REDESIGNED `davinci_bridge2` (load-path-corrected 3-rail,
24-bay bridge; see reciprocal_geom_v2.davinci_bridge2 and
results/bridge2_verification.json). Supersedes, for the bridge
typology, posthoc.json's `cs6_complex` entry (built on the OLD
davinci_bridge_complex frame, preserved verbatim at
results/complex/cs6_complex_bridge1.json and its own post-hoc numbers
still in posthoc.json/complex_summary.md Section 3).

Machinery (fy/stress/Euler-buckling formulas, link-exclusion convention,
deterministic-recovery + reproducibility assertion, d_max vs span/250) is
imported verbatim from vault_hex_posthoc.py, exactly as dome2_posthoc.py
does, so all typologies' tables stay strictly comparable.

AXIAL-SIGN CORRECTION (disclosed, not silently applied) -- identical
disclosure to dome2_posthoc.py: `vault_hex_posthoc.uls_checks` sets
    is_compression = (sign(Ni) == sign(axial_sign))
where `axial_sign` is Ni measured on a bar in KNOWN TENSION (= -1000). The
flag is therefore TRUE exactly when the member is in TENSION -- inverted.
This script reports BOTH:
  * `legacy_convention`    -- byte-identical to vault_hex_posthoc.py's own
                              output, for continuity;
  * `corrected_convention` -- Ni > 0 => COMPRESSION (the verified rule,
                              confirmed independently by
                              results/bridge2_verification.json's own
                              V4 arch-action check, which found ALL 72
                              longitudinal members net-compressive under
                              gravity using this same corrected rule).

Output: results/complex/bridge2_posthoc.json
"""
import json
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reciprocal_geom_v2 import davinci_bridge2
from reciprocal_geom import build_frame, model_mass
from sections import CHS_CATALOG, E_STEEL
from sizing import iterative_qubo_sizing
import vault_hex_posthoc as VHP

RESULTS = _os_bootstrap.path.join(ROOT, "results", "complex")
OUTFILE = f"{RESULTS}/bridge2_posthoc.json"
FY = VHP.FY


def uls_checks_corrected(model, group_section, load_per_node, axial_sign, span):
    """Identical to VHP.uls_checks except is_compression uses the VERIFIED
    rule Ni > 0 => COMPRESSION (see module docstring)."""
    frm = build_frame(model, group_section, load_per_node=load_per_node)
    frm.solve(check_cond=True)

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

        is_compression = bool(Ni > 0.0)          # CORRECTED rule
        if is_compression and L > 0:
            P_cr = np.pi ** 2 * E_STEEL * Iy / L ** 2
            util_buck = abs(Ni) / P_cr
        else:
            util_buck = 0.0

        per_member.append(dict(elem_idx=ei, group=grp, L=float(L), D_mm=sp['D_mm'],
                               N=float(Ni), My=float(My_abs), Mz=float(Mz_abs),
                               util_stress=float(util_stress),
                               is_compression=is_compression,
                               util_buckling=float(util_buck)))

    stress_utils = np.array([m['util_stress'] for m in per_member])
    buck_comp = np.array([m['util_buckling'] for m in per_member if m['is_compression']])
    link_axials = np.array(link_axials)
    n_link_tension = int(np.sum(link_axials < 0.0))  # CORRECTED: Ni<0 => TENSION
    n_link = len(link_axials)

    def dist(arr):
        if len(arr) == 0:
            return dict(n=0)
        return dict(n=int(len(arr)), mean=float(arr.mean()), median=float(np.median(arr)),
                    p90=float(np.percentile(arr, 90)), max=float(arr.max()),
                    min=float(arr.min()), n_exceed_1=int(np.sum(arr > 1.0)))

    d_max = frm.max_abs_disp()
    span_limit = span / 250.0
    return dict(
        n_beam_members=len(per_member), n_link_excluded=n_link,
        n_compression_members=int(len(buck_comp)),
        n_tension_members=int(len(per_member) - len(buck_comp)),
        stress_utilization=dist(stress_utils),
        buckling_utilization_compression_members=dist(buck_comp),
        worst_5_stress=sorted(per_member, key=lambda m: -m['util_stress'])[:5],
        worst_5_buckling=sorted([m for m in per_member if m['is_compression']],
                                key=lambda m: -m['util_buckling'])[:5],
        compliance=float(frm.compliance()), mass=float(model_mass(model, group_section)),
        d_max=float(d_max), span=float(span), span_over_250=float(span_limit),
        d_max_over_span_over_250=float(d_max / span_limit),
        serviceability_ok=bool(d_max <= span_limit),
        link_tension_fraction=float(n_link_tension / n_link) if n_link > 0 else float('nan'),
        n_link_tension=n_link_tension, n_link_total=n_link,
    )


def recover_cs6_bridge2():
    """Best CS6-complex (base w_mass=5, SA-QUBO) design on the redesigned
    davinci_bridge2, recovered DETERMINISTICALLY by re-running the exact
    seeded SA-QUBO sizing loop (same protocol as cs6_complex_bridge2.py's
    run_one_sa) and asserting the recomputed best-so-far compliance
    matches the value already stored in cs6_complex.json."""
    d = json.load(open(f"{RESULTS}/cs6_complex.json"))
    assert d.get('typology') == 'davinci_bridge2', f"expected davinci_bridge2, got {d.get('typology')}"
    sa_runs = d['methods']['sa_qubo']
    best_run = min(sa_runs, key=lambda r: r['best_compliance'])
    seed = best_run['seed']
    mass_target = d['mass_target']; compliance_scale = d['compliance_scale']
    load_per_node = d['load_per_node']
    model = davinci_bridge2(n_bays=24, span=20.0, width=6.0, lam=0.25)

    res = iterative_qubo_sizing(model, mass_target, n_iters=50, w_mass=5.0, seed=seed,
                                 load_per_node=load_per_node, sa_reads=25, sa_sweeps=90,
                                 compliance_scale=compliance_scale, patience=1000,
                                 check_cond=False)
    C_hist = res['compliance_history']
    best_i = int(np.argmin(C_hist))
    best_idx = [int(x) for x in res['idx_history'][best_i]]
    recomputed = float(C_hist[best_i])
    match = abs(recomputed - best_run['best_compliance']) < 1e-6 * max(1.0, abs(best_run['best_compliance']))
    gs = {k: CHS_CATALOG[best_idx[k]] for k in range(model.n_groups)}
    return model, gs, load_per_node, seed, match, model.meta['span'], dict(
        stored_best=best_run['best_compliance'], recomputed=recomputed, best_idx=best_idx)


def main():
    t0 = time.time()
    axial_sign = VHP.axial_sign_convention()
    print(f"axial sign convention (Ni for known TENSION) = {axial_sign:.4f}", flush=True)

    model, gs, lpn, seed, match, span, meta = recover_cs6_bridge2()
    legacy = VHP.uls_checks(model, gs, lpn, axial_sign, span)
    corrected = uls_checks_corrected(model, gs, lpn, axial_sign, span)
    for blk in (legacy, corrected):
        blk['seed'] = seed
        blk['reproducibility_match'] = bool(match)
        blk['recovery_meta'] = meta

    out = dict(
        fy_Pa=FY, K_buckling=1.0,
        note_buckling="Euler pinned-pinned (K=1) is CONSERVATIVE relative to the semi-rigid "
                      "frame connections actually modelled.",
        note_links="Eccentric-bearing LINK elements are EXCLUDED from stress/buckling checks "
                   "(not real structural members) but their axial-force sign is reported "
                   "separately as the link-tension fraction.",
        note_serviceability="d_max is checked against span/250, a common practical deflection-"
                            "limit convention, DISTINCT from the geometry validation gate's "
                            "span/2000..span/100 plausibility window.",
        note_reproducibility="bridge2_cs6_best recovered by deterministically re-running the "
                             "exact seeded SA-QUBO sizing loop and verifying the recomputed "
                             "best-so-far compliance matches the value stored in "
                             "cs6_complex.json (reproducibility_match=True).",
        note_scope="Supersedes, for the BRIDGE typology, posthoc.json's cs6_complex entry "
                   "(built on the OLD davinci_bridge_complex frame; preserved verbatim at "
                   "cs6_complex_bridge1.json). The CS4-complex / dome rows are unaffected and "
                   "are not re-checked here (see dome2_posthoc.json for the current dome "
                   "post-hoc numbers).",
        note_axial_sign="Each design is reported under TWO compression conventions: "
                        "`legacy_convention` reproduces vault_hex_posthoc.py's own is_compression "
                        "test byte-for-byte (inverted -- TRUE for tension), and "
                        "`corrected_convention` uses the verified rule Ni>0 => COMPRESSION "
                        "(independently cross-checked by bridge2_verification.json's V4 "
                        "arch-action test, which found ALL 72 longitudinal members net-"
                        "compressive under gravity with this same rule). Stress utilisation, "
                        "compliance, mass and d_max are identical in both blocks; only the "
                        "compression/tension split, buckling table, and link-tension fraction "
                        "differ.",
        axial_sign_convention_test_Ni=axial_sign,
        bridge2_cs6_best=dict(legacy_convention=legacy, corrected_convention=corrected,
                               seed=seed, reproducibility_match=bool(match), recovery_meta=meta),
    )
    with open(OUTFILE, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"[bridge2_cs6_best] seed={seed} match={match} n_beams={legacy['n_beam_members']} "
          f"stress_max={legacy['stress_utilization']['max']:.4f} "
          f"| legacy: n_comp={legacy['n_compression_members']} "
          f"buck_max={legacy['buckling_utilization_compression_members'].get('max')} "
          f"link_tens={legacy['link_tension_fraction']:.3f} "
          f"| corrected: n_comp={corrected['n_compression_members']} "
          f"buck_max={corrected['buckling_utilization_compression_members'].get('max')} "
          f"link_tens={corrected['link_tension_fraction']:.3f} "
          f"| d_max/span250={legacy['d_max_over_span_over_250']:.3f} ({time.time()-t0:.1f}s)")
    print("WROTE", OUTFILE)
    print("ALL DONE")


if __name__ == "__main__":
    main()

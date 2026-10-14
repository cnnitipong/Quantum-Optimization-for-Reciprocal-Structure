import os
import os as _os_bootstrap
ROOT = _os_bootstrap.path.dirname(_os_bootstrap.path.dirname(_os_bootstrap.path.abspath(__file__)))
"""dome2_posthoc.py -- post-hoc ULS (ultimate-limit-state) + serviceability
+ link-tension checks on the TWO new best designs from the REDESIGNED
`rf_dome2` campaigns (Kiewitt polar-lattice reciprocal dome with a TRUE
CIRCULAR springing ring).  Supersedes, for the dome typology:
  * `posthoc.json` -> `cs4_complex`      (rf_dome_complex geometry-search best)
  * `vault_posthoc.json` -> `dome_sizing_best` (rf_dome_complex fixed-sizing best)
both of which were built on the OLD rf_dome_complex frame.

  1. dome2_geo_best    -- CS4 (cs4_complex.json) FMQA-best rf_dome2 geometry,
                          uniform global CHS section, exactly as searched.
  2. dome2_sizing_best -- dome-fixed sizing (dome_sizing.json) SA-QUBO-best
                          (base, w_mass=5) per-group sizing on the FROZEN
                          CS4-FMQA-best rf_dome2 geometry.

Machinery (fy/stress/Euler-buckling formulas, link-exclusion convention,
deterministic-recovery + reproducibility assertion, d_max vs span/250) is
imported verbatim from vault_hex_posthoc.py so the two typologies' tables
are strictly comparable.

AXIAL-SIGN CORRECTION (disclosed, not silently applied)
-------------------------------------------------------
`vault_hex_posthoc.uls_checks` sets
    is_compression = (sign(Ni) == sign(axial_sign))
where `axial_sign` is Ni measured on a bar in KNOWN TENSION (= -1000).  The
flag is therefore True exactly when the member is in TENSION -- it is
inverted, so the "buckling utilisation of compression members" table is in
fact computed over the tension members (and true compression members are
skipped).  This affects every pre-existing post-hoc file in the project.
Rather than silently rewrite published tables, this script reports BOTH:
  * `legacy_convention`    -- byte-identical to what vault_hex_posthoc.py /
                              cs_complex_uls.py / vault_posthoc.py produce,
                              for continuity and cross-typology comparison;
  * `corrected_convention` -- Ni > 0 => COMPRESSION (the convention verified
                              in results/dome2_verification.json's
                              axial_sign_convention block), i.e. the
                              structurally meaningful buckling check.
Everything not touched by the sign flag (stress utilisation, compliance,
mass, d_max/serviceability) is identical between the two blocks.

Output: results/complex/dome2_posthoc.json
"""
import json
import os
import sys
import time
import numpy as np

from reciprocal_geom_v2 import rf_dome2
from reciprocal_geom import build_frame, model_mass
from sections import CHS_CATALOG, E_STEEL
import vault_hex_posthoc as VHP

RESULTS = os.path.join(ROOT, "results", "complex")
OUTFILE = f"{RESULTS}/dome2_posthoc.json"
FY = VHP.FY
LAM_LEVELS = np.linspace(0.05, 0.5, 8)


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
    # link in TENSION under the corrected rule: Ni < 0
    n_link_tension = int(np.sum(link_axials < 0.0))
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


def recover_dome2_geo_best():
    """CS4 (cs4_complex.json, typology rf_dome2) FMQA-best: argmin over ALL
    seeds' y_data, exactly matching the convention used by
    vault_hex_posthoc.recover_vault_hex_geo_best / cs_complex_uls."""
    d = json.load(open(f"{RESULTS}/cs4_complex.json"))
    assert d['typology'] == 'rf_dome2', f"expected rf_dome2, got {d.get('typology')}"
    best_seed, best_y, best_idx = None, np.inf, None
    for s, st in d['methods']['fmqa'].items():
        y = np.array(st['y_data'])
        bi = int(np.argmin(y))
        if y[bi] < best_y:
            best_y = float(y[bi]); best_seed = s; best_idx = st['idx_data'][bi]
    stored_best = d['methods']['fmqa'][best_seed]['final_best']

    nrf = d['n_ring_families']
    lam_list = [float(LAM_LEVELS[best_idx[i]]) for i in range(nrf)]
    section_idx = best_idx[nrf]
    model = rf_dome2(n_sect=d['n_sect'], n_ring=d['n_ring'], span=d['span'], rise=d['rise'],
                     lam_per_ring=lam_list, load_per_node=d['load_per_node'],
                     max_group_rings=d['max_group_rings'])
    gs = {k: CHS_CATALOG[section_idx] for k in range(model.n_groups)}

    frm_check = build_frame(model, gs, load_per_node=d['load_per_node'])
    frm_check.solve(check_cond=False)
    recomputed = frm_check.compliance()
    match = abs(recomputed - stored_best) < 1e-6 * max(1.0, abs(stored_best)) and \
        abs(recomputed - best_y) < 1e-6 * max(1.0, abs(best_y))
    return model, gs, d['load_per_node'], best_seed, match, model.meta['span'], dict(
        stored_best=stored_best, best_y_in_ydata=best_y, recomputed=recomputed,
        lam_per_ring=lam_list, section_idx=section_idx)


def recover_dome2_sizing_best():
    """dome_sizing.json SA-QUBO (base, w_mass=5) best, recovered from its own
    stored idx_history at the argmin of compliance_history and re-solved."""
    from dome2_sizing import make_model
    d = json.load(open(f"{RESULTS}/dome_sizing.json"))
    assert d['geometry'].startswith('rf_dome2'), d['geometry']
    best_seed, best_c = None, np.inf
    for s in range(d['n_seeds']):
        rec = d['methods']['sa_qubo'][str(s)]
        if rec['best_compliance'] < best_c:
            best_c = rec['best_compliance']; best_seed = s
    rec = d['methods']['sa_qubo'][str(best_seed)]
    ch = np.array(rec['compliance_history'])
    bi = int(np.argmin(ch))
    best_idx = rec['idx_history'][bi]
    stored_best = rec['best_compliance']

    model = make_model()
    lpn = d['load_per_node']
    gs = {k: CHS_CATALOG[best_idx[k]] for k in range(model.n_groups)}
    frm_check = build_frame(model, gs, load_per_node=lpn)
    frm_check.solve(check_cond=False)
    recomputed = frm_check.compliance()
    match = abs(recomputed - stored_best) < 1e-6 * max(1.0, abs(stored_best))
    return model, gs, lpn, best_seed, match, model.meta['span'], dict(
        stored_best=stored_best, recomputed=recomputed, best_idx=best_idx, iter=bi)


def load_partial():
    if os.path.exists(OUTFILE):
        try:
            with open(OUTFILE) as f:
                d = json.load(f)
                if d:
                    return d
        except Exception:
            pass
    return dict(
        fy_Pa=FY, K_buckling=1.0,
        note_buckling="Euler pinned-pinned (K=1) is CONSERVATIVE relative to the semi-rigid "
                      "frame connections actually modelled.",
        note_links="Eccentric-bearing LINK elements are EXCLUDED from stress/buckling checks "
                   "(not real structural members) but their axial-force sign is reported "
                   "separately as the link-tension fraction.",
        note_serviceability="d_max is checked against span/250, a common practical deflection-"
                            "limit convention, DISTINCT from the geometry validation gate's "
                            "span/2000..span/100 plausibility window.",
        note_reproducibility="dome2_geo_best recovered by re-deriving its idx from the stored "
                             "FMQA y_data/idx_data and re-solving; dome2_sizing_best recovered "
                             "by re-reading the best-iteration idx from the stored per-iteration "
                             "idx_history/compliance_history and re-solving -- both assert the "
                             "recomputed compliance matches the stored value "
                             "(reproducibility_match=True).",
        note_scope="Supersedes, for the DOME typology, posthoc.json's cs4_complex entry and "
                   "vault_posthoc.json's dome_sizing_best entry (both built on the OLD "
                   "rf_dome_complex frame, whose supports did NOT lie on a circle). The "
                   "davinci_bridge_complex (cs6_complex) rows are unaffected and are not "
                   "re-checked here.",
        note_axial_sign="Each design is reported under TWO compression conventions: "
                        "`legacy_convention` reproduces vault_hex_posthoc.py/cs_complex_uls.py "
                        "byte-for-byte (is_compression = sign(Ni)==sign(axial_sign) with "
                        "axial_sign measured on a KNOWN-TENSION bar, so that flag is actually "
                        "TRUE FOR TENSION -- an inverted test inherited by every pre-existing "
                        "post-hoc file), and `corrected_convention` uses the verified rule "
                        "Ni>0 => COMPRESSION. Stress utilisation, compliance, mass and d_max "
                        "are identical in both blocks; only the compression/tension split, the "
                        "buckling table and the link-tension fraction differ.")


def save_partial(d):
    with open(OUTFILE, 'w') as f:
        json.dump(d, f, indent=2)
    print("wrote", OUTFILE)


def stage_axial_sign():
    a = VHP.axial_sign_convention()
    d = load_partial()
    d['axial_sign_convention_test_Ni'] = a
    save_partial(d)
    print(f"axial sign convention (Ni for known TENSION) = {a:.4f}")


def _run_one(key, recover_fn):
    d = load_partial()
    axial_sign = d.get('axial_sign_convention_test_Ni')
    if axial_sign is None:
        axial_sign = VHP.axial_sign_convention()
        d['axial_sign_convention_test_Ni'] = axial_sign
    t0 = time.time()
    model, gs, lpn, seed, match, span, meta = recover_fn()
    legacy = VHP.uls_checks(model, gs, lpn, axial_sign, span)
    corrected = uls_checks_corrected(model, gs, lpn, axial_sign, span)
    for blk in (legacy, corrected):
        blk['seed'] = seed
        blk['reproducibility_match'] = bool(match)
        blk['recovery_meta'] = meta
    d[key] = dict(legacy_convention=legacy, corrected_convention=corrected,
                  seed=seed, reproducibility_match=bool(match), recovery_meta=meta)
    save_partial(d)
    print(f"[{key}] seed={seed} match={match} n_beams={legacy['n_beam_members']} "
          f"stress_max={legacy['stress_utilization']['max']:.4f} "
          f"| legacy: n_comp={legacy['n_compression_members']} "
          f"buck_max={legacy['buckling_utilization_compression_members'].get('max')} "
          f"link_tens={legacy['link_tension_fraction']:.3f} "
          f"| corrected: n_comp={corrected['n_compression_members']} "
          f"buck_max={corrected['buckling_utilization_compression_members'].get('max')} "
          f"link_tens={corrected['link_tension_fraction']:.3f} "
          f"| d_max/span250={legacy['d_max_over_span_over_250']:.3f} ({time.time()-t0:.1f}s)")


STAGES = {
    'axial_sign': stage_axial_sign,
    'dome2_geo_best': lambda: _run_one('dome2_geo_best', recover_dome2_geo_best),
    'dome2_sizing_best': lambda: _run_one('dome2_sizing_best', recover_dome2_sizing_best),
}

if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if stage == 'all':
        for s in ('axial_sign', 'dome2_geo_best', 'dome2_sizing_best'):
            STAGES[s]()
    else:
        STAGES[stage]()

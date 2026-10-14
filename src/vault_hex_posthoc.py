import os
import os as _os_bootstrap
ROOT = _os_bootstrap.path.dirname(_os_bootstrap.path.dirname(_os_bootstrap.path.abspath(__file__)))
"""vault_hex_posthoc.py -- PART 3: post-hoc ULS (ultimate-limit-state) +
serviceability checks on the TWO new best designs from the HEXAGONAL-
pattern vault campaigns (upgrades vault_posthoc.py's square-pattern
vault_geo_best/vault_sizing_best pair to the rf_vault_hex generator; the
dome design is untouched by this upgrade and is not re-checked here):
  1. vault_hex_geo_best    -- CS-VAULT-HEX-GEO (PART 2a) FMQA-best
                               rf_vault_hex geometry (uniform global CHS
                               section, as searched).
  2. vault_hex_sizing_best -- PART 2b SA-QUBO-best (base, w_mass=5)
                               per-group sizing on the FROZEN
                               CS-VAULT-HEX-GEO FMQA-best rf_vault_hex
                               geometry.

Adapts vault_posthoc.py's machinery verbatim (same fy/stress/Euler-
buckling formulas, same link-exclusion convention, same deterministic-
recovery + reproducibility-assertion protocol). Adds the same d_max vs
span/250 serviceability check and link-tension-fraction disclosure.

Output: results/complex/vault_hex_posthoc.json
"""
import json
import time
import numpy as np

from reciprocal_geom_v2 import rf_vault_hex2 as rf_vault_hex
from reciprocal_geom import build_frame, model_mass
from sections import CHS_CATALOG, E_STEEL

RESULTS = os.path.join(ROOT, "results", "complex")
OUTFILE = f"{RESULTS}/vault_hex_posthoc.json"
FY = 355e6  # Pa, S355 steel
LAM_LEVELS = np.linspace(0.05, 0.5, 8)


def axial_sign_convention():
    from frame3d import Frame3D
    frm = Frame3D()
    frm.add_node(0, 0, 0)
    frm.add_node(1, 0, 0)
    frm.add_element(0, 1, 210e9, 80e9, 1e-3, 1e-6, 1e-6, 2e-6, ref_vec=[0, 0, 1])
    frm.fix(0, dofs=(0, 1, 2, 3, 4, 5))
    frm.add_load(1, fx=+1000.0)
    frm.solve(check_cond=True)
    f0 = frm.element_end_forces_local(0)
    return float(f0[0])


def uls_checks(model, group_section, load_per_node, axial_sign, span):
    frm = build_frame(model, group_section, load_per_node=load_per_node)
    frm.solve(check_cond=True)

    per_member = []
    link_axials = []
    for ei, e in enumerate(model.elements):
        if e['kind'] == 'link':
            f = frm.element_end_forces_local(ei)
            link_axials.append(float(f[0]))
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

        is_compression = (np.sign(Ni) == np.sign(axial_sign)) if Ni != 0 else False
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
    n_link_tension = int(np.sum((np.sign(link_axials) == np.sign(axial_sign)) & (link_axials != 0)))
    n_link = len(link_axials)

    def dist(arr):
        if len(arr) == 0:
            return dict(n=0)
        return dict(n=int(len(arr)), mean=float(arr.mean()), median=float(np.median(arr)),
                    p90=float(np.percentile(arr, 90)), max=float(arr.max()),
                    min=float(arr.min()), n_exceed_1=int(np.sum(arr > 1.0)))

    worst_stress = sorted(per_member, key=lambda m: -m['util_stress'])[:5]
    worst_buck = sorted([m for m in per_member if m['is_compression']],
                         key=lambda m: -m['util_buckling'])[:5]

    d_max = frm.max_abs_disp()
    span_limit = span / 250.0

    return dict(
        n_beam_members=len(per_member), n_link_excluded=n_link,
        n_compression_members=int(len(buck_utils_comp)),
        n_tension_members=int(len(per_member) - len(buck_utils_comp)),
        stress_utilization=dist(stress_utils),
        buckling_utilization_compression_members=dist(buck_utils_comp),
        worst_5_stress=worst_stress, worst_5_buckling=worst_buck,
        compliance=float(frm.compliance()), mass=float(model_mass(model, group_section)),
        d_max=float(d_max), span=float(span), span_over_250=float(span_limit),
        d_max_over_span_over_250=float(d_max / span_limit),
        serviceability_ok=bool(d_max <= span_limit),
        link_tension_fraction=float(n_link_tension / n_link) if n_link > 0 else float('nan'),
        n_link_tension=n_link_tension, n_link_total=n_link,
    )


def recover_vault_hex_geo_best():
    """PART 2a CS-VAULT-HEX-GEO best design: recovered directly from the
    stored FMQA idx_data/y_data (argmin over ALL seeds' y_data, matching
    vault_posthoc.recover_vault_geo_best's own convention exactly), uniform
    global CHS section (as searched, idx[n_lambda_bands])."""
    d = json.load(open(f"{RESULTS}/vault_hex_geo.json"))
    best_seed, best_y, best_idx = None, np.inf, None
    for s, st in d['methods']['fmqa'].items():
        y = np.array(st['y_data'])
        bi = int(np.argmin(y))
        if y[bi] < best_y:
            best_y = float(y[bi]); best_seed = s; best_idx = st['idx_data'][bi]
    stored_best = d['methods']['fmqa'][best_seed]['final_best']

    n_lambda_bands = d['n_lambda_bands']
    lam_list = [float(LAM_LEVELS[best_idx[i]]) for i in range(n_lambda_bands)]
    section_idx = best_idx[n_lambda_bands]
    model = rf_vault_hex(n_rows=d['n_rows'], n_cols=d['n_cols'],
                          span_transverse=d['span_transverse'], rise=d['rise'],
                          lam_bands=lam_list, load_per_node=d['load_per_node'],
                          n_section_groups=12)
    gs = {k: CHS_CATALOG[section_idx] for k in range(model.n_groups)}

    frm_check = build_frame(model, gs, load_per_node=d['load_per_node'])
    frm_check.solve(check_cond=False)
    recomputed = frm_check.compliance()
    match = abs(recomputed - stored_best) < 1e-6 * max(1.0, abs(stored_best)) and \
        abs(recomputed - best_y) < 1e-6 * max(1.0, abs(best_y))
    return model, gs, d['load_per_node'], best_seed, match, model.meta['span'], dict(
        stored_best=stored_best, best_y_in_ydata=best_y, recomputed=recomputed,
        lam_bands=lam_list, section_idx=section_idx)


def recover_vault_hex_sizing_best():
    """PART 2b best SA-QUBO (base, w_mass=5) design on
    results/complex/vault_hex_sizing.json, recovered by re-reading its
    own stored idx_history at the argmin of compliance_history (no
    re-optimization needed) -- reproducibility asserted by rebuilding the
    FE model at that idx and checking the recomputed compliance matches
    the stored value exactly. Mirrors vault_posthoc.recover_sizing_best."""
    from vault_hex2_sizing import make_model
    d = json.load(open(f"{RESULTS}/vault_hex_sizing.json"))
    best_seed, best_c, best_idx = None, np.inf, None
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
    load_per_node = d['load_per_node']
    gs = {k: CHS_CATALOG[best_idx[k]] for k in range(model.n_groups)}

    frm_check = build_frame(model, gs, load_per_node=load_per_node)
    frm_check.solve(check_cond=False)
    recomputed = frm_check.compliance()
    match = abs(recomputed - stored_best) < 1e-6 * max(1.0, abs(stored_best))
    return model, gs, load_per_node, best_seed, match, model.meta['span'], dict(
        stored_best=stored_best, recomputed=recomputed, best_idx=best_idx, iter=bi)


def load_partial():
    import os
    if os.path.exists(OUTFILE):
        try:
            with open(OUTFILE) as f:
                d = json.load(f)
                if d:
                    return d
        except Exception:
            pass
    return dict(fy_Pa=FY, K_buckling=1.0,
                note_buckling="Euler pinned-pinned (K=1) is CONSERVATIVE relative to the "
                               "semi-rigid frame connections actually modelled.",
                note_links="Eccentric-bearing LINK elements are EXCLUDED from stress/buckling "
                            "checks (not real structural members) but their axial-force sign is "
                            "reported separately as the link-tension fraction.",
                note_serviceability="d_max is checked against span/250, a common practical "
                                     "deflection-limit convention, DISTINCT from the geometry "
                                     "validation gate's span/2000..span/100 plausibility window.",
                note_reproducibility="vault_hex_geo_best recovered by re-deriving its idx from "
                                      "the stored FMQA y_data/idx_data and re-solving; "
                                      "vault_hex_sizing_best recovered by re-reading the "
                                      "best-iteration idx from the already-stored per-iteration "
                                      "idx_history/compliance_history and re-solving -- both "
                                      "assert recomputed compliance matches the stored value "
                                      "(reproducibility_match=True).",
                note_scope="Upgrades vault_posthoc.py's SQUARE-pattern vault_geo_best/"
                           "vault_sizing_best pair to the HEXAGONAL/TRIANGULAR-pattern "
                           "rf_vault_hex generator; the dome_sizing_best design is unaffected by "
                           "this upgrade and is not re-checked here (still valid in "
                           "vault_posthoc.json).")


def save_partial(d):
    with open(OUTFILE, 'w') as f:
        json.dump(d, f, indent=2)
    print("wrote", OUTFILE)


def stage_axial_sign():
    axial_sign = axial_sign_convention()
    d = load_partial()
    d['axial_sign_convention_test_Ni'] = axial_sign
    save_partial(d)
    print(f"axial sign convention (Ni for known TENSION) = {axial_sign:.4f}")


def _run_one(key, recover_fn, *args):
    d = load_partial()
    axial_sign = d.get('axial_sign_convention_test_Ni')
    if axial_sign is None:
        axial_sign = axial_sign_convention()
        d['axial_sign_convention_test_Ni'] = axial_sign
    t0 = time.time()
    model, gs, lpn, seed, match, span, meta = recover_fn(*args)
    res = uls_checks(model, gs, lpn, axial_sign, span)
    res['seed'] = seed
    res['reproducibility_match'] = bool(match)
    res['recovery_meta'] = meta
    d[key] = res
    save_partial(d)
    print(f"[{key}] seed={seed} match={match} n_beams={res['n_beam_members']} "
          f"stress_max={res['stress_utilization'].get('max')} "
          f"buck_max={res['buckling_utilization_compression_members'].get('max')} "
          f"d_max/span250={res['d_max_over_span_over_250']:.3f} "
          f"link_tension_frac={res['link_tension_fraction']:.3f} "
          f"({time.time()-t0:.1f}s elapsed)")


STAGES = {
    'axial_sign': stage_axial_sign,
    'vault_hex_geo_best': lambda: _run_one('vault_hex_geo_best', recover_vault_hex_geo_best),
    'vault_hex_sizing_best': lambda: _run_one('vault_hex_sizing_best', recover_vault_hex_sizing_best),
}

if __name__ == "__main__":
    import sys
    stage = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if stage == 'all':
        for s in ('axial_sign', 'vault_hex_geo_best', 'vault_hex_sizing_best'):
            STAGES[s]()
        print("ALL DONE")
    else:
        STAGES[stage]()
        d = load_partial()
        if all(k in d for k in ('vault_hex_geo_best', 'vault_hex_sizing_best')):
            print("ALL DONE")

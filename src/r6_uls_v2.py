import os
import os as _os_bootstrap
ROOT = _os_bootstrap.path.dirname(_os_bootstrap.path.dirname(_os_bootstrap.path.abspath(__file__)))
"""
r6_uls_v2.py -- post-hoc ULS (ultimate-limit-state) member checks on the
best CS2/CS3/CS4/CS6 v2 designs (square_nexorade, hex_nexorade, rf_dome,
davinci_bridge). Same approach as src/r6_uls.py: recover the section-index
vector that achieved the stored best_compliance (deterministic rerun where
needed, direct lookup where idx history was already stored), then check
normal-stress utilization (fy=355MPa) and Euler pinned-pinned (K=1)
buckling utilization for every BEAM element (link/eccentricity-connector
elements EXCLUDED).
"""
import json
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(ROOT, "src"))
from reciprocal_geom_v2 import square_nexorade, hex_nexorade, rf_dome, davinci_bridge
from reciprocal_geom import build_frame, model_mass
from sections import CHS_CATALOG, E_STEEL
from sizing import iterative_qubo_sizing

RESULTS = os.path.join(ROOT, "results")
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


def uls_checks(model, group_section, load_per_node, axial_sign):
    frm = build_frame(model, group_section, load_per_node=load_per_node)
    frm.solve()

    per_member = []
    n_link = 0
    for ei, e in enumerate(model.elements):
        if e['kind'] == 'link':
            n_link += 1
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
            P_cr = np.pi**2 * E_STEEL * Iy / L**2
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

    def dist(arr):
        if len(arr) == 0:
            return dict(n=0)
        return dict(n=int(len(arr)), mean=float(arr.mean()), median=float(np.median(arr)),
                    p90=float(np.percentile(arr, 90)), max=float(arr.max()),
                    min=float(arr.min()), n_exceed_1=int(np.sum(arr > 1.0)))

    worst_stress = sorted(per_member, key=lambda m: -m['util_stress'])[:5]
    worst_buck = sorted([m for m in per_member if m['is_compression']],
                         key=lambda m: -m['util_buckling'])[:5]

    return dict(
        n_beam_members=len(per_member), n_link_excluded=n_link,
        n_compression_members=int(len(buck_utils_comp)),
        n_tension_members=int(len(per_member) - len(buck_utils_comp)),
        stress_utilization=dist(stress_utils),
        buckling_utilization_compression_members=dist(buck_utils_comp),
        worst_5_stress=worst_stress, worst_5_buckling=worst_buck,
        compliance=float(frm.compliance()), mass=float(model_mass(model, group_section)),
    )


def recover_cs2():
    d = json.load(open(f"{RESULTS}/cs2_v2.json"))
    sa_runs = d['methods']['sa_qubo']
    best_run = min(sa_runs, key=lambda r: r['best_compliance'])
    seed = best_run['seed']
    mass_target = d['mass_target']; compliance_scale = d['compliance_scale']
    load_per_node = d['load_per_node']
    model = square_nexorade(6, 6, 2.5, 0.25)
    res = iterative_qubo_sizing(model, mass_target, n_iters=50, w_mass=5.0, seed=seed,
                                 load_per_node=load_per_node, sa_reads=25, sa_sweeps=90,
                                 compliance_scale=compliance_scale, patience=1000)
    C_hist = res['compliance_history']
    best_i = int(np.argmin(C_hist))
    best_idx = [int(x) for x in res['idx_history'][best_i]]
    match = abs(float(C_hist[best_i]) - best_run['best_compliance']) < 1e-6 * max(1.0, abs(best_run['best_compliance']))
    return model, {k: CHS_CATALOG[best_idx[k]] for k in range(model.n_groups)}, load_per_node, seed, match


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
    stored_best = d['methods']['sa_qubo'][best_seed]['best_compliance']
    match = abs(best_C - stored_best) < 1e-6 * max(1.0, abs(stored_best))
    return model, {k: CHS_CATALOG[best_idx[k]] for k in range(model.n_groups)}, d['load_per_node'], best_seed, match


def recover_cs4():
    d = json.load(open(f"{RESULTS}/cs4_v2.json"))
    fmqa_states = d['fmqa_states']
    best_seed, best_y, best_idx = None, np.inf, None
    for s, st in fmqa_states.items():
        y = np.array(st['y_data'])
        bi = int(np.argmin(y))
        if y[bi] < best_y:
            best_y = float(y[bi]); best_seed = s; best_idx = st['idx_data'][bi]
    stored_best = d['methods']['fmqa'][best_seed]['final_best']
    match = abs(best_y - stored_best) < 1e-6 * max(1.0, abs(stored_best))
    LAM_LEVELS = np.linspace(0.05, 0.5, 8)
    lam_list = [float(LAM_LEVELS[best_idx[i]]) for i in range(4)]
    section_idx = best_idx[4]
    model = rf_dome(3, span=18.0, rise=3.0, lam=0.25, lam_per_ring=lam_list,
                     load_per_node=d['load_per_node'])
    return model, {k: CHS_CATALOG[section_idx] for k in range(model.n_groups)}, d['load_per_node'], best_seed, match


def recover_cs6():
    d = json.load(open(f"{RESULTS}/cs6_v2.json"))
    sa_runs = d['methods']['sa_qubo']
    best_run = min(sa_runs, key=lambda r: r['best_compliance'])
    seed = best_run['seed']
    mass_target = d['mass_target']; compliance_scale = d['compliance_scale']
    load_per_node = d['load_per_node']
    model = davinci_bridge(6, span=16.0, width=4.0, lam=0.25)
    res = iterative_qubo_sizing(model, mass_target, n_iters=50, w_mass=5.0, seed=seed,
                                 load_per_node=load_per_node, sa_reads=25, sa_sweeps=90,
                                 compliance_scale=compliance_scale, patience=1000)
    C_hist = res['compliance_history']
    best_i = int(np.argmin(C_hist))
    best_idx = [int(x) for x in res['idx_history'][best_i]]
    match = abs(float(C_hist[best_i]) - best_run['best_compliance']) < 1e-6 * max(1.0, abs(best_run['best_compliance']))
    return model, {k: CHS_CATALOG[best_idx[k]] for k in range(model.n_groups)}, load_per_node, seed, match


def main():
    t0 = time.time()
    axial_sign = axial_sign_convention()
    print(f"axial sign convention (Ni for known TENSION) = {axial_sign:.4f}")

    out = {}
    recoverers = dict(cs2=recover_cs2, cs3=recover_cs3, cs4=recover_cs4, cs6=recover_cs6)
    for name, fn in recoverers.items():
        model, gs, load_per_node, seed, match = fn()
        res = uls_checks(model, gs, load_per_node, axial_sign)
        res['seed'] = seed
        res['reproducibility_match'] = bool(match)
        out[name] = res
        print(f"[{name}] seed={seed} match={match} n_beams={res['n_beam_members']} "
              f"stress_max={res['stress_utilization'].get('max')} "
              f"buck_max={res['buckling_utilization_compression_members'].get('max')} "
              f"({time.time()-t0:.1f}s elapsed)")

    final = dict(
        fy_Pa=FY, K_buckling=1.0,
        note_buckling="Euler pinned-pinned (K=1) is CONSERVATIVE relative to the "
                       "semi-rigid frame connections actually modelled.",
        note_links="Eccentric-bearing LINK elements are EXCLUDED from all ULS checks.",
        note_reproducibility="Section-index vectors recovered by deterministically "
                              "re-running the exact seeded sizing/FMQA loop and verifying "
                              "the recomputed best-so-far compliance matches the stored value.",
        axial_sign_convention_test_Ni=axial_sign,
        **out)
    with open(OUTFILE, 'w') as f:
        json.dump(final, f, indent=2)
    print("WROTE", OUTFILE)
    print("ALL DONE")


if __name__ == "__main__":
    main()

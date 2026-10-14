import os as _os_bootstrap
ROOT = _os_bootstrap.path.dirname(_os_bootstrap.path.dirname(_os_bootstrap.path.abspath(__file__)))
"""
R2-v2 -- matched-mass fairness re-run for CS2 (square_nexorade) and CS3
(hex_nexorade): SA-QUBO F1 sizing at w_mass=20, 10 seeds each, 50 real-FE
budget, otherwise IDENTICAL protocol to cs2_v2.py / cs3_v2.py (same
geometry, same mass target = 60% of max-section mass, same load, same
move-limit=2, same SA reads/sweeps as the w_mass=5 baseline already in
results/cs2_v2.json / cs3_v2.json).
"""
import json
import os
import sys
import time
import numpy as np
from reciprocal_geom_v2 import square_nexorade, hex_nexorade
from reciprocal_geom import model_mass, build_frame
from sections import CHS_CATALOG
from sizing import iterative_qubo_sizing

RESULTS = os.path.join(ROOT, "results")
OUTFILE = f"{RESULTS}/r2_matched_mass_v2.json"
N_FE_BUDGET = 50
N_SEEDS = 10
W_MASS = 20.0
TIME_BUDGET_S = float(sys.argv[1]) if len(sys.argv) > 1 else 38.0


def load_progress():
    if os.path.exists(OUTFILE):
        with open(OUTFILE) as f:
            try:
                d = json.load(f)
                if d:
                    return d
            except Exception:
                pass
    return None


def build_setup_cs2():
    model = square_nexorade(6, 6, 2.5, 0.25)
    load_per_node = model.meta['load_per_node']
    gs_max = {k: CHS_CATALOG[-1] for k in range(model.n_groups)}
    mass_max = model_mass(model, gs_max)
    mass_target = 0.6 * mass_max
    gs_mid = {k: CHS_CATALOG[3] for k in range(model.n_groups)}
    frm_mid = build_frame(model, gs_mid, load_per_node=load_per_node)
    frm_mid.solve()
    compliance_scale = frm_mid.compliance()
    return model, mass_target, mass_max, compliance_scale, load_per_node


def build_setup_cs3():
    model = hex_nexorade(3, 2.2, 0.25)
    load_per_node = model.meta['load_per_node']
    gs_max = {k: CHS_CATALOG[-1] for k in range(model.n_groups)}
    mass_max = model_mass(model, gs_max)
    mass_target = 0.6 * mass_max
    gs_mid = {k: CHS_CATALOG[3] for k in range(model.n_groups)}
    frm_mid = build_frame(model, gs_mid, load_per_node=load_per_node)
    frm_mid.solve()
    compliance_scale = frm_mid.compliance()
    return model, mass_target, mass_max, compliance_scale, load_per_node


def best_so_far(arr):
    return np.minimum.accumulate(np.asarray(arr, dtype=float))


def run_one_sa(model, mass_target, compliance_scale, load_per_node, seed):
    t0 = time.time()
    res = iterative_qubo_sizing(model, mass_target, n_iters=N_FE_BUDGET,
                                 w_mass=W_MASS, seed=seed,
                                 load_per_node=load_per_node,
                                 sa_reads=25, sa_sweeps=90,
                                 compliance_scale=compliance_scale,
                                 patience=1000)
    dt = time.time() - t0
    C_bsf = best_so_far(res['compliance_history'])
    return dict(seed=seed, time_s=dt,
                compliance_history=res['compliance_history'].tolist(),
                mass_history=res['mass_history'].tolist(),
                compliance_best_so_far=C_bsf.tolist(),
                final_idx=[int(x) for x in res['final_idx']],
                final_compliance=float(res['compliance_history'][-1]),
                final_mass=float(res['mass_history'][-1]),
                best_compliance=float(C_bsf[-1]))


def main():
    prog = load_progress()
    m2, mt2, mmax2, cs2_scale, lpn2 = build_setup_cs2()
    m3, mt3, mmax3, cs3_scale, lpn3 = build_setup_cs3()
    if prog is None:
        prog = dict(w_mass=W_MASS, n_fe_budget=N_FE_BUDGET, cs2={}, cs3={})
        prog['cs2'] = dict(mass_target=mt2, mass_max=mmax2,
                            compliance_scale=cs2_scale, load_per_node=lpn2,
                            n_groups=m2.n_groups, runs=[])
        prog['cs3'] = dict(mass_target=mt3, mass_max=mmax3,
                            compliance_scale=cs3_scale, load_per_node=lpn3,
                            n_groups=m3.n_groups, runs=[])
        with open(OUTFILE, 'w') as f:
            json.dump(prog, f)

    models = {'cs2': m2, 'cs3': m3}
    t_start = time.time()
    for case in ('cs2', 'cs3'):
        model = models[case]
        mt = prog[case]['mass_target']
        cscale = prog[case]['compliance_scale']
        lpn = prog[case]['load_per_node']
        done_seeds = {r['seed'] for r in prog[case]['runs']}
        for seed in range(N_SEEDS):
            if seed in done_seeds:
                continue
            if time.time() - t_start > TIME_BUDGET_S:
                print("Time budget reached; saving progress.")
                with open(OUTFILE, 'w') as f:
                    json.dump(prog, f)
                print_status(prog)
                return
            r = run_one_sa(model, mt, cscale, lpn, seed)
            prog[case]['runs'].append(r)
            print(f"[{case}] seed={seed} done in {r['time_s']:.1f}s "
                  f"best_C={r['best_compliance']:.2f} final_M={r['final_mass']:.1f}")
            with open(OUTFILE, 'w') as f:
                json.dump(prog, f)

    print_status(prog)
    n_total = len(prog['cs2']['runs']) + len(prog['cs3']['runs'])
    if n_total >= 2 * N_SEEDS:
        print("ALL DONE")


def print_status(prog):
    for c in ('cs2', 'cs3'):
        print(f"  {c}: {len(prog[c]['runs'])}/{N_SEEDS} seeds complete")


if __name__ == "__main__":
    main()

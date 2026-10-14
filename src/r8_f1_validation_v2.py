import os
import os as _os_bootstrap
ROOT = _os_bootstrap.path.dirname(_os_bootstrap.path.dirname(_os_bootstrap.path.abspath(__file__)))
"""
r8_f1_validation_v2.py -- CS1-v2 second part: R8-style F1-loop validation
on a small, EXACTLY brute-forceable v2 instance. Instance: v2's fan_unit
(reciprocal_geom_v2.fan_unit -- UNCHANGED rotor primitive, kept as-is per
the v2 geometry spec) with single_group=False: 3 member groups x 8 CHS
sections = 512 designs, exhaustively enumerated with real FE. Since
fan_unit is literally the same src/parameters as v1's r8_f1_validation.py
used, this is expected to (and, run here explicitly rather than assumed,
does) reproduce v1's numbers -- included for completeness/self-containment
of the v2 results tree rather than silently inheriting the v1 file.

w_mass=20, 50 iterations, 10 seeds (task spec).
"""
import json
import time
import itertools
import numpy as np
import sys

sys.path.insert(0, os.path.join(ROOT, "src"))
from reciprocal_geom_v2 import fan_unit
from reciprocal_geom import model_mass, build_frame
from sections import CHS_CATALOG, N_SECTIONS
from sizing import evaluate_design, iterative_qubo_sizing

RESULTS = os.path.join(ROOT, "results")
OUTFILE = f"{RESULTS}/r8_f1_validation_v2.json"
LOAD_PER_NODE = -2200.0  # fan_unit's own v2 default (matches CS1-v2 protocol)
W_MASS = 20.0
N_ITERS = 50
N_SEEDS = 10
FEAS_TOL = 0.05


def build_setup():
    model = fan_unit(3, R=3.0, lam=0.25, single_group=False, load_per_node=LOAD_PER_NODE)
    assert model.n_groups == 3
    gs_max = {k: CHS_CATALOG[-1] for k in range(3)}
    mass_max = model_mass(model, gs_max)
    mass_target = 0.6 * mass_max
    gs_mid = {k: CHS_CATALOG[3] for k in range(3)}
    frm_mid = build_frame(model, gs_mid, load_per_node=LOAD_PER_NODE)
    frm_mid.solve()
    compliance_scale = frm_mid.compliance()
    return model, mass_target, mass_max, compliance_scale


def enumerate_all(model, mass_target):
    all_results = []
    for combo in itertools.product(range(N_SECTIONS), repeat=3):
        C, M, _ = evaluate_design(model, list(combo), LOAD_PER_NODE)
        rel_err = abs(M - mass_target) / mass_target
        all_results.append(dict(idx=list(combo), compliance=C, mass=M,
                                 mass_rel_err=rel_err, feasible=bool(rel_err <= FEAS_TOL)))
    return all_results


def main():
    t0 = time.time()
    model, mass_target, mass_max, compliance_scale = build_setup()
    print(f"n_groups={model.n_groups} mass_max={mass_max:.2f} mass_target={mass_target:.2f}")

    all_results = enumerate_all(model, mass_target)
    print(f"enumerated {len(all_results)} designs in {time.time()-t0:.2f}s")
    feasible = [r for r in all_results if r['feasible']]
    print(f"{len(feasible)}/{len(all_results)} mass-feasible")
    true_opt = min(feasible, key=lambda r: r['compliance'])
    uncon_opt = min(all_results, key=lambda r: r['compliance'])

    seed_results = []
    for seed in range(N_SEEDS):
        ts = time.time()
        res = iterative_qubo_sizing(model, mass_target, n_iters=N_ITERS, w_mass=W_MASS,
                                     seed=seed, load_per_node=LOAD_PER_NODE,
                                     sa_reads=25, sa_sweeps=90,
                                     compliance_scale=compliance_scale, patience=1000)
        C_hist = np.array(res['compliance_history'])
        M_hist = np.array(res['mass_history'])
        rel_err_hist = np.abs(M_hist - mass_target) / mass_target
        feas_mask = rel_err_hist <= FEAS_TOL
        if feas_mask.any():
            best_feas_i = int(np.argmin(np.where(feas_mask, C_hist, np.inf)))
            best_feas_C = float(C_hist[best_feas_i])
            gap_pct = 100.0 * (best_feas_C - true_opt['compliance']) / true_opt['compliance']
            success = bool(gap_pct < 1e-6)
        else:
            best_feas_C, gap_pct, success = None, None, False
        seed_results.append(dict(seed=seed, time_s=time.time() - ts,
                                  best_feasible_compliance=best_feas_C,
                                  optimality_gap_pct=gap_pct, success=success,
                                  found_any_feasible=bool(feas_mask.any())))
        print(f"seed={seed} success={success} gap_pct={gap_pct} time={time.time()-ts:.2f}s")

    n_success = sum(1 for r in seed_results if r['success'])
    gaps = [r['optimality_gap_pct'] for r in seed_results if r['optimality_gap_pct'] is not None]

    out = dict(
        instance="fan_unit(3, R=3.0, lam=0.25, single_group=False) [v2, unchanged rotor]: "
                 "3 groups x 8 CHS sections = 512 designs",
        n_designs_total=len(all_results), mass_target=mass_target, mass_max=mass_max,
        n_feasible_designs=len(feasible), true_constrained_optimum=true_opt,
        true_unconstrained_optimum=uncon_opt, w_mass=W_MASS, n_iters=N_ITERS, n_seeds=N_SEEDS,
        seed_results=seed_results, success_rate=n_success / N_SEEDS,
        mean_optimality_gap_pct=(float(np.mean(gaps)) if gaps else None),
        std_optimality_gap_pct=(float(np.std(gaps)) if gaps else None),
    )
    with open(OUTFILE, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"success_rate={out['success_rate']*100:.0f}% mean_gap={out['mean_optimality_gap_pct']}")
    print("ALL DONE")


if __name__ == "__main__":
    main()

import os as _os_bootstrap
ROOT = _os_bootstrap.path.dirname(_os_bootstrap.path.dirname(_os_bootstrap.path.abspath(__file__)))
"""
Penalty-weight sensitivity study (CS2-v2, square_nexorade): re-runs the F1
iterative QUBO sizing with 4 different mass-target penalty weights w_mass,
3 seeds each, 30 iterations -- same protocol as src/cs_penalty_sensitivity.py.
"""
import json
import os
import sys
import time
import numpy as np
from reciprocal_geom_v2 import square_nexorade
from reciprocal_geom import model_mass, build_frame
from sections import CHS_CATALOG
from sizing import iterative_qubo_sizing

RESULTS = os.path.join(ROOT, "results")
OUTFILE = f"{RESULTS}/penalty_sensitivity_v2.json"
W_MASS_LEVELS = [0.5, 2.0, 5.0, 20.0]
N_SEEDS = 3
N_ITERS = 30
TIME_BUDGET_S = float(sys.argv[1]) if len(sys.argv) > 1 else 36.0


def load_progress():
    if os.path.exists(OUTFILE):
        try:
            with open(OUTFILE) as f:
                d = json.load(f)
                if d:
                    return d
        except Exception:
            pass
    return None


def build_setup():
    model = square_nexorade(6, 6, 2.5, 0.25)
    load_per_node = model.meta['load_per_node']
    gs_max = {k: CHS_CATALOG[-1] for k in range(model.n_groups)}
    mass_max = model_mass(model, gs_max)
    mass_target = 0.6 * mass_max
    gs_mid = {k: CHS_CATALOG[3] for k in range(model.n_groups)}
    frm_mid = build_frame(model, gs_mid, load_per_node=load_per_node)
    frm_mid.solve()
    compliance_scale = frm_mid.compliance()
    return model, mass_target, compliance_scale, load_per_node


def main():
    prog = load_progress()
    model, mass_target, compliance_scale, load_per_node = build_setup()
    if prog is None:
        prog = dict(mass_target=mass_target, w_mass_levels=W_MASS_LEVELS, runs=[])

    done_keys = {(r['w_mass'], r['seed']) for r in prog['runs']}
    t_start = time.time()
    for w_mass in W_MASS_LEVELS:
        for seed in range(N_SEEDS):
            if (w_mass, seed) in done_keys:
                continue
            if time.time() - t_start > TIME_BUDGET_S:
                print("Time budget reached; saving.")
                with open(OUTFILE, 'w') as f:
                    json.dump(prog, f)
                print(f"{len(prog['runs'])}/{len(W_MASS_LEVELS)*N_SEEDS} runs complete")
                return
            res = iterative_qubo_sizing(model, mass_target, n_iters=N_ITERS,
                                         w_mass=w_mass, seed=seed,
                                         load_per_node=load_per_node,
                                         sa_reads=25, sa_sweeps=90,
                                         compliance_scale=compliance_scale,
                                         patience=1000)
            final_C = float(res['compliance_history'][-1])
            final_M = float(res['mass_history'][-1])
            prog['runs'].append(dict(w_mass=w_mass, seed=seed,
                                      final_compliance=final_C, final_mass=final_M,
                                      mass_error_pct=100 * (final_M - mass_target) / mass_target))
            print(f"w_mass={w_mass} seed={seed} final_C={final_C:.1f} final_M={final_M:.1f} "
                  f"mass_err={100*(final_M-mass_target)/mass_target:.1f}%")
            with open(OUTFILE, 'w') as f:
                json.dump(prog, f)

    print(f"{len(prog['runs'])}/{len(W_MASS_LEVELS)*N_SEEDS} runs complete")
    if len(prog['runs']) >= len(W_MASS_LEVELS) * N_SEEDS:
        print("ALL DONE")


if __name__ == "__main__":
    main()

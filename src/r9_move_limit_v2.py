import os as _os_bootstrap
ROOT = _os_bootstrap.path.dirname(_os_bootstrap.path.dirname(_os_bootstrap.path.abspath(__file__)))
"""
r9_move_limit_v2.py -- move-limit (trust-region window) sweep on CS2-v2
(square_nexorade). Same protocol as src/r9_move_limit.py: move_limit in
{1,2,3,7}, 3 seeds per level, n_iters=50 (same FE budget as CS2-v2),
w_mass=5, sa_reads=25, sa_sweeps=90.
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
OUTFILE = f"{RESULTS}/r9_move_limit_v2.json"
N_FE_BUDGET = 50
W_MASS = 5.0
MOVE_LIMITS = [1, 2, 3, 7]
N_SEEDS = 3
TIME_BUDGET_S = float(sys.argv[1]) if len(sys.argv) > 1 else 38.0


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
    return model, mass_target, mass_max, compliance_scale, load_per_node


def run_one(model, mass_target, compliance_scale, load_per_node, move_limit, seed):
    t0 = time.time()
    res = iterative_qubo_sizing(model, mass_target, n_iters=N_FE_BUDGET,
                                 w_mass=W_MASS, seed=seed, move_limit=move_limit,
                                 load_per_node=load_per_node, sa_reads=25, sa_sweeps=90,
                                 compliance_scale=compliance_scale, patience=1000)
    dt = time.time() - t0
    C_bsf = np.minimum.accumulate(res['compliance_history'])
    return dict(move_limit=move_limit, seed=seed, time_s=dt,
                best_compliance=float(C_bsf[-1]),
                final_compliance=float(res['compliance_history'][-1]),
                final_mass=float(res['mass_history'][-1]))


def main():
    prog = load_progress()
    model, mass_target, mass_max, compliance_scale, load_per_node = build_setup()
    if prog is None:
        prog = dict(mass_target=mass_target, mass_max=mass_max, w_mass=W_MASS,
                    n_fe_budget=N_FE_BUDGET, move_limits=MOVE_LIMITS, n_seeds=N_SEEDS,
                    runs=[])

    t_start = time.time()
    done = {(r['move_limit'], r['seed']) for r in prog['runs']}
    for ml in MOVE_LIMITS:
        for seed in range(N_SEEDS):
            if (ml, seed) in done:
                continue
            if time.time() - t_start > TIME_BUDGET_S:
                print("time budget reached; saving progress")
                with open(OUTFILE, 'w') as f:
                    json.dump(prog, f, indent=2)
                print_status(prog)
                return
            r = run_one(model, mass_target, compliance_scale, load_per_node, ml, seed)
            prog['runs'].append(r)
            print(f"move_limit={ml} seed={seed} best_C={r['best_compliance']:.2f} "
                  f"time={r['time_s']:.1f}s")
            with open(OUTFILE, 'w') as f:
                json.dump(prog, f, indent=2)

    if 'summary' not in prog:
        summary = {}
        for ml in MOVE_LIMITS:
            rs = [r for r in prog['runs'] if r['move_limit'] == ml]
            comps = np.array([r['best_compliance'] for r in rs])
            summary[str(ml)] = dict(n=len(rs), mean_best_compliance=float(comps.mean()),
                                     std_best_compliance=float(comps.std()) if len(rs) > 1 else 0.0)
        prog['summary'] = summary
        with open(OUTFILE, 'w') as f:
            json.dump(prog, f, indent=2)

    print_status(prog)
    print("ALL DONE")


def print_status(prog):
    done = {}
    for r in prog['runs']:
        done.setdefault(r['move_limit'], 0)
        done[r['move_limit']] += 1
    for ml in MOVE_LIMITS:
        print(f"  move_limit={ml}: {done.get(ml,0)}/{N_SEEDS} seeds complete")


if __name__ == "__main__":
    main()

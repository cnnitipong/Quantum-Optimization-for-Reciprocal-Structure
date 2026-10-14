import os as _os_bootstrap
ROOT = _os_bootstrap.path.dirname(_os_bootstrap.path.dirname(_os_bootstrap.path.abspath(__file__)))
"""
R3-v2 -- greedy-linearized ablation study on CS2 (square_nexorade). 10
seeds, BOTH w_mass=5 and w_mass=20, comparing greedy_coordinate_sizing
(deterministic per-iteration subproblem solve, random initial design per
seed) vs iterative_qubo_sizing / SA-QUBO with the SAME random initial
design per seed (paired comparison). Same real-FE budget (50) and mass
target (60% of max-section mass) as cs2_v2.py.
"""
import json
import os
import sys
import time
import numpy as np
from reciprocal_geom_v2 import square_nexorade
from reciprocal_geom import model_mass, build_frame
from sections import CHS_CATALOG, N_SECTIONS
from sizing import iterative_qubo_sizing
from greedy_sizing import greedy_coordinate_sizing

RESULTS = os.path.join(ROOT, "results")
OUTFILE = f"{RESULTS}/r3_ablation_v2.json"
N_FE_BUDGET = 50
N_SEEDS = 10
W_MASS_LEVELS = [5.0, 20.0]
TIME_BUDGET_S = float(sys.argv[1]) if len(sys.argv) > 1 else 38.0


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


def random_init(seed, n_groups):
    rng = np.random.default_rng(seed)
    return [int(rng.integers(0, N_SECTIONS)) for _ in range(n_groups)]


def best_so_far(arr):
    return np.minimum.accumulate(np.asarray(arr, dtype=float))


def run_one_greedy(model, mass_target, load_per_node, w_mass, seed):
    init = random_init(seed, model.n_groups)
    t0 = time.time()
    res = greedy_coordinate_sizing(model, mass_target, n_iters=N_FE_BUDGET,
                                    w_mass=w_mass, seed=seed, init_idx=init,
                                    load_per_node=load_per_node,
                                    move_limit=2, patience=1000)
    dt = time.time() - t0
    C_bsf = best_so_far(res['compliance_history'])
    return dict(seed=seed, time_s=dt, init_idx=init,
                compliance_history=res['compliance_history'].tolist(),
                mass_history=res['mass_history'].tolist(),
                compliance_best_so_far=C_bsf.tolist(),
                final_idx=[int(x) for x in res['final_idx']],
                final_compliance=float(res['compliance_history'][-1]),
                final_mass=float(res['mass_history'][-1]),
                best_compliance=float(C_bsf[-1]))


def run_one_sa_paired(model, mass_target, compliance_scale, load_per_node, w_mass, seed):
    init = random_init(seed, model.n_groups)
    t0 = time.time()
    res = iterative_qubo_sizing(model, mass_target, n_iters=N_FE_BUDGET,
                                 w_mass=w_mass, seed=seed, init_idx=init,
                                 load_per_node=load_per_node,
                                 sa_reads=25, sa_sweeps=90,
                                 compliance_scale=compliance_scale,
                                 move_limit=2, patience=1000)
    dt = time.time() - t0
    C_bsf = best_so_far(res['compliance_history'])
    return dict(seed=seed, time_s=dt, init_idx=init,
                compliance_history=res['compliance_history'].tolist(),
                mass_history=res['mass_history'].tolist(),
                compliance_best_so_far=C_bsf.tolist(),
                final_idx=[int(x) for x in res['final_idx']],
                final_compliance=float(res['compliance_history'][-1]),
                final_mass=float(res['mass_history'][-1]),
                best_compliance=float(C_bsf[-1]))


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


def main():
    model, mass_target, mass_max, compliance_scale, load_per_node = build_setup()
    prog = load_progress()
    if prog is None:
        prog = dict(mass_target=mass_target, mass_max=mass_max,
                     compliance_scale=compliance_scale, load_per_node=load_per_node,
                     n_groups=model.n_groups, n_fe_budget=N_FE_BUDGET,
                     methods={f"greedy_w{int(w)}": [] for w in W_MASS_LEVELS})
        prog['methods'].update({f"sa_qubo_paired_w{int(w)}": [] for w in W_MASS_LEVELS})
        with open(OUTFILE, 'w') as f:
            json.dump(prog, f)

    t_start = time.time()
    tasks = []
    for w in W_MASS_LEVELS:
        for seed in range(N_SEEDS):
            tasks.append(('greedy', w, seed))
    for w in W_MASS_LEVELS:
        for seed in range(N_SEEDS):
            tasks.append(('sa', w, seed))

    for kind, w, seed in tasks:
        key = f"{'greedy' if kind == 'greedy' else 'sa_qubo_paired'}_w{int(w)}"
        done_seeds = {r['seed'] for r in prog['methods'][key]}
        if seed in done_seeds:
            continue
        if time.time() - t_start > TIME_BUDGET_S:
            print("Time budget reached; saving.")
            with open(OUTFILE, 'w') as f:
                json.dump(prog, f)
            print_status(prog)
            return
        if kind == 'greedy':
            r = run_one_greedy(model, mass_target, load_per_node, w, seed)
        else:
            r = run_one_sa_paired(model, mass_target, compliance_scale, load_per_node, w, seed)
        prog['methods'][key].append(r)
        print(f"[{key}] seed={seed} done in {r['time_s']:.1f}s "
              f"best_C={r['best_compliance']:.2f} final_M={r['final_mass']:.1f}")
        with open(OUTFILE, 'w') as f:
            json.dump(prog, f)

    print_status(prog)
    n_total = sum(len(v) for v in prog['methods'].values())
    if n_total >= len(tasks):
        print("ALL DONE")


def print_status(prog):
    for m, runs in prog['methods'].items():
        print(f"  {m}: {len(runs)}/{N_SEEDS} seeds complete")


if __name__ == "__main__":
    main()

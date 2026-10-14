import os as _os_bootstrap
ROOT = _os_bootstrap.path.dirname(_os_bootstrap.path.dirname(_os_bootstrap.path.abspath(__file__)))
"""
CS2 (v2 geometry) -- square_nexorade (literature-faithful rotated-square
nexorade, member-fused per interior grid edge, perimeter-only supports)
discrete sizing under a mass-target constraint.

Geometry: square_nexorade(6, 6, cell=2.5, lam=0.25) -- SAME construction
parameters used to pass the geometry-v2 validation gates
(results/geometry_v2_stats.json: n_groups=12, 220 nodes, 260 FE elements,
60 physical members [40 interior/fused, 20 boundary], 20 perimeter
supports). Members are grouped by symmetry orbit (ring, direction) exactly
as built into square_nexorade's `group = ring*4 + k` assignment -- 3 rings x
4 directions = 12 groups, unchanged from the v1 CS2 protocol's 12-group
count. Service load = model's own default (-450 N/node, the value used by
the geometry-v2 gates), applied at every connection ("tip"/engagement)
node exactly as square_nexorade's own load_idx list defines.

Protocol identical to v1 cs2_grillage.py: 50 real-FE-evaluation budget,
10 seeds, w_mass=5, mass target = 60% of all-max-section mass. SA-QUBO (F1)
vs GA (pop 20) vs random search. Resumable/checkpointed (45s/bash-call
sandbox).
"""
import json
import os
import sys
import time
import numpy as np
from reciprocal_geom_v2 import square_nexorade
from reciprocal_geom import build_frame, model_mass
from sections import CHS_CATALOG, N_SECTIONS
from sizing import iterative_qubo_sizing, evaluate_design
from qubo import genetic_algorithm, random_search

RESULTS = os.path.join(ROOT, "results")
OUTFILE = f"{RESULTS}/cs2_v2.json"
N_FE_BUDGET = 50
N_SEEDS = 10
W_MASS = 5.0
MASS_FRAC = 0.6
TIME_BUDGET_S = float(sys.argv[1]) if len(sys.argv) > 1 else 38.0


def make_model():
    return square_nexorade(6, 6, 2.5, 0.25)


def make_logging_objective(model, mass_target, w_mass, compliance_scale, load_per_node):
    log = {'C': [], 'M': [], 'obj': []}

    def obj(idx_vec):
        try:
            C, M, _ = evaluate_design(model, idx_vec, load_per_node)
        except np.linalg.LinAlgError:
            C, M = 1e9, 1e9
        rel = (M - mass_target) / mass_target
        f = C / compliance_scale + w_mass * rel**2
        log['C'].append(C); log['M'].append(M); log['obj'].append(f)
        return f

    return obj, log


def best_so_far(arr):
    return np.minimum.accumulate(np.asarray(arr, dtype=float))


def load_progress():
    if os.path.exists(OUTFILE):
        with open(OUTFILE) as f:
            return json.load(f)
    return None


def build_setup():
    model = make_model()
    load_per_node = model.meta['load_per_node']
    gs_max = {k: CHS_CATALOG[-1] for k in range(model.n_groups)}
    mass_max = model_mass(model, gs_max)
    mass_target = MASS_FRAC * mass_max
    gs_mid = {k: CHS_CATALOG[3] for k in range(model.n_groups)}
    frm_mid = build_frame(model, gs_mid, load_per_node=load_per_node)
    frm_mid.solve()
    compliance_scale = frm_mid.compliance()
    return model, mass_target, mass_max, compliance_scale, load_per_node


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


def run_one_ga(model, mass_target, compliance_scale, load_per_node, seed):
    obj, log = make_logging_objective(model, mass_target, W_MASS, compliance_scale, load_per_node)
    t0 = time.time()
    res = genetic_algorithm(obj, [N_SECTIONS] * model.n_groups,
                             n_evals=N_FE_BUDGET, pop_size=20, seed=seed)
    dt = time.time() - t0
    bsf_idx = [int(np.argmin(log['obj'][:i + 1])) for i in range(len(log['obj']))]
    C_bsf = [log['C'][i] for i in bsf_idx]
    best_i = int(np.argmin(log['obj']))
    return dict(seed=seed, time_s=dt, compliance_history=log['C'],
                mass_history=log['M'], compliance_best_so_far=C_bsf,
                final_idx=[int(x) for x in res['best_x']],
                final_compliance=float(log['C'][best_i]),
                final_mass=float(log['M'][best_i]),
                best_compliance=float(log['C'][best_i]))


def run_one_random(model, mass_target, compliance_scale, load_per_node, seed):
    obj, log = make_logging_objective(model, mass_target, W_MASS, compliance_scale, load_per_node)
    t0 = time.time()
    res = random_search(obj, [N_SECTIONS] * model.n_groups,
                         n_evals=N_FE_BUDGET, seed=seed)
    dt = time.time() - t0
    bsf_idx = [int(np.argmin(log['obj'][:i + 1])) for i in range(len(log['obj']))]
    C_bsf = [log['C'][i] for i in bsf_idx]
    best_i = int(np.argmin(log['obj']))
    return dict(seed=seed, time_s=dt, compliance_history=log['C'],
                mass_history=log['M'], compliance_best_so_far=C_bsf,
                final_idx=[int(x) for x in res['best_x']],
                final_compliance=float(log['C'][best_i]),
                final_mass=float(log['M'][best_i]),
                best_compliance=float(log['C'][best_i]))


RUNNERS = {'sa_qubo': run_one_sa, 'ga': run_one_ga, 'random': run_one_random}


def main():
    prog = load_progress()
    model, mass_target, mass_max, compliance_scale, load_per_node = build_setup()
    if prog is None:
        prog = dict(mass_target=mass_target, mass_max=mass_max,
                    compliance_scale=compliance_scale, load_per_node=load_per_node,
                    n_groups=model.n_groups, n_fe_budget=N_FE_BUDGET,
                    w_mass=W_MASS, geometry='square_nexorade(6,6,cell=2.5,lam=0.25)',
                    methods={'sa_qubo': [], 'ga': [], 'random': []})

    t_start = time.time()
    for method in ('sa_qubo', 'ga', 'random'):
        done_seeds = {r['seed'] for r in prog['methods'][method]}
        for seed in range(N_SEEDS):
            if seed in done_seeds:
                continue
            if time.time() - t_start > TIME_BUDGET_S:
                print("Time budget reached; saving progress.")
                with open(OUTFILE, 'w') as f:
                    json.dump(prog, f)
                print_status(prog)
                return
            r = RUNNERS[method](model, mass_target, compliance_scale, load_per_node, seed)
            prog['methods'][method].append(r)
            print(f"[{method}] seed={seed} done in {r['time_s']:.1f}s "
                  f"best_C={r['best_compliance']:.2f} final_M={r['final_mass']:.1f}")
            with open(OUTFILE, 'w') as f:
                json.dump(prog, f)

    print_status(prog)
    n_total = sum(len(v) for v in prog['methods'].values())
    if n_total >= 3 * N_SEEDS:
        print("ALL DONE")


def print_status(prog):
    for m, runs in prog['methods'].items():
        print(f"  {m}: {len(runs)}/{N_SEEDS} seeds complete")


if __name__ == "__main__":
    main()

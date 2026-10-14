import os as _os_bootstrap
ROOT = _os_bootstrap.path.dirname(_os_bootstrap.path.dirname(_os_bootstrap.path.abspath(__file__)))
"""
cs6_complex_bridge2.py -- CS6-complex sizing campaign RE-RUN on the
REDESIGNED `davinci_bridge2` (load-path-corrected 3-rail, 24-bay Leonardo
bridge; 259 nodes, 302 FE elements, 118 physical members, 16 groups --
see reciprocal_geom_v2.davinci_bridge2 and
results/bridge2_verification.json). Supersedes cs6_complex.py's
campaign on the OLD davinci_bridge_complex, whose data is preserved
verbatim at results/complex/cs6_complex_bridge1.json.

IDENTICAL protocol to cs6_complex.py (same 8 method-keys, same 50-real-FE
budget, same 10 seeds, same w_mass levels, same SA-QUBO
sa_reads/sa_sweeps, same mid-catalog fixed start / paired random-init
convention) -- only the geometry changed. Non-singularity of this
geometry/section-assignment class was verified beforehand (worst-case
cond ~4.42e11 across lam in [0.05,0.5] x sections {0,3,7}, see
results/complex/nonsingular_class_verification.json's
`davinci_bridge2` entry), so check_cond=False is used throughout the
inner loop exactly as before.

Output: results/complex/cs6_complex.json (OVERWRITES the campaign
data; the old davinci_bridge_complex run is preserved at
cs6_complex_bridge1.json).
"""
import json
import os
import sys
import time
import numpy as np
from reciprocal_geom_v2 import davinci_bridge2
from reciprocal_geom import build_frame, model_mass
from sections import CHS_CATALOG, N_SECTIONS
from sizing import iterative_qubo_sizing, evaluate_design
from greedy_sizing import greedy_coordinate_sizing
from qubo import genetic_algorithm, random_search

RESULTS = _os_bootstrap.path.join(ROOT, "results", "complex")
OUTFILE = f"{RESULTS}/cs6_complex.json"
N_FE_BUDGET = 50
N_SEEDS = 10
W_MASS_LEVELS = [5.0, 20.0]
MASS_FRAC = 0.6
TIME_BUDGET_S = float(sys.argv[1]) if len(sys.argv) > 1 else 38.0


def make_model():
    return davinci_bridge2(n_bays=24, span=20.0, width=6.0, lam=0.25)


def build_setup():
    model = make_model()
    load_per_node = model.meta['load_per_node']
    gs_max = {k: CHS_CATALOG[-1] for k in range(model.n_groups)}
    mass_max = model_mass(model, gs_max)
    mass_target = MASS_FRAC * mass_max
    gs_mid = {k: CHS_CATALOG[3] for k in range(model.n_groups)}
    frm_mid = build_frame(model, gs_mid, load_per_node=load_per_node)
    frm_mid.solve(check_cond=False)
    compliance_scale = frm_mid.compliance()
    return model, mass_target, mass_max, compliance_scale, load_per_node


def make_logging_objective(model, mass_target, w_mass, compliance_scale, load_per_node):
    log = {'C': [], 'M': [], 'obj': []}

    def obj(idx_vec):
        try:
            C, M, _ = evaluate_design(model, idx_vec, load_per_node, check_cond=False)
        except np.linalg.LinAlgError:
            C, M = 1e9, 1e9
        rel = (M - mass_target) / mass_target
        f = C / compliance_scale + w_mass * rel**2
        log['C'].append(C); log['M'].append(M); log['obj'].append(f)
        return f

    return obj, log


def best_so_far(arr):
    return np.minimum.accumulate(np.asarray(arr, dtype=float))


def random_init(seed, n_groups):
    rng = np.random.default_rng(seed)
    return [int(rng.integers(0, N_SECTIONS)) for _ in range(n_groups)]


def load_progress():
    if os.path.exists(OUTFILE):
        with open(OUTFILE) as f:
            try:
                d = json.load(f)
                if d and d.get('geometry', '').startswith('davinci_bridge2'):
                    return d
            except Exception:
                pass
    return None


# ---------------- runners (identical protocol to cs6_complex.py) ----------------
def run_one_sa(model, mass_target, compliance_scale, load_per_node, seed, w_mass=5.0, init_idx=None):
    t0 = time.time()
    res = iterative_qubo_sizing(model, mass_target, n_iters=N_FE_BUDGET,
                                 w_mass=w_mass, seed=seed,
                                 load_per_node=load_per_node,
                                 sa_reads=25, sa_sweeps=90, init_idx=init_idx,
                                 compliance_scale=compliance_scale,
                                 patience=1000, check_cond=False)
    dt = time.time() - t0
    C_bsf = best_so_far(res['compliance_history'])
    return dict(seed=seed, time_s=dt,
                compliance_history=res['compliance_history'].tolist(),
                mass_history=res['mass_history'].tolist(),
                compliance_best_so_far=C_bsf.tolist(),
                idx_history=[[int(v) for v in row] for row in res['idx_history']],
                final_idx=[int(x) for x in res['final_idx']],
                final_compliance=float(res['compliance_history'][-1]),
                final_mass=float(res['mass_history'][-1]),
                best_compliance=float(C_bsf[-1]))


def run_one_greedy(model, mass_target, load_per_node, w_mass, seed, init_idx=None):
    t0 = time.time()
    res = greedy_coordinate_sizing(model, mass_target, n_iters=N_FE_BUDGET,
                                    w_mass=w_mass, seed=seed, init_idx=init_idx,
                                    load_per_node=load_per_node,
                                    move_limit=2, patience=1000, check_cond=False)
    dt = time.time() - t0
    C_bsf = best_so_far(res['compliance_history'])
    return dict(seed=seed, time_s=dt, init_idx=init_idx,
                compliance_history=res['compliance_history'].tolist(),
                mass_history=res['mass_history'].tolist(),
                compliance_best_so_far=C_bsf.tolist(),
                final_idx=[int(x) for x in res['final_idx']],
                final_compliance=float(res['compliance_history'][-1]),
                final_mass=float(res['mass_history'][-1]),
                best_compliance=float(C_bsf[-1]))


def run_one_ga(model, mass_target, compliance_scale, load_per_node, seed, w_mass=5.0):
    obj, log = make_logging_objective(model, mass_target, w_mass, compliance_scale, load_per_node)
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


def run_one_random(model, mass_target, compliance_scale, load_per_node, seed, w_mass=5.0):
    obj, log = make_logging_objective(model, mass_target, w_mass, compliance_scale, load_per_node)
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


def main():
    prog = load_progress()
    model, mass_target, mass_max, compliance_scale, load_per_node = build_setup()
    method_keys = ['sa_qubo', 'ga', 'random', 'sa_qubo_w20',
                   'greedy_w5', 'sa_paired_w5', 'greedy_w20', 'sa_paired_w20']
    if prog is None:
        prog = dict(mass_target=mass_target, mass_max=mass_max,
                    compliance_scale=compliance_scale, load_per_node=load_per_node,
                    n_groups=model.n_groups, n_fe_budget=N_FE_BUDGET,
                    geometry='davinci_bridge2(n_bays=24,span=20.0,width=6.0,lam=0.25)',
                    typology='davinci_bridge2',
                    redesign_of='davinci_bridge_complex',
                    note_supersedes='Supersedes cs6_complex_bridge1.json (old davinci_bridge_complex '
                                     'campaign); see reciprocal_geom_v2.davinci_bridge2 and '
                                     'results/bridge_diagnosis.json for the redesign rationale.',
                    methods={k: [] for k in method_keys})

    t_start = time.time()
    tasks = []
    for seed in range(N_SEEDS):
        tasks.append(('sa_qubo', seed))
        tasks.append(('ga', seed))
        tasks.append(('random', seed))
    for seed in range(N_SEEDS):
        tasks.append(('sa_qubo_w20', seed))
    for w in W_MASS_LEVELS:
        for seed in range(N_SEEDS):
            tasks.append((f'greedy_w{int(w)}', seed))
    for w in W_MASS_LEVELS:
        for seed in range(N_SEEDS):
            tasks.append((f'sa_paired_w{int(w)}', seed))

    for key, seed in tasks:
        done_seeds = {r['seed'] for r in prog['methods'][key]}
        if seed in done_seeds:
            continue
        if time.time() - t_start > TIME_BUDGET_S:
            print("Time budget reached; saving.")
            with open(OUTFILE, 'w') as f:
                json.dump(prog, f)
            print_status(prog)
            return
        if key == 'sa_qubo':
            r = run_one_sa(model, mass_target, compliance_scale, load_per_node, seed, w_mass=5.0)
        elif key == 'ga':
            r = run_one_ga(model, mass_target, compliance_scale, load_per_node, seed, w_mass=5.0)
        elif key == 'random':
            r = run_one_random(model, mass_target, compliance_scale, load_per_node, seed, w_mass=5.0)
        elif key == 'sa_qubo_w20':
            r = run_one_sa(model, mass_target, compliance_scale, load_per_node, seed, w_mass=20.0)
        elif key.startswith('greedy_w'):
            w = float(key.split('w')[-1])
            init = random_init(seed, model.n_groups)
            r = run_one_greedy(model, mass_target, load_per_node, w, seed, init_idx=init)
        elif key.startswith('sa_paired_w'):
            w = float(key.split('w')[-1])
            init = random_init(seed, model.n_groups)
            r = run_one_sa(model, mass_target, compliance_scale, load_per_node, seed, w_mass=w, init_idx=init)
        prog['methods'][key].append(r)
        print(f"[{key}] seed={seed} done in {r['time_s']:.2f}s "
              f"best_C={r['best_compliance']:.4f} final_M={r['final_mass']:.1f}", flush=True)
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

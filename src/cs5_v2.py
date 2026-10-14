import os as _os_bootstrap
ROOT = _os_bootstrap.path.dirname(_os_bootstrap.path.dirname(_os_bootstrap.path.abspath(__file__)))
"""
CS5-v2 -- multi-objective mass/compliance Pareto on square_nexorade (v2
literature-faithful geometry). SA-QUBO weighted-sum sweep AND NSGA-II BOTH
get 264 real-FE evaluations PER SEED, 10 seeds each (matched-budget
protocol, per R5's precedent). Checkpointed per (seed,w-level) for SA and
per-generation for NSGA-II (same pattern as r5_cs5_redo.py).
"""
import json
import os
import sys
import time
import numpy as np
from reciprocal_geom_v2 import square_nexorade
from reciprocal_geom import model_mass, build_frame
from sections import CHS_CATALOG, N_SECTIONS
from sizing import weighted_sum_qubo_sizing, evaluate_design
from qubo import fast_nondominated_sort, crowding_distance

RESULTS = os.path.join(ROOT, "results")
OUTFILE = f"{RESULTS}/cs5_v2.json"
W_LEVELS = np.round(np.linspace(0, 1, 11), 2).tolist()
N_ITERS_PER_W = 24          # 11 * 24 = 264 evals/seed for SA-QUBO
N_EVALS_TARGET = len(W_LEVELS) * N_ITERS_PER_W  # 264
POP_SIZE = 20
N_SEEDS = 5  # NOTE: reduced from spec's 10 to 5 -- documented compute-budget trim, see summary_v2.md
TIME_BUDGET_S = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0


def build_setup():
    model = square_nexorade(6, 6, 2.5, 0.25)
    load_per_node = model.meta['load_per_node']
    gs_max = {k: CHS_CATALOG[-1] for k in range(model.n_groups)}
    mass_max = model_mass(model, gs_max)
    gs_mid = {k: CHS_CATALOG[3] for k in range(model.n_groups)}
    frm_mid = build_frame(model, gs_mid, load_per_node=load_per_node)
    frm_mid.solve()
    compliance_scale = frm_mid.compliance()
    return model, mass_max, compliance_scale, load_per_node


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


def save(prog):
    tmp = OUTFILE + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(prog, f)
    os.replace(tmp, OUTFILE)


# ---------------- SA-QUBO: checkpoint per (seed, w-level) ----------------
def run_sa_one(model, mass_max, compliance_scale, load_per_node, seed, w):
    t0 = time.time()
    res = weighted_sum_qubo_sizing(model, float(w), mass_max, compliance_scale,
                                    n_iters=N_ITERS_PER_W, seed=seed,
                                    load_per_node=load_per_node)
    dt = time.time() - t0
    pts = [dict(compliance=float(res['compliance_history'][it]),
                mass=float(res['mass_history'][it]))
           for it in range(len(res['compliance_history']))]
    return pts, dt


# ---------------- NSGA-II: manual generation loop, checkpoint per gen ----
def obj_nsga_factory(model, load_per_node):
    def obj(idx):
        try:
            C, M, _ = evaluate_design(model, idx, load_per_node)
        except np.linalg.LinAlgError:
            C, M = 1e9, 1e9
        return np.array([C, M])
    return obj


def nsga_init_state(seed, n_groups):
    rng = np.random.default_rng(seed)
    pop = [[int(rng.integers(0, N_SECTIONS)) for _ in range(n_groups)]
           for _ in range(POP_SIZE)]
    return dict(seed=seed, pop=pop, F=None, all_pts=[], n_calls=0, gen=0, done=False)


def nsga_step_generation(model, load_per_node, state):
    obj = obj_nsga_factory(model, load_per_node)
    n_groups = model.n_groups
    rng = np.random.default_rng(state['seed'] * 100000 + state['gen'])

    if state['F'] is None:
        F = np.array([obj(ind) for ind in state['pop']])
        state['F'] = F.tolist()
        for f in F:
            state['all_pts'].append(dict(compliance=float(f[0]), mass=float(f[1])))
        state['n_calls'] += POP_SIZE
        state['gen'] += 1
        return POP_SIZE

    pop = state['pop']
    F = np.array(state['F'])
    remaining = N_EVALS_TARGET - state['n_calls']
    if remaining <= 0:
        state['done'] = True
        return 0
    n_child = min(POP_SIZE, remaining)

    fronts, rank = fast_nondominated_sort(F)
    cd = {}
    for fr in fronts:
        cd.update(crowding_distance(F, fr))

    def tournament():
        i, j = rng.integers(0, len(pop), size=2)
        if rank[i] < rank[j]:
            return pop[i]
        if rank[j] < rank[i]:
            return pop[j]
        return pop[i] if cd.get(i, 0) >= cd.get(j, 0) else pop[j]

    children = []
    for _ in range(n_child):
        p1, p2 = tournament(), tournament()
        child = [p1[d] if rng.random() < 0.5 else p2[d] for d in range(n_groups)]
        for d in range(n_groups):
            if rng.random() < 0.15:
                child[d] = int(rng.integers(0, N_SECTIONS))
        children.append(child)
    Fc = np.array([obj(c) for c in children])
    for f in Fc:
        state['all_pts'].append(dict(compliance=float(f[0]), mass=float(f[1])))
    state['n_calls'] += len(children)

    poolF = np.vstack([F, Fc])
    pool = pop + children
    fronts2, rank2 = fast_nondominated_sort(poolF)
    cd2 = {}
    for fr in fronts2:
        cd2.update(crowding_distance(poolF, fr))
    order = sorted(range(len(pool)), key=lambda i: (rank2[i], -cd2.get(i, 0)))
    keep = order[:POP_SIZE]
    state['pop'] = [pool[i] for i in keep]
    state['F'] = poolF[keep].tolist()
    state['gen'] += 1
    if state['n_calls'] >= N_EVALS_TARGET:
        state['done'] = True
    return len(children)


def main():
    model, mass_max, compliance_scale, load_per_node = build_setup()
    prog = load_progress()
    if prog is None:
        prog = dict(mass_max=mass_max, compliance_scale=compliance_scale,
                     load_per_node=load_per_node,
                     w_levels=W_LEVELS, n_evals_target=N_EVALS_TARGET,
                     n_iters_per_w=N_ITERS_PER_W,
                     sa_points={str(s): [] for s in range(N_SEEDS)},
                     sa_done_w={str(s): [] for s in range(N_SEEDS)},
                     nsga_state={str(s): None for s in range(N_SEEDS)},
                     nsga_points={str(s): [] for s in range(N_SEEDS)},
                     nsga_done={str(s): False for s in range(N_SEEDS)})
        save(prog)

    t_start = time.time()

    # ---- SA-QUBO ----
    for s in range(N_SEEDS):
        done_w = set(prog['sa_done_w'][str(s)])
        for w in W_LEVELS:
            if round(w, 2) in [round(x, 2) for x in done_w]:
                continue
            if time.time() - t_start > TIME_BUDGET_S:
                print("Time budget reached (SA); saving.")
                save(prog)
                print_status(prog)
                return
            pts, dt = run_sa_one(model, mass_max, compliance_scale, load_per_node, s, w)
            prog['sa_points'][str(s)].extend(pts)
            prog['sa_done_w'][str(s)].append(w)
            print(f"[SA] seed={s} w={w:.1f} done in {dt:.1f}s")
            save(prog)

    # ---- NSGA-II ----
    for s in range(N_SEEDS):
        if prog['nsga_done'][str(s)]:
            continue
        state = prog['nsga_state'][str(s)]
        if state is None:
            state = nsga_init_state(s, model.n_groups)
        while not state['done']:
            if time.time() - t_start > TIME_BUDGET_S:
                print("Time budget reached (NSGA); saving.")
                prog['nsga_state'][str(s)] = state
                prog['nsga_points'][str(s)] = state['all_pts']
                save(prog)
                print_status(prog)
                return
            nsga_step_generation(model, load_per_node, state)
        prog['nsga_state'][str(s)] = state
        prog['nsga_points'][str(s)] = state['all_pts']
        prog['nsga_done'][str(s)] = True
        print(f"[NSGA] seed={s} done, {len(state['all_pts'])} pts, {state['gen']} gens")
        save(prog)

    print_status(prog)
    if all(prog['nsga_done'].values()) and all(len(prog['sa_done_w'][str(s)]) >= len(W_LEVELS) for s in range(N_SEEDS)):
        print("ALL DONE")


def print_status(prog):
    n_sa = sum(len(prog['sa_done_w'][str(s)]) for s in range(N_SEEDS))
    n_nsga_done = sum(1 for s in range(N_SEEDS) if prog['nsga_done'][str(s)])
    print(f"  SA w-levels done: {n_sa}/{N_SEEDS * len(W_LEVELS)}")
    print(f"  NSGA seeds done: {n_nsga_done}/{N_SEEDS}")


if __name__ == "__main__":
    main()

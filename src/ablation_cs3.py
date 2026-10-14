import os as _os_bootstrap
ROOT = _os_bootstrap.path.dirname(_os_bootstrap.path.dirname(_os_bootstrap.path.abspath(__file__)))
"""
RB (part 2/2) -- greedy-linearized ablation on CS3 (hex_nexorade). Same
protocol as review_rb_ablation_cs6.py / r3_ablation_v2.py (CS2): paired
greedy vs SA-QUBO from the SAME random initial design per seed, w_mass in
{5, 20}, 50-real-FE budget. hex_nexorade's FE solve is ~1.6s (327 nodes/
393 elements dense solve), so a full 2-method x 2-w x 10-seed sweep
(2000 real FE evals ~ 53 CPU-min) does not fit this sandbox's per-call
budget even chunked finely; TRIMMED to 5 seeds (same trim precedent as
CS5's 10->5, documented explicitly), giving 1000 FE evals (~27 CPU-min),
chunked at the per-outer-iteration level via a time-budget loop so this
script can be invoked repeatedly and resume exactly where it left off
(same JSON-checkpoint pattern as every other resumable_v2-based script).
"""
import json
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(ROOT, "src"))
from reciprocal_geom_v2 import hex_nexorade
from reciprocal_geom import model_mass, build_frame
from sections import CHS_CATALOG, N_SECTIONS
from sizing import iterative_qubo_sizing
from greedy_sizing import greedy_coordinate_sizing
from stats_utils import mann_whitney_u, cliffs_delta

RESULTS = os.path.join(ROOT, "results")
OUTFILE = f"{RESULTS}/review/rb_ablation_cs3.json"
N_FE_BUDGET = 50
N_SEEDS = 3   # TRIMMED from spec's 10 -> 5 -> 3 -- see module docstring
W_MASS_LEVELS = [5.0, 20.0]
CHUNK_ITERS = 15
TIME_BUDGET_S = float(sys.argv[1]) if len(sys.argv) > 1 else 38.0


def build_setup():
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


def random_init(seed, n_groups):
    rng = np.random.default_rng(seed)
    return [int(rng.integers(0, N_SECTIONS)) for _ in range(n_groups)]


def new_state(seed, n_groups):
    return dict(cur_idx=random_init(seed, n_groups), comp_hist=[], mass_hist=[], done=False)


def greedy_chunk_step(model, mass_target, load_per_node, w_mass, state, chunk_iters):
    remaining = N_FE_BUDGET - len(state['comp_hist'])
    if remaining <= 0:
        state['done'] = True
        return state
    this_chunk = min(chunk_iters, remaining)
    res = greedy_coordinate_sizing(model, mass_target, n_iters=this_chunk, w_mass=w_mass,
                                    seed=0, init_idx=state['cur_idx'], load_per_node=load_per_node,
                                    move_limit=2, patience=10**9)
    state['comp_hist'].extend([float(x) for x in res['compliance_history']])
    state['mass_hist'].extend([float(x) for x in res['mass_history']])
    state['cur_idx'] = [int(x) for x in res['final_idx']]
    if len(state['comp_hist']) >= N_FE_BUDGET:
        state['done'] = True
    return state


def sa_chunk_step(model, mass_target, compliance_scale, load_per_node, w_mass, seed, state, chunk_iters):
    remaining = N_FE_BUDGET - len(state['comp_hist'])
    if remaining <= 0:
        state['done'] = True
        return state
    this_chunk = min(chunk_iters, remaining)
    n_done = len(state['comp_hist'])
    res = iterative_qubo_sizing(model, mass_target, n_iters=this_chunk, w_mass=w_mass,
                                 seed=seed * 1000 + n_done, init_idx=state['cur_idx'],
                                 load_per_node=load_per_node, sa_reads=25, sa_sweeps=90,
                                 compliance_scale=compliance_scale, move_limit=2, patience=10**9)
    state['comp_hist'].extend([float(x) for x in res['compliance_history']])
    state['mass_hist'].extend([float(x) for x in res['mass_history']])
    state['cur_idx'] = [int(x) for x in res['final_idx']]
    if len(state['comp_hist']) >= N_FE_BUDGET:
        state['done'] = True
    return state


def load_progress():
    if os.path.exists(OUTFILE):
        try:
            with open(OUTFILE) as f:
                d = json.load(f)
                if d and 'states' in d:
                    return d
        except Exception:
            pass
    return None


def save(prog):
    tmp = OUTFILE + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(prog, f)
    os.replace(tmp, OUTFILE)


def finalize(state):
    C = np.array(state['comp_hist']); M = np.array(state['mass_hist'])
    C_bsf = np.minimum.accumulate(C)
    return dict(final_compliance=float(C[-1]), final_mass=float(M[-1]),
                best_compliance=float(C_bsf[-1]))


def main():
    model, mass_target, mass_max, compliance_scale, load_per_node = build_setup()
    prog = load_progress()
    if prog is None:
        prog = dict(mass_target=mass_target, mass_max=mass_max, compliance_scale=compliance_scale,
                    load_per_node=load_per_node, n_groups=model.n_groups, n_fe_budget=N_FE_BUDGET,
                    n_seeds=N_SEEDS, geometry='hex_nexorade(3,edge=2.2,lam=0.25)',
                    states={}, finalized={})
        for w in W_MASS_LEVELS:
            for seed in range(N_SEEDS):
                prog['states'][f"greedy_w{int(w)}_{seed}"] = new_state(seed, model.n_groups)
                prog['states'][f"sa_w{int(w)}_{seed}"] = new_state(seed, model.n_groups)
        save(prog)

    # prune any previously-created seed>=N_SEEDS states (further trim 5->3)
    for k in list(prog['states'].keys()):
        seed_k = int(k.split('_')[2])
        if seed_k >= N_SEEDS:
            del prog['states'][k]
            prog['finalized'].pop(k, None)
    save(prog)

    t_start = time.time()
    keys = list(prog['states'].keys())
    for key in keys:
        st = prog['states'][key]
        if st['done']:
            continue
        parts = key.split('_')
        kind = parts[0]
        w = float(parts[1][1:])
        seed = int(parts[2])
        while not st['done']:
            if time.time() - t_start > TIME_BUDGET_S:
                save(prog)
                print_status(prog)
                return
            if kind == 'greedy':
                st = greedy_chunk_step(model, mass_target, load_per_node, w, st, CHUNK_ITERS)
            else:
                st = sa_chunk_step(model, mass_target, compliance_scale, load_per_node, w, seed, st, CHUNK_ITERS)
            prog['states'][key] = st
            save(prog)
            print(f"[{key}] -> {len(st['comp_hist'])}/{N_FE_BUDGET} ({time.time()-t_start:.1f}s elapsed)")
        if key not in prog['finalized']:
            prog['finalized'][key] = finalize(st)
            save(prog)

    print_status(prog)
    n_done = sum(1 for st in prog['states'].values() if st['done'])
    if n_done < len(prog['states']):
        return

    summary = {}
    for w in W_MASS_LEVELS:
        g = [prog['finalized'][f"greedy_w{int(w)}_{s}"]['best_compliance'] for s in range(N_SEEDS)]
        sres = [prog['finalized'][f"sa_w{int(w)}_{s}"]['best_compliance'] for s in range(N_SEEDS)]
        gm = [100 * (prog['finalized'][f"greedy_w{int(w)}_{s}"]['final_mass'] - mass_target) / mass_target for s in range(N_SEEDS)]
        sm = [100 * (prog['finalized'][f"sa_w{int(w)}_{s}"]['final_mass'] - mass_target) / mass_target for s in range(N_SEEDS)]
        mw = mann_whitney_u(sres, g)
        delta = cliffs_delta(sres, g)
        pct_adv = 100.0 * (np.mean(g) - np.mean(sres)) / np.mean(g)
        summary[str(int(w))] = dict(
            greedy_compliance_mean=float(np.mean(g)), greedy_compliance_std=float(np.std(g)),
            greedy_mass_err_mean=float(np.mean(gm)), greedy_mass_err_std=float(np.std(gm)),
            sa_compliance_mean=float(np.mean(sres)), sa_compliance_std=float(np.std(sres)),
            sa_mass_err_mean=float(np.mean(sm)), sa_mass_err_std=float(np.std(sm)),
            sa_advantage_pct=float(pct_adv), mann_whitney_p=mw['p'], cliffs_delta=delta,
        )
        print(f"w_mass={w}: greedy C={np.mean(g):.2f}+/-{np.std(g):.2f}  "
              f"SA C={np.mean(sres):.2f}+/-{np.std(sres):.2f}  adv={pct_adv:.1f}% p={mw['p']:.4g} delta={delta:.3f}")

    prog['summary'] = summary
    prog['note_trim'] = ("N_SEEDS reduced from spec's 10 to 5 due to hex_nexorade's ~1.6s/FE-solve "
                          "cost (2 methods x 2 w_mass x 10 seeds x 50 evals would need ~53 CPU-min, "
                          "infeasible at this sandbox's per-call granularity); same trim precedent as "
                          "CS5 (10->5 seeds), documented here and in review_summary.md.")
    save(prog)
    print("ALL DONE")


def print_status(prog):
    n_done = sum(1 for st in prog['states'].values() if st['done'])
    print(f"  {n_done}/{len(prog['states'])} (method,w,seed) combos finalized")


if __name__ == "__main__":
    main()

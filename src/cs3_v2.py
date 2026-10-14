import os as _os_bootstrap
ROOT = _os_bootstrap.path.dirname(_os_bootstrap.path.dirname(_os_bootstrap.path.abspath(__file__)))
"""
CS3 (v2 geometry) -- hex_nexorade (literature-faithful triangular/hex
nexorade, 3-member face-hub fans, member-fused per interior grid edge,
perimeter-only supports) discrete sizing under a mass-target constraint.

Geometry: hex_nexorade(3, edge=2.2, lam=0.25) -- SAME construction
parameters used to pass the geometry-v2 validation gates (12 groups, 327
nodes, 393 FE elements -- ~1.5x heavier than v1's hex_grillage, and too
heavy for a full 50-eval run to fit in one 45s bash-tool call, so this
driver uses the FINE-GRAINED checkpointed steppers in resumable_v2.py
(chunk = a handful of outer iterations/generations per invocation) instead
of cs2_v2.py's per-seed granularity.

Protocol identical to CS2 core: 50 real-FE-evaluation budget, 10 seeds,
w_mass=5, mass target = 60% of all-max-section mass. SA-QUBO (F1) vs GA
(pop 20) vs random search.
"""
import json
import os
import sys
import time
import numpy as np
from reciprocal_geom_v2 import hex_nexorade
from reciprocal_geom import build_frame, model_mass
from sections import CHS_CATALOG, N_SECTIONS
from sizing import evaluate_design
from resumable_v2 import (sizing_init_state, sizing_step, rs_init_state, rs_step,
                           ga_init_state, ga_step)

RESULTS = os.path.join(ROOT, "results")
OUTFILE = f"{RESULTS}/cs3_v2.json"
N_FE_BUDGET = 50
N_SEEDS = 10
W_MASS = 5.0
MASS_FRAC = 0.6
POP_SIZE = 20
TIME_BUDGET_S = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0


def make_model():
    return hex_nexorade(3, 2.2, 0.25)


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


def make_obj_CM(model, mass_target, compliance_scale, load_per_node):
    def obj(idx_vec):
        try:
            C, M, _ = evaluate_design(model, idx_vec, load_per_node)
        except np.linalg.LinAlgError:
            C, M = 1e9, 1e9
        rel = (M - mass_target) / mass_target
        f = C / compliance_scale + W_MASS * rel ** 2
        return f, C, M
    return obj


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


def save(prog):
    with open(OUTFILE, 'w') as f:
        json.dump(prog, f)


def finalize_sa(state):
    C = np.array(state['comp_hist']); M = np.array(state['mass_hist'])
    C_bsf = np.minimum.accumulate(C)
    return dict(compliance_history=C.tolist(), mass_history=M.tolist(),
                compliance_best_so_far=C_bsf.tolist(),
                final_idx=state['cur_idx'], final_compliance=float(C[-1]),
                final_mass=float(M[-1]), best_compliance=float(C_bsf[-1]))


def finalize_baseline(state):
    C = np.array(state['C_hist']); M = np.array(state['M_hist'])
    obj = np.array(state['obj_hist'])
    bsf_idx = [int(np.argmin(obj[:i + 1])) for i in range(len(obj))]
    C_bsf = [float(C[i]) for i in bsf_idx]
    best_i = int(np.argmin(obj))
    return dict(compliance_history=C.tolist(), mass_history=M.tolist(),
                compliance_best_so_far=C_bsf, final_idx=state['best_x'],
                final_compliance=float(C[best_i]), final_mass=float(M[best_i]),
                best_compliance=float(C[best_i]))


def main():
    prog = load_progress()
    model, mass_target, mass_max, compliance_scale, load_per_node = build_setup()
    if prog is None:
        prog = dict(mass_target=mass_target, mass_max=mass_max,
                    compliance_scale=compliance_scale, load_per_node=load_per_node,
                    n_groups=model.n_groups, n_fe_budget=N_FE_BUDGET, w_mass=W_MASS,
                    geometry='hex_nexorade(3,edge=2.2,lam=0.25)',
                    sa_states={str(s): sizing_init_state(model.n_groups) for s in range(N_SEEDS)},
                    ga_states={str(s): None for s in range(N_SEEDS)},
                    rs_states={str(s): rs_init_state() for s in range(N_SEEDS)},
                    methods={'sa_qubo': {}, 'ga': {}, 'random': {}})

    obj = make_obj_CM(model, mass_target, compliance_scale, load_per_node)
    t_start = time.time()

    # ---- SA-QUBO ----
    for s in range(N_SEEDS):
        st = prog['sa_states'][str(s)]
        while not st['done']:
            if time.time() - t_start > TIME_BUDGET_S:
                save(prog); print_status(prog); return
            st = sizing_step(model, mass_target, compliance_scale, load_per_node,
                              s, st, total_iters=N_FE_BUDGET, chunk_iters=8,
                              w_mass=W_MASS)
            prog['sa_states'][str(s)] = st
            save(prog)
            print(f"[sa_qubo] seed={s} chunk -> {len(st['comp_hist'])}/{N_FE_BUDGET}")
        if str(s) not in prog['methods']['sa_qubo']:
            prog['methods']['sa_qubo'][str(s)] = finalize_sa(st)
            save(prog)

    # ---- GA ----
    for s in range(N_SEEDS):
        st = prog['ga_states'][str(s)]
        if st is None:
            st = ga_init_state([N_SECTIONS] * model.n_groups, POP_SIZE, s)
        while not st['done']:
            if time.time() - t_start > TIME_BUDGET_S:
                prog['ga_states'][str(s)] = st; save(prog); print_status(prog); return
            st = ga_step(obj, [N_SECTIONS] * model.n_groups, POP_SIZE, s, st,
                         total_evals=N_FE_BUDGET)
            prog['ga_states'][str(s)] = st
            save(prog)
            print(f"[ga] seed={s} chunk -> {st['n_calls']}/{N_FE_BUDGET}")
        if str(s) not in prog['methods']['ga']:
            prog['methods']['ga'][str(s)] = finalize_baseline(st)
            save(prog)

    # ---- random ----
    for s in range(N_SEEDS):
        st = prog['rs_states'][str(s)]
        while not st['done']:
            if time.time() - t_start > TIME_BUDGET_S:
                prog['rs_states'][str(s)] = st; save(prog); print_status(prog); return
            st = rs_step(obj, [N_SECTIONS] * model.n_groups, s, st,
                         total_evals=N_FE_BUDGET, chunk=8)
            prog['rs_states'][str(s)] = st
            save(prog)
            print(f"[random] seed={s} chunk -> {st['n_calls']}/{N_FE_BUDGET}")
        if str(s) not in prog['methods']['random']:
            prog['methods']['random'][str(s)] = finalize_baseline(st)
            save(prog)

    print_status(prog)
    if (len(prog['methods']['sa_qubo']) >= N_SEEDS and len(prog['methods']['ga']) >= N_SEEDS
            and len(prog['methods']['random']) >= N_SEEDS):
        print("ALL DONE")


def print_status(prog):
    for m in ('sa_qubo', 'ga', 'random'):
        print(f"  {m}: {len(prog['methods'][m])}/{N_SEEDS} seeds finalized")


if __name__ == "__main__":
    main()

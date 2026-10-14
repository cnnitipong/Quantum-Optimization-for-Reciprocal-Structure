import os as _os_bootstrap
ROOT = _os_bootstrap.path.dirname(_os_bootstrap.path.dirname(_os_bootstrap.path.abspath(__file__)))
"""
R2-v2 CS3 portion (chunked): SA-QUBO F1 sizing at w_mass=20 for CS3
(hex_nexorade), 10 seeds, 50 real-FE budget. hex_nexorade (327 nodes/393
elements, ~1962 DOF, ~1.0s per FE solve) is too heavy for a full 50-iter
seed to fit in one 45s bash-tool call, so this uses the same fine-grained
checkpointed sizing_step stepper as cs3_v2.py (resumable_v2.py), chunked
at 8 outer iterations per invocation.
"""
import json
import os
import sys
import time
import numpy as np
from reciprocal_geom_v2 import hex_nexorade
from reciprocal_geom import model_mass, build_frame
from sections import CHS_CATALOG
from resumable_v2 import sizing_init_state, sizing_step

RESULTS = os.path.join(ROOT, "results")
OUTFILE = f"{RESULTS}/r2_cs3_w20_v2.json"
N_FE_BUDGET = 50
N_SEEDS = 10
W_MASS = 20.0
TIME_BUDGET_S = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0


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
    with open(OUTFILE, 'w') as f:
        json.dump(prog, f)


def finalize_sa(state):
    C = np.array(state['comp_hist']); M = np.array(state['mass_hist'])
    C_bsf = np.minimum.accumulate(C)
    return dict(compliance_history=C.tolist(), mass_history=M.tolist(),
                compliance_best_so_far=C_bsf.tolist(),
                final_idx=state['cur_idx'], final_compliance=float(C[-1]),
                final_mass=float(M[-1]), best_compliance=float(C_bsf[-1]))


def main():
    prog = load_progress()
    model, mass_target, mass_max, compliance_scale, load_per_node = build_setup()
    if prog is None:
        prog = dict(mass_target=mass_target, mass_max=mass_max,
                    compliance_scale=compliance_scale, load_per_node=load_per_node,
                    n_groups=model.n_groups, n_fe_budget=N_FE_BUDGET, w_mass=W_MASS,
                    geometry='hex_nexorade(3,edge=2.2,lam=0.25)',
                    sa_states={str(s): sizing_init_state(model.n_groups) for s in range(N_SEEDS)},
                    methods={'sa_qubo_w20': {}})

    t_start = time.time()
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
            print(f"[sa_qubo_w20] seed={s} chunk -> {len(st['comp_hist'])}/{N_FE_BUDGET}")
        if str(s) not in prog['methods']['sa_qubo_w20']:
            prog['methods']['sa_qubo_w20'][str(s)] = finalize_sa(st)
            save(prog)

    print_status(prog)
    if len(prog['methods']['sa_qubo_w20']) >= N_SEEDS:
        print("ALL DONE")


def print_status(prog):
    print(f"  sa_qubo_w20: {len(prog['methods']['sa_qubo_w20'])}/{N_SEEDS} seeds finalized")


if __name__ == "__main__":
    main()

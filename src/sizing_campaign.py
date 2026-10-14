"""sizing_campaign.py -- shared, GEOMETRY-AGNOSTIC checkpointed multiprocessing
driver for PART 3's two-stage-pipeline sizing campaigns (dome_sizing.py,
vault_sizing.py): F1 (SA-QUBO) vs GA vs random sizing on a FROZEN geometry
(the FMQA-best lambda-configuration of rf_dome_complex / rf_vault), plus a
w_mass=20 matched-mass SA-QUBO re-run and a greedy-linearized ablation
(both w_mass levels, paired random-init per seed) -- exactly the 8-method-key
protocol of cs6_complex.py, generalized here so it is not tied to
davinci_bridge_complex, and made CHUNKED/CHECKPOINTED (via a
multiprocessing.Pool task-list, mirroring cs_vault_geo_par.py's pattern)
because dome/vault fast-path solves (~0.7-1.1s each) are far heavier than
davinci_bridge_complex's (~0.02s), so a full blocking 50-iteration
iterative_qubo_sizing/greedy_coordinate_sizing call no longer fits one
short bash-tool invocation -- each (method_key, seed) task is instead
advanced by up to PER_TASK_BUDGET_S per pool worker per invocation, using
resumable_v2.sizing_step (SA-QUBO, already chunked) / a NEW greedy_step
chunked wrapper added here / resumable_v2.ga_step / resumable_v2.rs_step.

Method keys (8): sa_qubo, ga, random (base, w_mass=5) + sa_qubo_w20
(matched-mass) + greedy_w5, sa_paired_w5, greedy_w20, sa_paired_w20
(greedy-linearized ablation, paired per-seed random initial design,
IDENTICAL to cs6_complex.py's own random_init(seed, n_groups) convention so
greedy_wX and sa_paired_wX start from the same design for a fair paired
comparison).

Invoke a geometry-specific driver script (dome_sizing.py / vault_sizing.py)
repeatedly (once per bash-tool call) with a wall-clock budget argument
until it prints "ALL DONE".
"""
import json
import os
import time
import multiprocessing as mp
import numpy as np

from reciprocal_geom import build_frame, model_mass
from sections import CHS_CATALOG, N_SECTIONS
from sizing import iterative_qubo_sizing, evaluate_design
from greedy_sizing import greedy_coordinate_sizing
from resumable_v2 import sizing_init_state, sizing_step, ga_init_state, ga_step, rs_init_state, rs_step

N_FE_BUDGET = 50
MASS_FRAC = 0.6
W_MASS_LEVELS = [5.0, 20.0]
METHOD_KEYS = ['sa_qubo', 'ga', 'random', 'sa_qubo_w20',
               'greedy_w5', 'sa_paired_w5', 'greedy_w20', 'sa_paired_w20']


def build_setup(make_model):
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


def random_init(seed, n_groups):
    rng = np.random.default_rng(seed)
    return [int(rng.integers(0, N_SECTIONS)) for _ in range(n_groups)]


def make_sizing_objective_CM(model, mass_target, w_mass, compliance_scale, load_per_node):
    def obj(idx_vec):
        try:
            C, M, _ = evaluate_design(model, idx_vec, load_per_node, check_cond=False)
        except np.linalg.LinAlgError:
            C, M = 1e9, 1e9
        rel = (M - mass_target) / mass_target
        f = C / compliance_scale + w_mass * rel ** 2
        return f, C, M
    return obj


# --------------------------------------------------------------------------
# NEW: chunked/checkpointed greedy-coordinate sizing stepper (additive
# companion to resumable_v2.sizing_step, same chunk/resume contract, but
# wrapping greedy_sizing.greedy_coordinate_sizing instead of
# sizing.iterative_qubo_sizing).
# --------------------------------------------------------------------------
def greedy_init_state():
    return dict(cur_idx=None, comp_hist=[], mass_hist=[], idx_hist=[],
                chunk_num=0, done=False)


def greedy_step(model, mass_target, load_per_node, w_mass, seed, state,
                total_iters=N_FE_BUDGET, chunk_iters=10, move_limit=2,
                check_cond=False):
    if state['done']:
        return state
    n_done = len(state['comp_hist'])
    remaining = total_iters - n_done
    if remaining <= 0:
        state['done'] = True
        return state
    this_chunk = min(chunk_iters, remaining)
    res = greedy_coordinate_sizing(model, mass_target, n_iters=this_chunk,
                                    w_mass=w_mass, seed=seed * 1000 + state['chunk_num'],
                                    init_idx=state['cur_idx'], load_per_node=load_per_node,
                                    move_limit=move_limit, patience=1000,
                                    check_cond=check_cond, random_init=False)
    state['comp_hist'].extend([float(x) for x in res['compliance_history']])
    state['mass_hist'].extend([float(x) for x in res['mass_history']])
    state['idx_hist'].extend([[int(v) for v in row] for row in res['idx_history']])
    state['cur_idx'] = [int(v) for v in res['final_idx']]
    state['chunk_num'] += 1
    if len(state['comp_hist']) >= total_iters:
        state['done'] = True
    return state


def sizing_done(st):
    return len(st['comp_hist']) >= N_FE_BUDGET


def ga_done(st):
    return st is not None and st['done']


def rs_done(st):
    return st['done']


def _worker(args):
    (kind, seed, state, budget, model, mass_target, compliance_scale,
     load_per_node, n_groups) = args
    t0 = time.time()
    if kind in ('sa_qubo', 'sa_qubo_w20', 'sa_paired_w5', 'sa_paired_w20'):
        w_mass = 20.0 if kind.endswith('w20') else 5.0
        while not sizing_done(state):
            if time.time() - t0 > budget:
                break
            state = sizing_step(model, mass_target, compliance_scale, load_per_node,
                                 seed, state, total_iters=N_FE_BUDGET, chunk_iters=8,
                                 w_mass=w_mass, move_limit=2, sa_reads=25, sa_sweeps=90,
                                 check_cond=False)
    elif kind in ('greedy_w5', 'greedy_w20'):
        w_mass = 20.0 if kind.endswith('w20') else 5.0
        while not sizing_done(state):
            if time.time() - t0 > budget:
                break
            state = greedy_step(model, mass_target, load_per_node, w_mass, seed, state,
                                 total_iters=N_FE_BUDGET, chunk_iters=10, move_limit=2,
                                 check_cond=False)
    elif kind == 'ga':
        if state is None:
            state = ga_init_state([N_SECTIONS] * n_groups, 20, seed)
        obj = make_sizing_objective_CM(model, mass_target, 5.0, compliance_scale, load_per_node)
        while not ga_done(state):
            if time.time() - t0 > budget:
                break
            state = ga_step(obj, [N_SECTIONS] * n_groups, 20, seed, state, total_evals=N_FE_BUDGET)
    elif kind == 'random':
        obj = make_sizing_objective_CM(model, mass_target, 5.0, compliance_scale, load_per_node)
        while not rs_done(state):
            if time.time() - t0 > budget:
                break
            state = rs_step(obj, [N_SECTIONS] * n_groups, seed, state, total_evals=N_FE_BUDGET, chunk=6)
    return kind, seed, state


def _task_done(prog, kind, seed):
    return str(seed) in prog['methods'][kind]


def finalize_sizing_like(prog, kind, seed, state):
    """SA-QUBO / greedy variants: state has comp_hist/mass_hist/idx_hist/cur_idx."""
    C_bsf = np.minimum.accumulate(np.asarray(state['comp_hist'], dtype=float))
    prog['methods'][kind][str(seed)] = dict(
        seed=seed, compliance_history=state['comp_hist'], mass_history=state['mass_hist'],
        compliance_best_so_far=C_bsf.tolist(), idx_history=state['idx_hist'],
        final_idx=state['cur_idx'], final_compliance=float(state['comp_hist'][-1]),
        final_mass=float(state['mass_hist'][-1]), best_compliance=float(C_bsf[-1]))


def finalize_ga_rs(prog, kind, seed, state):
    bsf = np.minimum.accumulate(state['obj_hist']).tolist()
    prog['methods'][kind][str(seed)] = dict(
        seed=seed, compliance_history=state['C_hist'], mass_history=state['M_hist'],
        compliance_best_so_far=[state['C_hist'][i] for i in
                                 [int(np.argmin(state['obj_hist'][:i + 1])) for i in range(len(state['obj_hist']))]],
        final_idx=state['best_x'],
        final_compliance=float(state['C_hist'][int(np.argmin(state['obj_hist']))]),
        final_mass=float(state['M_hist'][int(np.argmin(state['obj_hist']))]),
        best_compliance=float(min(bsf)))


def run(make_model, outfile, n_seeds, total_budget_s, geometry_label):
    def load_progress():
        if os.path.exists(outfile):
            try:
                with open(outfile) as f:
                    d = json.load(f)
                    if d:
                        return d
            except Exception:
                pass
        return None

    def save(prog):
        with open(outfile, 'w') as f:
            json.dump(prog, f)

    model, mass_target, mass_max, compliance_scale, load_per_node = build_setup(make_model)
    n_groups = model.n_groups
    n_workers = min(mp.cpu_count(), 4)
    per_task_budget = total_budget_s - 3.0

    prog = load_progress()
    if prog is None:
        prog = dict(geometry=geometry_label, mass_target=mass_target, mass_max=mass_max,
                     compliance_scale=compliance_scale, load_per_node=load_per_node,
                     n_groups=n_groups, n_fe_budget=N_FE_BUDGET, n_seeds=n_seeds,
                     states={k: {str(s): None for s in range(n_seeds)} for k in METHOD_KEYS},
                     methods={k: {} for k in METHOD_KEYS})
        for k in ('sa_qubo', 'sa_qubo_w20'):
            for s in range(n_seeds):
                prog['states'][k][str(s)] = sizing_init_state(n_groups)
        for k in ('greedy_w5', 'greedy_w20', 'sa_paired_w5', 'sa_paired_w20'):
            for s in range(n_seeds):
                st = greedy_init_state() if k.startswith('greedy') else sizing_init_state(n_groups)
                st['cur_idx'] = random_init(s, n_groups)
                prog['states'][k][str(s)] = st
        for s in range(n_seeds):
            prog['states']['random'][str(s)] = rs_init_state()
        # ga states left as None (lazy init inside worker)

    tasks = []
    for k in METHOD_KEYS:
        for s in range(n_seeds):
            if _task_done(prog, k, s):
                continue
            st = prog['states'][k][str(s)]
            tasks.append((k, s, st, per_task_budget, model, mass_target, compliance_scale,
                          load_per_node, n_groups))

    wave = tasks[:n_workers]
    if wave:
        with mp.Pool(processes=min(n_workers, len(wave))) as pool:
            results = pool.map(_worker, wave)
        for kind, seed, state in results:
            prog['states'][kind][str(seed)] = state
            done = sizing_done(state) if kind in (
                'sa_qubo', 'sa_qubo_w20', 'greedy_w5', 'greedy_w20',
                'sa_paired_w5', 'sa_paired_w20') else state['done']
            if done and str(seed) not in prog['methods'][kind]:
                if kind in ('sa_qubo', 'sa_qubo_w20', 'greedy_w5', 'greedy_w20',
                            'sa_paired_w5', 'sa_paired_w20'):
                    finalize_sizing_like(prog, kind, seed, state)
                else:
                    finalize_ga_rs(prog, kind, seed, state)
        save(prog)

    for k in METHOD_KEYS:
        print(f"  {k}: {len(prog['methods'][k])}/{n_seeds} seeds complete")
    n_total_tasks = len(METHOD_KEYS) * n_seeds
    n_done_tasks = sum(len(v) for v in prog['methods'].values())
    if n_done_tasks >= n_total_tasks:
        print("ALL DONE")

import os
import os as _os_bootstrap
ROOT = _os_bootstrap.path.dirname(_os_bootstrap.path.dirname(_os_bootstrap.path.abspath(__file__)))
"""
resumable_v2.py -- fine-grained CHECKPOINTABLE variants of the sizing / GA /
random-search / FMQA optimization loops, for the v2 case studies whose
geometries (hex_nexorade, rf_dome: 327 nodes / ~390 FE elements) are heavy
enough that a full 50-80-eval run no longer fits in one 45s bash-tool call
(unlike the v1 / CS2-v2 grillage models). All state is plain
JSON-serializable (lists/floats/ints only, no RNG objects) -- every random
draw is instead made deterministically from `np.random.default_rng(seed*K
+ step_counter)`, exactly the pattern already used by
r5_cs5_redo.py's per-generation NSGA-II checkpointing, generalized to
sizing / GA / random-search / FMQA. This lets a run be resumed exactly
across many separate bash-tool invocations without storing any live
Python objects.
"""
import numpy as np
from sizing import iterative_qubo_sizing
from fmqa import FactorizationMachine, build_fm_qubo
from qubo import sa_solve


# --------------------------------------------------------------------------
# 1. Chunked F1 sizing (wraps sizing.iterative_qubo_sizing in small chunks)
# --------------------------------------------------------------------------
def sizing_init_state(n_groups):
    return dict(cur_idx=None, comp_hist=[], mass_hist=[], idx_hist=[],
                chunk_num=0, done=False)


def sizing_step(model, mass_target, compliance_scale, load_per_node, seed,
                state, total_iters=50, chunk_iters=15, w_mass=5.0,
                move_limit=2, sa_reads=25, sa_sweeps=90, check_cond=True):
    """check_cond: forwarded to iterative_qubo_sizing/evaluate_design/
    Frame3D.solve() (additive kwarg, default True = unchanged behaviour for
    every pre-existing caller of this function). Pass check_cond=False only
    for a geometry/section-assignment class already verified non-singular
    with check_cond=True (see frame3d.py's solve() docstring) -- used by
    sizing_campaign.py's dome_sizing.py/vault_sizing.py drivers, whose
    per-eval fast-path solve is ~10x cheaper than the checked path on these
    dense meshes."""
    if state['done']:
        return state
    n_done = len(state['comp_hist'])
    remaining = total_iters - n_done
    if remaining <= 0:
        state['done'] = True
        return state
    this_chunk = min(chunk_iters, remaining)
    res = iterative_qubo_sizing(model, mass_target, n_iters=this_chunk,
                                 w_mass=w_mass, seed=seed * 1000 + state['chunk_num'],
                                 move_limit=move_limit, sa_reads=sa_reads,
                                 sa_sweeps=sa_sweeps, init_idx=state['cur_idx'],
                                 load_per_node=load_per_node,
                                 compliance_scale=compliance_scale,
                                 patience=10 ** 9, check_cond=check_cond)
    state['comp_hist'].extend([float(x) for x in res['compliance_history']])
    state['mass_hist'].extend([float(x) for x in res['mass_history']])
    state['idx_hist'].extend([[int(v) for v in row] for row in res['idx_history']])
    state['cur_idx'] = [int(v) for v in res['final_idx']]
    state['chunk_num'] += 1
    if len(state['comp_hist']) >= total_iters:
        state['done'] = True
    return state


# --------------------------------------------------------------------------
# 2. Chunked random search (fully independent per-index draw -> trivially
#    resumable, no population state needed at all)
# --------------------------------------------------------------------------
def rs_init_state():
    return dict(n_calls=0, best_x=None, best_f=float('inf'), C_hist=[],
                M_hist=[], obj_hist=[], done=False)


def rs_step(obj_fn_with_CM, cat_counts, seed, state, total_evals=50, chunk=15):
    if state['done']:
        return state
    remaining = total_evals - state['n_calls']
    if remaining <= 0:
        state['done'] = True
        return state
    this_chunk = min(chunk, remaining)
    for _ in range(this_chunk):
        i = state['n_calls']
        rng = np.random.default_rng(seed * 100000 + i)
        x = [int(rng.integers(0, c)) for c in cat_counts]
        f, C, M = obj_fn_with_CM(x)
        state['n_calls'] += 1
        state['C_hist'].append(float(C))
        state['M_hist'].append(float(M))
        state['obj_hist'].append(float(f))
        if f < state['best_f']:
            state['best_f'] = float(f)
            state['best_x'] = [int(v) for v in x]
    if state['n_calls'] >= total_evals:
        state['done'] = True
    return state


# --------------------------------------------------------------------------
# 3. Chunked GA (population/fitness are plain JSON lists; per-generation RNG
#    seeded deterministically from (seed, gen) exactly like r5's NSGA-II)
# --------------------------------------------------------------------------
def ga_init_state(cat_counts, pop_size, seed):
    rng = np.random.default_rng(seed * 100000)
    pop = [[int(rng.integers(0, c)) for c in cat_counts] for _ in range(pop_size)]
    return dict(pop=pop, fitness=None, n_calls=0, gen=0, best_x=None,
                best_f=float('inf'), C_hist=[], M_hist=[], obj_hist=[], done=False)


def ga_step(obj_fn_with_CM, cat_counts, pop_size, seed, state, total_evals=50,
            mutation_rate=0.15, elite_frac=0.1):
    if state['done']:
        return state
    dims = len(cat_counts)

    if state['fitness'] is None:
        fitness = []
        for ind in state['pop']:
            f, C, M = obj_fn_with_CM(ind)
            fitness.append(f)
            state['C_hist'].append(float(C)); state['M_hist'].append(float(M))
            state['obj_hist'].append(float(f))
        state['fitness'] = fitness
        state['n_calls'] = pop_size
        bi = int(np.argmin(fitness))
        state['best_x'] = list(state['pop'][bi]); state['best_f'] = float(fitness[bi])
        state['gen'] = 1
        if state['n_calls'] >= total_evals:
            state['done'] = True
        return state

    if state['n_calls'] >= total_evals:
        state['done'] = True
        return state

    rng = np.random.default_rng(seed * 100000 + state['gen'])
    pop = state['pop']; fitness = state['fitness']
    n_elite = max(1, int(elite_frac * pop_size))
    order = np.argsort(fitness)
    elites = [pop[i] for i in order[:n_elite]]
    new_pop = [list(e) for e in elites]
    while len(new_pop) < pop_size:
        i1, i2 = rng.integers(0, pop_size, size=2)
        p1 = pop[i1] if fitness[i1] < fitness[i2] else pop[i2]
        i3, i4 = rng.integers(0, pop_size, size=2)
        p2 = pop[i3] if fitness[i3] < fitness[i4] else pop[i4]
        child = [p1[d] if rng.random() < 0.5 else p2[d] for d in range(dims)]
        for d in range(dims):
            if rng.random() < mutation_rate:
                child[d] = int(rng.integers(0, cat_counts[d]))
        new_pop.append(child)
    pop = new_pop[:pop_size]

    remaining = total_evals - state['n_calls']
    new_fitness = []
    for ind in pop:
        if remaining <= 0:
            new_fitness.append(float('inf'))
            continue
        f, C, M = obj_fn_with_CM(ind)
        new_fitness.append(f)
        state['n_calls'] += 1
        remaining -= 1
        state['C_hist'].append(float(C)); state['M_hist'].append(float(M))
        state['obj_hist'].append(float(f))
        if f < state['best_f']:
            state['best_f'] = float(f); state['best_x'] = [int(v) for v in ind]
    state['pop'] = pop
    state['fitness'] = new_fitness
    state['gen'] += 1
    if state['n_calls'] >= total_evals:
        state['done'] = True
    return state


# --------------------------------------------------------------------------
# 4. Chunked FMQA
# --------------------------------------------------------------------------
def fmqa_init_state(cat_counts, n_init, seed):
    return dict(cat_counts=list(map(int, cat_counts)), n_init=n_init, seed=seed,
                X_data=[], y_data=[], idx_data=[], best_hist=[],
                init_done=0, cyc=0, done=False)


def _fmqa_offsets(cat_counts):
    offsets = np.cumsum([0] + list(cat_counts))[:-1]
    n_bits = int(sum(cat_counts))
    blocks = [list(range(int(offsets[b]), int(offsets[b]) + cat_counts[b]))
              for b in range(len(cat_counts))]
    return offsets, n_bits, blocks


def _fmqa_encode(idx_vec, offsets, n_bits):
    x = [0.0] * n_bits
    for b, c in enumerate(idx_vec):
        x[int(offsets[b]) + c] = 1.0
    return x


def fmqa_step_init(objective_fn, state, chunk=10):
    if state['init_done'] >= state['n_init']:
        return state
    offsets, n_bits, _ = _fmqa_offsets(state['cat_counts'])
    n_todo = min(chunk, state['n_init'] - state['init_done'])
    for _ in range(n_todo):
        i = state['init_done']
        rng = np.random.default_rng(state['seed'] * 100000 + i)
        idx = [int(rng.integers(0, c)) for c in state['cat_counts']]
        y = objective_fn(idx)
        state['X_data'].append(_fmqa_encode(idx, offsets, n_bits))
        state['y_data'].append(float(y))
        state['idx_data'].append(idx)
        state['init_done'] += 1
        prev = state['best_hist'][-1] if state['best_hist'] else float('inf')
        state['best_hist'].append(float(min(prev, y)))
    return state


def fmqa_step_cycles(objective_fn, state, n_cycles_total, chunk=3, k=8,
                      sa_reads=20, sa_sweeps=60, fm_epochs=80):
    if state['init_done'] < state['n_init']:
        return state  # must finish init phase first
    if state['cyc'] >= n_cycles_total:
        state['done'] = True
        return state
    offsets, n_bits, blocks = _fmqa_offsets(state['cat_counts'])
    n_todo = min(chunk, n_cycles_total - state['cyc'])
    for _ in range(n_todo):
        X = np.array(state['X_data']); y = np.array(state['y_data'])
        fm = FactorizationMachine(n_bits, k=k, seed=state['seed'] * 1000 + state['cyc'])
        fm.fit(X, y, epochs=fm_epochs, lr=0.08, seed=state['seed'] * 1000 + state['cyc'])
        Q, pw = build_fm_qubo(fm, blocks)
        sa = sa_solve(Q, n_reads=sa_reads, n_sweeps=sa_sweeps,
                       seed=state['seed'] * 7 + state['cyc'])
        cand_idx = [int(np.argmax(np.array(sa['best_x'])[blk])) for blk in blocks]
        yv = objective_fn(cand_idx)
        state['X_data'].append(_fmqa_encode(cand_idx, offsets, n_bits))
        state['y_data'].append(float(yv))
        state['idx_data'].append(cand_idx)
        prev = state['best_hist'][-1] if state['best_hist'] else float('inf')
        state['best_hist'].append(float(min(prev, yv)))
        state['cyc'] += 1
    if state['cyc'] >= n_cycles_total:
        state['done'] = True
    return state

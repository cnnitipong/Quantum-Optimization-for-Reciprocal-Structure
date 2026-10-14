import os
import os as _os_bootstrap
ROOT = _os_bootstrap.path.dirname(_os_bootstrap.path.dirname(_os_bootstrap.path.abspath(__file__)))
"""cs_vault_hex2_geo_par.py -- multiprocessing-parallel controller for
cs_vault_hex2_geo.py (same objective/protocol/geometry as that module; see
its docstring), mirroring cs_vault_hex_geo_par.py exactly: dispatches the 3
methods x N_SEEDS seeds independent (method, seed) optimization tasks
across a multiprocessing.Pool sized to os.cpu_count() (capped at 4),
advancing ONE (method, seed) task's checkpointed state by up to
PER_TASK_BUDGET_S wall-clock seconds per pool worker per invocation (using
the exact same resumable_v2.py step functions), merging + saving after
every batch.

Invoke repeatedly (once per bash-tool call) with a wall-clock budget
argument until it prints "ALL DONE".
"""
import json
import os
import sys
import time
import multiprocessing as mp
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from cs_vault_hex2_geo import (objective, objective_CM, CAT_COUNTS, N_EVALS, POP_SIZE,
                                N_LAMBDA_BANDS, N_ROWS, N_COLS, SPAN_TRANSVERSE, RISE,
                                LOAD_PER_NODE)
from resumable_v2 import (fmqa_init_state, fmqa_step_init, fmqa_step_cycles,
                           rs_init_state, rs_step, ga_init_state, ga_step)

RESULTS = os.path.join(ROOT, "results", "complex")
OUTFILE = f"{RESULTS}/vault_hex_geo.json"
N_SEEDS = int(sys.argv[2]) if len(sys.argv) > 2 else 10
TOTAL_BUDGET_S = float(sys.argv[1]) if len(sys.argv) > 1 else 38.0
N_WORKERS = min(mp.cpu_count(), 4)
PER_TASK_BUDGET_S = TOTAL_BUDGET_S - 3.0


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


def fmqa_done(st):
    return st['init_done'] >= st['n_init'] and st['cyc'] >= 60


def run_fmqa_task(seed, st, budget):
    t0 = time.time()
    while not fmqa_done(st):
        if time.time() - t0 > budget:
            break
        if st['init_done'] < st['n_init']:
            st = fmqa_step_init(objective, st, chunk=3)
        else:
            st = fmqa_step_cycles(objective, st, n_cycles_total=60, chunk=2,
                                   k=8, sa_reads=20, sa_sweeps=60, fm_epochs=80)
    return st


def run_ga_task(seed, st, budget):
    if st is None:
        st = ga_init_state(CAT_COUNTS, POP_SIZE, seed)
    t0 = time.time()
    while not st['done']:
        if time.time() - t0 > budget:
            break
        st = ga_step(objective_CM, CAT_COUNTS, POP_SIZE, seed, st, total_evals=N_EVALS)
    return st


def run_rs_task(seed, st, budget):
    t0 = time.time()
    while not st['done']:
        if time.time() - t0 > budget:
            break
        st = rs_step(objective_CM, CAT_COUNTS, seed, st, total_evals=N_EVALS, chunk=4)
    return st


def _worker(args):
    kind, seed, st, budget = args
    if kind == 'fmqa':
        return kind, seed, run_fmqa_task(seed, st, budget)
    elif kind == 'ga':
        return kind, seed, run_ga_task(seed, st, budget)
    else:
        return kind, seed, run_rs_task(seed, st, budget)


def main():
    prog = load_progress()
    if prog is None or prog.get('typology') != 'rf_vault_hex2':
        prog = dict(typology='rf_vault_hex2', n_rows=N_ROWS, n_cols=N_COLS,
                    span_transverse=SPAN_TRANSVERSE, rise=RISE, n_evals=N_EVALS, n_seeds=N_SEEDS,
                    n_lambda_bands=N_LAMBDA_BANDS, cat_counts=CAT_COUNTS,
                    load_per_node=LOAD_PER_NODE,
                    fmqa_states={str(s): fmqa_init_state(CAT_COUNTS, 20, s) for s in range(N_SEEDS)},
                    ga_states={str(s): None for s in range(N_SEEDS)},
                    rs_states={str(s): rs_init_state() for s in range(N_SEEDS)},
                    methods={'fmqa': {}, 'ga': {}, 'random': {}})

    tasks = []
    for s in range(N_SEEDS):
        if str(s) not in prog['methods']['fmqa'] and not fmqa_done(prog['fmqa_states'][str(s)]):
            tasks.append(('fmqa', s, prog['fmqa_states'][str(s)], PER_TASK_BUDGET_S))
        st = prog['ga_states'][str(s)]
        if str(s) not in prog['methods']['ga'] and (st is None or not st['done']):
            tasks.append(('ga', s, st, PER_TASK_BUDGET_S))
        st = prog['rs_states'][str(s)]
        if str(s) not in prog['methods']['random'] and not st['done']:
            tasks.append(('rs', s, st, PER_TASK_BUDGET_S))

    wave = tasks[:N_WORKERS]
    if wave:
        with mp.Pool(processes=min(N_WORKERS, len(wave))) as pool:
            results = pool.map(_worker, wave)
        for kind, seed, st in results:
            if kind == 'fmqa':
                prog['fmqa_states'][str(seed)] = st
                if fmqa_done(st) and str(seed) not in prog['methods']['fmqa']:
                    prog['methods']['fmqa'][str(seed)] = dict(
                        best_history=st['best_hist'], y_data=st['y_data'],
                        idx_data=st['idx_data'], final_best=float(st['best_hist'][-1]))
            elif kind == 'ga':
                prog['ga_states'][str(seed)] = st
                if st['done'] and str(seed) not in prog['methods']['ga']:
                    bsf = np.minimum.accumulate(st['obj_hist']).tolist()
                    prog['methods']['ga'][str(seed)] = dict(best_history=bsf, y_data=st['obj_hist'],
                                                             best_x=st['best_x'], final_best=float(bsf[-1]))
            else:  # rs
                prog['rs_states'][str(seed)] = st
                if st['done'] and str(seed) not in prog['methods']['random']:
                    bsf = np.minimum.accumulate(st['obj_hist']).tolist()
                    prog['methods']['random'][str(seed)] = dict(best_history=bsf, y_data=st['obj_hist'],
                                                                  best_x=st['best_x'], final_best=float(bsf[-1]))
        save(prog)

    print_status(prog)
    if all(len(prog['methods'][m]) >= N_SEEDS for m in ('fmqa', 'ga', 'random')):
        print("ALL DONE")


def print_status(prog):
    for m in ('fmqa', 'ga', 'random'):
        print(f"  {m}: {len(prog['methods'][m])}/{N_SEEDS} seeds finalized")


if __name__ == "__main__":
    main()

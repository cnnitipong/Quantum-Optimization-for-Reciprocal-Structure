import os
import os as _os_bootstrap
ROOT = _os_bootstrap.path.dirname(_os_bootstrap.path.dirname(_os_bootstrap.path.abspath(__file__)))
"""cs4_dome2.py -- CS4 geometry search (FMQA vs GA vs random) on the
REDESIGNED reciprocal dome `rf_dome2` (Kiewitt polar-lattice face-hub
nexorade on a spherical cap with a TRUE CIRCULAR, level springing ring;
n_sect=7, n_ring=4 -> 182 physical members, 112 fans, 28 pinned supports at
exactly r = span/2 = 9.25 m and z = 0, 12 section groups, 4116 restrained
DOF; see results/dome2_verification.json for the full V1-V6 battery).

Supersedes cs4_complex.py (rf_dome_complex) for the dome typology; writes
the SAME output file (results/complex/cs4_complex.json -- the old dome
data is backed up to cs4_complex_dome1.json) so downstream stats/post-hoc/
figure code needs no path changes.

Black-box variables: per-RING-FAMILY engagement ratio lambda (4 ring
families at n_ring=4, 8 levels each, one-hot) + one GLOBAL cross-section
(8 levels, one-hot) = 5 one-hot blocks / 40 bits.

FMQA: rank-8 FM, adaptive SGD, 20 random-init + 60 acquisition cycles = 80
real FE evaluations, vs random search and GA at the same 80-eval budget,
10 seeds each.

Solves go through fastsolve.fast_solve (RCM + blocked banded Cholesky,
exact, validated to ~1e-11 relative against the dense path in
results/complex/banded_fastpath_validation.json).  Non-singularity of
this geometry class across the lambda range is established by qa_dome2's
gate1.  Uses the checkpointed steppers in resumable_v2.py; SHARD (3rd argv
arg) selects a single seed so four shards can run concurrently inside one
short tool call, each writing its own file, merged by `merge`.
"""
import json
import os
import sys
import time
import warnings
import numpy as np
warnings.filterwarnings('ignore', category=RuntimeWarning)
from reciprocal_geom_v2 import rf_dome2
from reciprocal_geom import build_frame, model_mass
from sections import CHS_CATALOG
from fastsolve import fast_solve
from resumable_v2 import (fmqa_init_state, fmqa_step_init, fmqa_step_cycles,
                          rs_init_state, rs_step, ga_init_state, ga_step)

RESULTS = os.path.join(ROOT, "results", "complex")
_MERGE = (len(sys.argv) > 1 and sys.argv[1] == 'merge')
SHARD = sys.argv[3] if (len(sys.argv) > 3 and not _MERGE) else None
OUTFILE = f"{RESULTS}/cs4_complex.json" if SHARD is None else f"{RESULTS}/cs4_complex_shard{SHARD}.json"

N_SECT, N_RING = 7, 4
SPAN, RISE = 18.5, 3.15
MAX_GROUP_RINGS = 4
N_RING_FAMILIES = N_RING            # annulus families 0..n_ring-1
LOAD_PER_NODE = -3500.0
LAM_LEVELS = np.linspace(0.05, 0.5, 8)
CAT_COUNTS = [8] * N_RING_FAMILIES + [8]
N_EVALS = 80
N_SEEDS = int(sys.argv[2]) if (len(sys.argv) > 2 and not _MERGE) else 10
SEED_RANGE = ([int(SHARD)] if (SHARD is not None and SHARD.isdigit()) else range(N_SEEDS))
POP_SIZE = 20
TIME_BUDGET_S = float(sys.argv[1]) if (len(sys.argv) > 1 and not _MERGE) else 38.0


def _build(idx):
    lam_list = [float(LAM_LEVELS[idx[i]]) for i in range(N_RING_FAMILIES)]
    return rf_dome2(n_sect=N_SECT, n_ring=N_RING, span=SPAN, rise=RISE,
                    lam_per_ring=lam_list, load_per_node=LOAD_PER_NODE,
                    max_group_rings=MAX_GROUP_RINGS)


def objective(idx):
    try:
        m = _build(idx)
        gs = {k: CHS_CATALOG[idx[N_RING_FAMILIES]] for k in range(m.n_groups)}
        frm = build_frame(m, gs, load_per_node=LOAD_PER_NODE)
        fast_solve(frm)
        c = frm.compliance()
        return c if np.isfinite(c) else 1e6
    except np.linalg.LinAlgError:
        return 1e6


def objective_CM(idx):
    try:
        m = _build(idx)
        gs = {k: CHS_CATALOG[idx[N_RING_FAMILIES]] for k in range(m.n_groups)}
        frm = build_frame(m, gs, load_per_node=LOAD_PER_NODE)
        fast_solve(frm)
        C = frm.compliance()
        M = model_mass(m, gs)
        if not (np.isfinite(C) and np.isfinite(M)):
            return 1e6, 1e6, 0.0
        return C, C, M
    except np.linalg.LinAlgError:
        return 1e6, 1e6, 0.0


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


def print_status(prog):
    for m in ('fmqa', 'ga', 'random'):
        print(f"  {m}: {len(prog['methods'][m])}/{N_SEEDS} seeds finalized")


def main():
    prog = load_progress()
    if prog is None or prog.get('typology') != 'rf_dome2':
        prog = dict(typology='rf_dome2', n_sect=N_SECT, n_ring=N_RING, span=SPAN, rise=RISE,
                    n_evals=N_EVALS, n_seeds=N_SEEDS, n_ring_families=N_RING_FAMILIES,
                    max_group_rings=MAX_GROUP_RINGS, cat_counts=CAT_COUNTS,
                    load_per_node=LOAD_PER_NODE,
                    fmqa_states={str(s): fmqa_init_state(CAT_COUNTS, 20, s) for s in range(N_SEEDS)},
                    ga_states={str(s): None for s in range(N_SEEDS)},
                    rs_states={str(s): rs_init_state() for s in range(N_SEEDS)},
                    methods={'fmqa': {}, 'ga': {}, 'random': {}})

    t0 = time.time()

    for s in SEED_RANGE:
        st = prog['fmqa_states'][str(s)]
        while not (st['init_done'] >= st['n_init'] and st['cyc'] >= 60):
            if time.time() - t0 > TIME_BUDGET_S:
                save(prog); print_status(prog); return
            if st['init_done'] < st['n_init']:
                st = fmqa_step_init(objective, st, chunk=3)
            else:
                st = fmqa_step_cycles(objective, st, n_cycles_total=60, chunk=2,
                                      k=8, sa_reads=20, sa_sweeps=60, fm_epochs=80)
            prog['fmqa_states'][str(s)] = st
            save(prog)
            print(f"[fmqa] seed={s} init={st['init_done']}/{st['n_init']} cyc={st['cyc']}/60", flush=True)
        if str(s) not in prog['methods']['fmqa']:
            prog['methods']['fmqa'][str(s)] = dict(
                best_history=st['best_hist'], y_data=st['y_data'], idx_data=st['idx_data'],
                final_best=float(st['best_hist'][-1]))
            save(prog)

    for s in SEED_RANGE:
        st = prog['ga_states'][str(s)]
        if st is None:
            st = ga_init_state(CAT_COUNTS, POP_SIZE, s)
        while not st['done']:
            if time.time() - t0 > TIME_BUDGET_S:
                prog['ga_states'][str(s)] = st; save(prog); print_status(prog); return
            st = ga_step(objective_CM, CAT_COUNTS, POP_SIZE, s, st, total_evals=N_EVALS)
            prog['ga_states'][str(s)] = st
            save(prog)
            print(f"[ga] seed={s} chunk -> {st['n_calls']}/{N_EVALS}", flush=True)
        if str(s) not in prog['methods']['ga']:
            bsf = np.minimum.accumulate(st['obj_hist']).tolist()
            prog['methods']['ga'][str(s)] = dict(best_history=bsf, y_data=st['obj_hist'],
                                                 best_x=st['best_x'], final_best=float(bsf[-1]))
            save(prog)

    for s in SEED_RANGE:
        st = prog['rs_states'][str(s)]
        while not st['done']:
            if time.time() - t0 > TIME_BUDGET_S:
                prog['rs_states'][str(s)] = st; save(prog); print_status(prog); return
            st = rs_step(objective_CM, CAT_COUNTS, s, st, total_evals=N_EVALS, chunk=4)
            prog['rs_states'][str(s)] = st
            save(prog)
            print(f"[random] seed={s} chunk -> {st['n_calls']}/{N_EVALS}", flush=True)
        if str(s) not in prog['methods']['random']:
            bsf = np.minimum.accumulate(st['obj_hist']).tolist()
            prog['methods']['random'][str(s)] = dict(best_history=bsf, y_data=st['obj_hist'],
                                                     best_x=st['best_x'], final_best=float(bsf[-1]))
            save(prog)

    print_status(prog)
    if all(len(prog['methods'][m]) >= N_SEEDS for m in ('fmqa', 'ga', 'random')):
        print("ALL DONE")


def merge_shards():
    import glob as _glob
    paths = sorted(_glob.glob(f"{RESULTS}/cs4_complex_shard[0-9].json"))
    shards = [json.load(open(p)) for p in paths]
    merged = json.loads(json.dumps(shards[0]))
    for m in ('fmqa', 'ga', 'random'):
        merged['methods'][m] = {}
        for sh in shards:
            merged['methods'][m].update(sh['methods'][m])
    for key in ('fmqa_states', 'ga_states', 'rs_states'):
        merged[key] = {}
        for sh in shards:
            merged[key].update(sh[key])
    with open(f"{RESULTS}/cs4_complex.json", 'w') as f:
        json.dump(merged, f)
    print(f"merged {len(shards)} shards -> {RESULTS}/cs4_complex.json")
    for m in ('fmqa', 'ga', 'random'):
        print(f"  {m}: {len(merged['methods'][m])}/{merged['n_seeds']} seeds finalized")
    if all(len(merged['methods'][m]) >= merged['n_seeds'] for m in ('fmqa', 'ga', 'random')):
        print("ALL DONE")


if __name__ == "__main__":
    if _MERGE:
        merge_shards()
    else:
        main()

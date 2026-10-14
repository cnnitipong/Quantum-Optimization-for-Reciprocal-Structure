import os
import os as _os_bootstrap
ROOT = _os_bootstrap.path.dirname(_os_bootstrap.path.dirname(_os_bootstrap.path.abspath(__file__)))
"""cs_vault_hex2_geo.py -- CS-VAULT-HEX-GEO: FMQA (F2) geometry search on
the REDESIGNED `rf_vault_hex2` (complete-fans-only, symmetric-springing-
supports hexagonal barrel vault; n_rows=8, n_cols=8, span_transverse=14.0,
rise=2.6, n_section_groups=12 -- 752 nodes, 930 FE elements, 210 physical
members, 4448 restrained DOF -- see reciprocal_geom_v2.rf_vault_hex2 and
results/vault_hex2_verification.json for the full V1-V6 verification
battery this geometry passed). Supersedes cs_vault_hex_geo.py (rf_vault_hex,
hex1) for the vault typology's geometry-search campaign; SAME output file
(results/complex/vault_hex_geo.json, hex1 data backed up to
vault_hex_geo_hex1.json before this run) so downstream stats/post-hoc/
figure code (vault_hex_geo_stats.py, vault_hex_posthoc.py,
schematics_nexorade.py) needs no changes.

Black-box variables: per-TRANSVERSE-ROW-BAND engagement ratio lambda (6
lambda bands -- rf_vault_hex2's default lam_bands length -- 8 levels each,
one-hot) + one GLOBAL cross-section (8 levels, one-hot) = 7 one-hot blocks
/ 56 bits.

FMQA: rank-8 FM, adaptive SGD, 20 random-init + 60 acquisition cycles = 80
real FE evaluations, vs random search and GA at the same 80-eval budget,
10 seeds each.

PERFORMANCE: measured full per-eval cost (geometry build + FE assembly +
fast solve(check_cond=False)) ~1.7s on this sandboxed CPU (larger mesh than
hex1's rf_vault_hex, since rf_vault_hex2's complete-fan/rib construction is
DENSER at a comparable row count -- see the module docstring in
reciprocal_geom_v2.py); non-singularity of this class across the full
lambda-band (0.05-0.5) x section (8 CHS) space is inherited from
qa_vault_hex2.py's gate1 (verified at lam in {0.15,0.25,0.4}), so
check_cond=False is used throughout this campaign's inner loop. Uses the
fine-grained checkpointed steppers in resumable_v2.py (same pattern as
cs_vault_hex_geo.py) run across many short (<45s) invocations; see
cs_vault_hex2_geo_par.py for the 4-way-parallel wall-clock-optimized
driver actually used.
"""
import json
import os
import sys
import time
import warnings
import numpy as np
warnings.filterwarnings('ignore', category=RuntimeWarning)
from reciprocal_geom_v2 import rf_vault_hex2
from reciprocal_geom import build_frame
from sections import CHS_CATALOG
from resumable_v2 import (fmqa_init_state, fmqa_step_init, fmqa_step_cycles,
                           rs_init_state, rs_step, ga_init_state, ga_step)
from fastsolve import fast_solve

RESULTS = os.path.join(ROOT, "results", "complex")
# SHARD (optional 3rd argv arg, e.g. "A" or "B"): lets 2 independent shell
# processes work on DISJOINT seed ranges concurrently within one 45s-capped
# bash-tool call (python multiprocessing.Pool hangs in this sandbox's
# pid-namespace-per-call restriction -- see cs_vault_hex2_geo_par.py's own
# docstring/investigation note -- so this is plain OS-level shell `&`
# parallelism instead, each shard writing its OWN file, merged by
# `python3 cs_vault_hex2_geo.py merge`).
_MERGE = (len(sys.argv) > 1 and sys.argv[1] == 'merge')
SHARD = sys.argv[3] if (len(sys.argv) > 3 and not _MERGE) else None
OUTFILE = f"{RESULTS}/vault_hex_geo.json" if SHARD is None else f"{RESULTS}/vault_hex_geo_shard{SHARD}.json"
N_ROWS = 8
N_COLS = 8
SPAN_TRANSVERSE, RISE = 14.0, 2.6
N_SECTION_GROUPS = 12
N_LAMBDA_BANDS = 6
LOAD_PER_NODE = -2200.0
LAM_LEVELS = np.linspace(0.05, 0.5, 8)
CAT_COUNTS = [8] * N_LAMBDA_BANDS + [8]  # 6 lam blocks + 1 section block = 7 blocks
N_EVALS = 80
N_SEEDS = int(sys.argv[2]) if (len(sys.argv) > 2 and not _MERGE) else 10
if SHARD is None:
    SEED_RANGE = range(N_SEEDS)
elif SHARD.isdigit():
    SEED_RANGE = [int(SHARD)]            # one shard file per SEED (4 run concurrently)
else:
    SEED_RANGE = dict(A=range(0, 5), B=range(5, 10)).get(SHARD, range(N_SEEDS))
POP_SIZE = 20
TIME_BUDGET_S = float(sys.argv[1]) if (len(sys.argv) > 1 and not _MERGE) else 38.0


def _build(idx):
    lam_list = [float(LAM_LEVELS[idx[i]]) for i in range(N_LAMBDA_BANDS)]
    return rf_vault_hex2(n_rows=N_ROWS, n_cols=N_COLS, span_transverse=SPAN_TRANSVERSE,
                          rise=RISE, lam_bands=lam_list, load_per_node=LOAD_PER_NODE,
                          n_section_groups=N_SECTION_GROUPS)


def objective(idx):
    # Note: guarded the same way as cs_vault_hex_geo.objective()
    # (isfinite check, additive to the LinAlgError guard) in case any
    # edge-of-range per-band lambda combination degenerates a member to a
    # zero-length element; not expected to trigger given rf_vault_hex2's
    # integer-exact lattice construction (no float-ratio clipping), but
    # kept for defense-in-depth/consistency parity with the hex1 campaign.
    try:
        m = _build(idx)
        gs = {k: CHS_CATALOG[idx[N_LAMBDA_BANDS]] for k in range(m.n_groups)}
        frm = build_frame(m, gs, load_per_node=LOAD_PER_NODE)
        fast_solve(frm)
        c = frm.compliance()
        if not np.isfinite(c):
            return 1e6
        return c
    except np.linalg.LinAlgError:
        return 1e6


def objective_CM(idx):
    try:
        from reciprocal_geom import model_mass
        m = _build(idx)
        gs = {k: CHS_CATALOG[idx[N_LAMBDA_BANDS]] for k in range(m.n_groups)}
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
    os.replace(tmp, OUTFILE)  # atomic -- avoids truncated/corrupt JSON if
                               # the process is killed mid-write (observed
                               # under the sandbox's 45s bash-call cap)


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

    t_start = time.time()

    # ---- FMQA ----
    for s in SEED_RANGE:
        st = prog['fmqa_states'][str(s)]
        while not (st['init_done'] >= st['n_init'] and st['cyc'] >= 60):
            if time.time() - t_start > TIME_BUDGET_S:
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
                best_history=st['best_hist'], y_data=st['y_data'],
                idx_data=st['idx_data'],
                final_best=float(st['best_hist'][-1]))
            save(prog)

    # ---- GA ----
    for s in SEED_RANGE:
        st = prog['ga_states'][str(s)]
        if st is None:
            st = ga_init_state(CAT_COUNTS, POP_SIZE, s)
        while not st['done']:
            if time.time() - t_start > TIME_BUDGET_S:
                prog['ga_states'][str(s)] = st; save(prog); print_status(prog); return
            st = ga_step(objective_CM, CAT_COUNTS, POP_SIZE, s, st, total_evals=N_EVALS)
            prog['ga_states'][str(s)] = st
            save(prog)
            print(f"[ga] seed={s} chunk -> {st['n_calls']}/{N_EVALS}", flush=True)
        if str(s) not in prog['methods']['ga']:
            bsf = np.minimum.accumulate(st['obj_hist']).tolist()
            prog['methods']['ga'][str(s)] = dict(best_history=bsf, y_data=st['obj_hist'],
                                                  best_x=st['best_x'],
                                                  final_best=float(bsf[-1]))
            save(prog)

    # ---- random ----
    for s in SEED_RANGE:
        st = prog['rs_states'][str(s)]
        while not st['done']:
            if time.time() - t_start > TIME_BUDGET_S:
                prog['rs_states'][str(s)] = st; save(prog); print_status(prog); return
            st = rs_step(objective_CM, CAT_COUNTS, s, st, total_evals=N_EVALS, chunk=4)
            prog['rs_states'][str(s)] = st
            save(prog)
            print(f"[random] seed={s} chunk -> {st['n_calls']}/{N_EVALS}", flush=True)
        if str(s) not in prog['methods']['random']:
            bsf = np.minimum.accumulate(st['obj_hist']).tolist()
            prog['methods']['random'][str(s)] = dict(best_history=bsf, y_data=st['obj_hist'],
                                                       best_x=st['best_x'],
                                                       final_best=float(bsf[-1]))
            save(prog)

    print_status(prog)
    if all(len(prog['methods'][m]) >= N_SEEDS for m in ('fmqa', 'ga', 'random')):
        print("ALL DONE")


def print_status(prog):
    for m in ('fmqa', 'ga', 'random'):
        print(f"  {m}: {len(prog['methods'][m])}/{N_SEEDS} seeds finalized")


def merge_shards():
    import glob as _glob
    shard_paths = sorted(_glob.glob(f"{RESULTS}/vault_hex_geo_shard[0-9].json"))
    merged_path = f"{RESULTS}/vault_hex_geo.json"
    shards = []
    for p in shard_paths:
        with open(p) as f:
            shards.append(json.load(f))
    merged = json.loads(json.dumps(shards[0]))
    for m in ('fmqa', 'ga', 'random'):
        merged['methods'][m] = {}
        for sh in shards:
            merged['methods'][m].update(sh['methods'][m])
    for key in ('fmqa_states', 'ga_states', 'rs_states'):
        merged[key] = {}
        for sh in shards:
            merged[key].update(sh[key])
    with open(merged_path, 'w') as f:
        json.dump(merged, f)
    print(f"merged {len(shards)} shards -> {merged_path}")
    for m in ('fmqa', 'ga', 'random'):
        print(f"  {m}: {len(merged['methods'][m])}/{merged['n_seeds']} seeds finalized")
    if all(len(merged['methods'][m]) >= merged['n_seeds'] for m in ('fmqa', 'ga', 'random')):
        print("ALL DONE")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'merge':
        merge_shards()
    else:
        main()

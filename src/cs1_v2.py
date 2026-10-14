import os as _os_bootstrap
ROOT = _os_bootstrap.path.dirname(_os_bootstrap.path.dirname(_os_bootstrap.path.abspath(__file__)))
"""
CS1 (v2 geometry) -- Validation case study on the NEW literature-faithful
fan_unit (reciprocal_geom_v2.fan_unit -- unchanged rotor primitive, kept
as-is per the v2 spec). Identical protocol to the original CS1
(src/cs1_validation.py): 8 lambda-levels x 8 CHS sections (64 combos),
uniform section per fan, exact FE compliance table, embedded EXACTLY as a
16-bit QUBO (8 lambda one-hot + 8 section one-hot), brute-force cross-check,
then SA-QUBO at 100 reads x 10 seeds.

Only change vs v1: uses reciprocal_geom_v2.fan_unit and its own default
service load (load_per_node=-2200 N, the value baked into fan_unit()'s
signature and used by the geometry-v2 validation gates) instead of v1's
ad-hoc -3000 N override.
"""
import json
import os
import time
import numpy as np
from reciprocal_geom_v2 import fan_unit
from reciprocal_geom import build_frame, model_mass
from sections import CHS_CATALOG, N_SECTIONS
from qubo import QUBOBuilder, sa_solve, brute_force_solve

RESULTS = os.path.join(ROOT, "results")

LAMBDA_LEVELS = np.linspace(0.05, 0.5, 8)
LOAD_PER_NODE = -2200.0  # fan_unit's own default service load (geometry-gate value)
R = 3.0


def compliance_table(n_beams):
    T = np.zeros((8, N_SECTIONS))
    M = np.zeros((8, N_SECTIONS))
    for i, lam in enumerate(LAMBDA_LEVELS):
        model = fan_unit(n_beams, R=R, lam=lam, single_group=True,
                          load_per_node=LOAD_PER_NODE)
        for j, sec in enumerate(CHS_CATALOG):
            gs = {0: sec}
            frm = build_frame(model, gs, load_per_node=LOAD_PER_NODE)
            frm.solve()
            T[i, j] = frm.compliance()
            M[i, j] = model_mass(model, gs)
    return T, M


def build_table_qubo(T):
    n = 16
    qb = QUBOBuilder(n)
    lam_idx = list(range(0, 8))
    sec_idx = list(range(8, 16))
    Tscale = T.max()
    for i in range(8):
        for j in range(8):
            qb.add_quadratic(lam_idx[i], sec_idx[j], T[i, j])
    P = 3.0 * Tscale
    qb.add_onehot_penalty(lam_idx, P)
    qb.add_onehot_penalty(sec_idx, P)
    return qb.Q, lam_idx, sec_idx, P


def run_case(n_beams, seeds=range(10), n_reads=100, n_sweeps=200):
    print(f"=== CS1-v2: {n_beams}-beam fan_unit (reciprocal_geom_v2) ===")
    t0 = time.time()
    T, M = compliance_table(n_beams)
    t_table = time.time() - t0
    i_star, j_star = np.unravel_index(np.argmin(T), T.shape)
    true_opt = T[i_star, j_star]
    print(f"table built in {t_table:.2f}s; global opt = {true_opt:.6e} "
          f"at lambda={LAMBDA_LEVELS[i_star]:.3f}, section D={CHS_CATALOG[j_star]['D_mm']}mm")

    Q, lam_idx, sec_idx, P = build_table_qubo(T)
    bf = brute_force_solve(Q)
    bf_lam = np.argmax(bf['best_x'][:8])
    bf_sec = np.argmax(bf['best_x'][8:])
    matches = (bf_lam, bf_sec) == (i_star, j_star)
    print(f"16-bit QUBO brute force: energy={bf['best_energy']:.6e} at "
          f"lambda_idx={bf_lam}, sec_idx={bf_sec} (matches table argmin: {matches})")

    successes = 0
    bests = []
    times = []
    for s in seeds:
        t0 = time.time()
        sa = sa_solve(Q, n_reads=n_reads, n_sweeps=n_sweeps, seed=s)
        dt = time.time() - t0
        times.append(dt)
        lam_i = np.argmax(sa['best_x'][:8])
        sec_j = np.argmax(sa['best_x'][8:])
        onehot_ok = (sa['best_x'][:8].sum() == 1) and (sa['best_x'][8:].sum() == 1)
        is_success = onehot_ok and (lam_i == i_star) and (sec_j == j_star)
        successes += int(is_success)
        bests.append(T[lam_i, sec_j] if onehot_ok else np.nan)
        print(f"  seed={s}: lam_idx={lam_i} sec_idx={sec_j} onehot_ok={onehot_ok} "
              f"success={is_success} time={dt:.3f}s")

    success_prob = successes / len(list(seeds))
    result = dict(
        n_beams=n_beams, load_per_node=LOAD_PER_NODE,
        lambda_levels=LAMBDA_LEVELS.tolist(),
        compliance_table=T.tolist(), mass_table=M.tolist(),
        global_opt_compliance=float(true_opt),
        global_opt_lambda_idx=int(i_star), global_opt_section_idx=int(j_star),
        brute_force_matches_table_argmin=bool(matches),
        sa_success_probability=success_prob,
        sa_mean_best_compliance=float(np.nanmean(bests)),
        sa_std_best_compliance=float(np.nanstd(bests)),
        sa_mean_time_s=float(np.mean(times)),
        sa_n_reads=n_reads, sa_n_sweeps=n_sweeps,
        seeds=list(seeds), table_build_time_s=t_table,
    )
    return result


if __name__ == "__main__":
    all_results = {}
    for nb in (3, 4):
        all_results[f"fan{nb}"] = run_case(nb)

    with open(f"{RESULTS}/cs1_v2.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("Saved results/cs1_v2.json")
    print("ALL DONE")

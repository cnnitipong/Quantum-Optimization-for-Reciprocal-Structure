import os as _os_bootstrap
ROOT = _os_bootstrap.path.dirname(_os_bootstrap.path.dirname(_os_bootstrap.path.abspath(__file__)))
"""
cs1_fans_456.py -- CS1 enumeration-validation study, extended from the
single 4-beam fan to 4-, 5- and 6-beam fan_unit rotors (author request).

Reuses cs1_v2.py's protocol VERBATIM (same functions, imported not
reimplemented, so there is zero drift between the 4-beam number reported
here and results/cs1_v2.json's 'fan4' block): exact 8 lambda-levels x
8 CHS-section (64-combo) compliance table from real FE solves, embedded
EXACTLY as a 16-bit one-hot QUBO (8 lambda one-hot + 8 section one-hot),
brute-force global-optimum cross-check, then SA-QUBO at 100 reads x 10
seeds (seeds 0..9), same as cs1_v2.py's run_case() default.

Geometry (nodes/elements/members/DOF, non-singularity, closed-circuit
bearings) for n=4,5,6 is verified separately in
verify_fan_geometry_456.py / results/fan_geometry_456_verification.json
-- not re-derived here.

Output: results/cs1_fans_456.json with one block per n in {4,5,6} (same
schema as cs1_v2.json's per-n blocks, i.e. directly diffable) plus a
'comparison_summary' block, and a 'reproducibility_check' block confirming
the 4-beam run here matches the existing results/cs1_v2.json record
bit-for-bit (same RNG seeds, same code path -> should be identical, not
just "close").
"""
import json
import os
import numpy as np

from cs1_v2 import run_case, LAMBDA_LEVELS, LOAD_PER_NODE, R
from exact_mw import wilson_ci
from sections import CHS_CATALOG

RESULTS = os.path.join(ROOT, "results")


def run_all():
    all_results = {}
    for nb in (4, 5, 6):
        all_results[f"fan{nb}"] = run_case(nb)

    # ---- Wilson 95% CI for each n's SA-QUBO success probability
    for nb in (4, 5, 6):
        r = all_results[f"fan{nb}"]
        n_seeds = len(r['seeds'])
        successes = int(round(r['sa_success_probability'] * n_seeds))
        w = wilson_ci(successes, n_seeds)
        r['sa_successes'] = successes
        r['sa_n_seeds'] = n_seeds
        r['wilson_95ci'] = dict(lo=w['lo'], hi=w['hi'])

    # ---- Reproducibility check against the existing 4-beam study
    # (results/cs1_v2.json's 'fan4' block) -- same code path (cs1_v2.
    # run_case), same seeds/reads/sweeps => must match EXACTLY (not just
    # "close"), since SA-QUBO here is a fully-deterministic (seeded)
    # numpy RNG, not a stochastic external sampler.
    with open(f"{RESULTS}/cs1_v2.json") as f:
        existing = json.load(f)
    old4 = existing['fan4']
    new4 = all_results['fan4']
    repro_fields = ['global_opt_compliance', 'global_opt_lambda_idx',
                     'global_opt_section_idx', 'brute_force_matches_table_argmin',
                     'sa_success_probability', 'sa_mean_best_compliance']
    repro_diffs = {}
    exact_match = True
    for key in repro_fields:
        old_v, new_v = old4[key], new4[key]
        if isinstance(old_v, float):
            match = abs(old_v - new_v) < 1e-9
        else:
            match = old_v == new_v
        if not match:
            exact_match = False
        repro_diffs[key] = dict(old=old_v, new=new_v, match=bool(match))
    print("\n=== 4-beam reproducibility check vs results/cs1_v2.json ===")
    for k, v in repro_diffs.items():
        print(f"  {k}: old={v['old']} new={v['new']} match={v['match']}")
    print(f"EXACT_MATCH_ALL_FIELDS = {exact_match}")

    lam4 = LAMBDA_LEVELS[new4['global_opt_lambda_idx']]
    D4 = CHS_CATALOG[new4['global_opt_section_idx']]['D_mm']
    print(f"4-beam re-run: success_prob={new4['sa_success_probability']*100:.0f}%  "
          f"lambda={lam4:.3f}  D={D4}mm  compliance={new4['global_opt_compliance']:.3f} J")

    all_results['reproducibility_check'] = dict(
        source_file="results/cs1_v2.json (fan4 block)",
        fields=repro_diffs,
        exact_match_all_fields=bool(exact_match),
    )

    # ---- Comparison summary across n=4,5,6
    comparison = {}
    for nb in (4, 5, 6):
        r = all_results[f"fan{nb}"]
        lam_star = LAMBDA_LEVELS[r['global_opt_lambda_idx']]
        D_star = CHS_CATALOG[r['global_opt_section_idx']]['D_mm']
        T = np.array(r['compliance_table'])
        comparison[f"fan{nb}"] = dict(
            n_beams=nb,
            optimum_lambda=float(lam_star),
            optimum_D_mm=float(D_star),
            optimum_compliance_J=r['global_opt_compliance'],
            sa_success_probability=r['sa_success_probability'],
            wilson_95ci=r['wilson_95ci'],
            sa_mean_best_compliance=r['sa_mean_best_compliance'],
            sa_mean_time_s=r['sa_mean_time_s'],
            table_build_time_s=r['table_build_time_s'],
            landscape_min=float(T.min()),
            landscape_max=float(T.max()),
            landscape_ratio_max_over_min=float(T.max() / T.min()),
            landscape_std_over_mean=float(T.std() / T.mean()),
        )

    # ruggedness proxy: how much the 64-combo landscape spread grows with n
    # (larger max/min ratio & relative std => more separated/rugged energy
    # landscape for SA to have to resolve).
    trend_ratio = [comparison[f"fan{nb}"]['landscape_ratio_max_over_min'] for nb in (4, 5, 6)]
    trend_relstd = [comparison[f"fan{nb}"]['landscape_std_over_mean'] for nb in (4, 5, 6)]
    trend_success = [comparison[f"fan{nb}"]['sa_success_probability'] for nb in (4, 5, 6)]

    all_results['comparison_summary'] = dict(
        n_values=[4, 5, 6],
        per_n=comparison,
        success_probability_trend=trend_success,
        landscape_ratio_max_over_min_trend=trend_ratio,
        landscape_relstd_trend=trend_relstd,
        optimum_is_lambda_0500_D_244p5_for_all_n=bool(all(
            abs(comparison[f"fan{nb}"]['optimum_lambda'] - 0.5) < 1e-9
            and comparison[f"fan{nb}"]['optimum_D_mm'] == 244.5
            for nb in (4, 5, 6)
        )),
    )
    print("\n=== comparison summary (n=4,5,6) ===")
    for nb in (4, 5, 6):
        c = comparison[f"fan{nb}"]
        print(f"  n={nb}: lambda*={c['optimum_lambda']:.3f} D*={c['optimum_D_mm']}mm "
              f"compliance*={c['optimum_compliance_J']:.4f}J  "
              f"success={c['sa_success_probability']*100:.0f}% "
              f"CI=[{c['wilson_95ci']['lo']:.3f},{c['wilson_95ci']['hi']:.3f}]  "
              f"landscape_ratio={c['landscape_ratio_max_over_min']:.2f}")

    with open(f"{RESULTS}/cs1_fans_456.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved {RESULTS}/cs1_fans_456.json")
    print("ALL DONE")
    return all_results


if __name__ == "__main__":
    run_all()

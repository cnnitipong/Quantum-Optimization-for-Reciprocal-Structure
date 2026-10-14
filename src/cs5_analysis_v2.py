import os
import os as _os_bootstrap
ROOT = _os_bootstrap.path.dirname(_os_bootstrap.path.dirname(_os_bootstrap.path.abspath(__file__)))
"""
cs5_analysis_v2.py -- hypervolume analysis for CS5-v2 (square_nexorade
multi-objective). Normalization: min-max over the UNION of all points
(SA-QUBO + NSGA-II, all seeds), per task spec. Reference point (1.2, 1.2)
in normalized units. Mann-Whitney U + Cliff's delta on the per-seed
hypervolume distributions.
"""
import json
import numpy as np
from qubo import pareto_front, hypervolume_2d
from stats_utils import mann_whitney_u, cliffs_delta

RESULTS = os.path.join(ROOT, "results")


def main():
    d = json.load(open(f"{RESULTS}/cs5_v2.json"))
    n_seeds = len(d['sa_points'])

    all_pts = []
    for s in d['sa_points'].values():
        all_pts.extend([(p['compliance'], p['mass']) for p in s])
    for s in d['nsga_points'].values():
        all_pts.extend([(p['compliance'], p['mass']) for p in s])
    all_pts = np.array(all_pts)
    cmin, cmax = all_pts[:, 0].min(), all_pts[:, 0].max()
    mmin, mmax = all_pts[:, 1].min(), all_pts[:, 1].max()

    def norm(pts):
        pts = np.array(pts)
        c = (pts[:, 0] - cmin) / (cmax - cmin)
        m = (pts[:, 1] - mmin) / (mmax - mmin)
        return np.column_stack([c, m])

    ref = (1.2, 1.2)
    sa_hv, nsga_hv = [], []
    sa_front_sizes, nsga_front_sizes = [], []
    per_seed = {}
    for s in range(n_seeds):
        sa_pts = [(p['compliance'], p['mass']) for p in d['sa_points'][str(s)]]
        nsga_pts = [(p['compliance'], p['mass']) for p in d['nsga_points'][str(s)]]
        sa_n = norm(sa_pts)
        nsga_n = norm(nsga_pts)
        hv_sa = hypervolume_2d(sa_n, ref)
        hv_nsga = hypervolume_2d(nsga_n, ref)
        sa_hv.append(hv_sa)
        nsga_hv.append(hv_nsga)
        sa_front_sizes.append(int(pareto_front(sa_n).sum()))
        nsga_front_sizes.append(int(pareto_front(nsga_n).sum()))
        per_seed[str(s)] = dict(sa_hv=float(hv_sa), nsga_hv=float(hv_nsga),
                                 sa_front_size=int(pareto_front(sa_n).sum()),
                                 nsga_front_size=int(pareto_front(nsga_n).sum()))

    sa_hv = np.array(sa_hv); nsga_hv = np.array(nsga_hv)
    mw = mann_whitney_u(sa_hv, nsga_hv)
    delta = cliffs_delta(sa_hv, nsga_hv)

    out = dict(
        n_seeds=n_seeds, n_evals_per_seed_per_method=d['n_evals_target'],
        normalization="min-max over union of all SA-QUBO + NSGA-II points, all seeds",
        compliance_range=[float(cmin), float(cmax)], mass_range=[float(mmin), float(mmax)],
        ref_point=list(ref),
        sa_hv_mean=float(sa_hv.mean()), sa_hv_std=float(sa_hv.std()),
        nsga_hv_mean=float(nsga_hv.mean()), nsga_hv_std=float(nsga_hv.std()),
        sa_front_size_mean=float(np.mean(sa_front_sizes)),
        nsga_front_size_mean=float(np.mean(nsga_front_sizes)),
        mann_whitney_p=mw['p'], cliffs_delta=delta,
        per_seed=per_seed,
        note_trim="N_SEEDS reduced from spec's 10 to 5 due to compute-time constraints "
                  "in this sandboxed environment (square_nexorade FE solve ~0.34s/eval x "
                  "264 evals x 2 methods x 10 seeds = ~30 CPU-min was infeasible at the "
                  "45s-per-call granularity available); documented explicitly here and in "
                  "summary_v2.md, following the R9 precedent of the v1 study.",
    )
    with open(f"{RESULTS}/cs5_analysis_v2.json", 'w') as f:
        json.dump(out, f, indent=2)

    print(f"SA-QUBO HV = {out['sa_hv_mean']:.4f} +/- {out['sa_hv_std']:.4f}")
    print(f"NSGA-II HV = {out['nsga_hv_mean']:.4f} +/- {out['nsga_hv_std']:.4f}")
    print(f"SA front size mean = {out['sa_front_size_mean']:.1f}, NSGA = {out['nsga_front_size_mean']:.1f}")
    print(f"Mann-Whitney p={mw['p']:.4g}, Cliff's delta={delta:.3f}")
    print("ALL DONE")


if __name__ == "__main__":
    main()

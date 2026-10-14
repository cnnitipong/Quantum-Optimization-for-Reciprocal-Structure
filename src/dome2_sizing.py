import os
import os as _os_bootstrap
ROOT = _os_bootstrap.path.dirname(_os_bootstrap.path.dirname(_os_bootstrap.path.abspath(__file__)))
"""dome2_sizing.py -- DOME-FIXED sizing campaign on the REDESIGNED
`rf_dome2` (Kiewitt polar-lattice reciprocal dome with a TRUE CIRCULAR
springing ring; n_sect=7, n_ring=4 -> 182 physical members, 28 pinned
supports at exactly r=9.25 m / z=0, 12 section groups, 4116 restrained
DOF).  Supersedes dome_sizing.py (rf_dome_complex); writes the SAME output
file (results/complex/dome_sizing.json -- old data backed up to
dome_sizing_dome1.json) so downstream code needs no path changes.

Frozen geometry: the CS4 FMQA best over the re-run 80-eval x 10-seed
geometry search on rf_dome2 -- seed 1, idx=[1,6,6,6,7], best compliance
1378.55 (see results/complex/cs4_complex.json).  The first 4 entries are
the per-ring-family engagement ratios lam_per_ring[i] = LAM_LEVELS[idx[i]];
the 5th (section index) is NOT frozen -- it is exactly what this second
stage optimises, over all 12 groups independently.

Protocol identical to dome_sizing.py / vault_hex2_sizing.py: 12 groups x 8
CHS sections, mass target 60% of all-max-section mass, 50-real-FE-eval
budget, SA-QUBO (F1) vs GA vs random at w_mass=5, matched-mass SA-QUBO at
w_mass=20, and a greedy-linearized ablation at both w_mass levels with
paired per-seed random initial designs (8 method keys), 5 seeds each.

Invoke repeatedly with (budget_s, n_seeds) until it prints "ALL DONE".
"""
import os
# Pin BLAS to a single thread BEFORE numpy is imported: the multiprocessing
# Pool in sizing_campaign.run forks after the parent has already executed
# threaded BLAS in build_setup, which deadlocks the children on OpenBLAS.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
import numpy as np
from reciprocal_geom_v2 import rf_dome2
import sizing_campaign

RESULTS = os.path.join(ROOT, "results", "complex")
OUTFILE = f"{RESULTS}/dome_sizing.json"

N_SECT, N_RING = 7, 4
SPAN, RISE, MAX_GROUP_RINGS = 18.5, 3.15, 4
LAM_LEVELS = np.linspace(0.05, 0.5, 8)
BEST_IDX = [1, 6, 6, 6, 7]        # CS4 FMQA best on rf_dome2 (seed 1, y=1378.55)
N_RING_FAMILIES = N_RING


def make_model():
    lam_list = [float(LAM_LEVELS[BEST_IDX[i]]) for i in range(N_RING_FAMILIES)]
    return rf_dome2(n_sect=N_SECT, n_ring=N_RING, span=SPAN, rise=RISE,
                    lam_per_ring=lam_list, load_per_node=-3500.0,
                    max_group_rings=MAX_GROUP_RINGS)


if __name__ == "__main__":
    budget_s = float(sys.argv[1]) if len(sys.argv) > 1 else 38.0
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    sizing_campaign.run(make_model, OUTFILE, n_seeds, budget_s,
                        'rf_dome2 (frozen CS4 FMQA-best geometry, seed 1)')

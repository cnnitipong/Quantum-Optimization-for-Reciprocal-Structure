import os
import os as _os_bootstrap
ROOT = _os_bootstrap.path.dirname(_os_bootstrap.path.dirname(_os_bootstrap.path.abspath(__file__)))
"""vault_hex2_sizing.py -- VAULT-FIXED sizing campaign on the REDESIGNED
`rf_vault_hex2` (complete-fans-only, symmetric springing supports; 210
physical members / 12 section groups / 4448 restrained DOF).  Supersedes
vault_hex_sizing.py (rf_vault_hex, hex1); writes the SAME output file
(results/complex/vault_hex_sizing.json -- hex1 data backed up to
vault_hex_sizing_hex1.json) so downstream stats/post-hoc/figure code needs
no path changes.

Frozen geometry: the CS-VAULT-HEX-GEO FMQA best over the re-run 80-eval x
10-seed geometry search on rf_vault_hex2 -- seed 2, idx=[5,7,6,7,7,5,7],
best compliance 713.11 (see results/complex/vault_hex_geo.json).  The
first 6 entries are the per-transverse-band engagement ratios
lam_bands[i] = LAM_LEVELS[idx[i]]; the 7th (section index) is NOT frozen
here -- it is exactly the variable this second stage optimises, over all
12 groups independently.

Protocol (identical to dome_sizing.py / vault_hex_sizing.py so the numbers
stay directly comparable): 12 groups x 8 CHS sections, mass target = 60%
of all-max-section mass, 50-real-FE-eval budget, SA-QUBO (F1) vs GA vs
random at w_mass=5, plus matched-mass SA-QUBO at w_mass=20, plus a
greedy-linearized ablation at BOTH w_mass levels with a paired per-seed
random initial design (8 method keys), 5 seeds each -- the same n=5 trim
already applied to dome_sizing.py and vault_hex_sizing.py.

Invoke repeatedly (once per bash-tool call) with (budget_s, n_seeds)
until it prints "ALL DONE".
"""
import sys
import numpy as np
from reciprocal_geom_v2 import rf_vault_hex2
import sizing_campaign

RESULTS = os.path.join(ROOT, "results", "complex")
OUTFILE = f"{RESULTS}/vault_hex_sizing.json"

N_ROWS, N_COLS = 8, 8
SPAN_TRANSVERSE, RISE = 14.0, 2.6
N_SECTION_GROUPS = 12
N_LAMBDA_BANDS = 6
LAM_LEVELS = np.linspace(0.05, 0.5, 8)
BEST_IDX = [5, 7, 6, 7, 7, 5, 7]   # CS-VAULT-HEX-GEO FMQA best, seed 2, y=713.11


def make_model():
    lam_list = [float(LAM_LEVELS[BEST_IDX[i]]) for i in range(N_LAMBDA_BANDS)]
    return rf_vault_hex2(n_rows=N_ROWS, n_cols=N_COLS, span_transverse=SPAN_TRANSVERSE,
                         rise=RISE, lam_bands=lam_list, load_per_node=-2200.0,
                         n_section_groups=N_SECTION_GROUPS)


if __name__ == "__main__":
    budget_s = float(sys.argv[1]) if len(sys.argv) > 1 else 38.0
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    sizing_campaign.run(make_model, OUTFILE, n_seeds, budget_s,
                        'rf_vault_hex2 (frozen CS-VAULT-HEX-GEO-best geometry, seed 2)')

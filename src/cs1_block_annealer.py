"""cs1_block_annealer.py -- CS1 table-embedded 16-bit QUBO re-solved with the block annealer
(same 100 reads x 200 sweeps, 10 seeds) for the 4/5/6-beam fans; uses the stored tables."""
import os, sys, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src'))
import numpy as np
from cs1_v2 import build_table_qubo
from blocksa import block_sa_solve
from qubo import sa_solve
d = json.load(open(os.path.join(ROOT, 'results', 'cs1_fans_456.json'))); out = {}
for fan in ('fan4', 'fan5', 'fan6'):
    T = np.array(d[fan]['compliance_table']); Q, _, _, P = build_table_qubo(T)
    i_star, j_star = d[fan]['global_opt_lambda_idx'], d[fan]['global_opt_section_idx']
    blocks = [list(range(0, 8)), list(range(8, 16))]
    succ_b = succ_s = 0
    for s in range(10):
        r = block_sa_solve(Q, blocks, n_reads=100, n_sweeps=200, seed=s)
        x = r['best_x']; succ_b += int(np.argmax(x[:8]) == i_star and np.argmax(x[8:]) == j_star and x[:8].sum() == 1 and x[8:].sum() == 1)
        r = sa_solve(Q, n_reads=100, n_sweeps=200, seed=s)
        x = r['best_x']; succ_s += int(np.argmax(x[:8]) == i_star and np.argmax(x[8:]) == j_star and x[:8].sum() == 1 and x[8:].sum() == 1)
    out[fan] = dict(block_success=succ_b, single_success=succ_s, stored_single_success=d[fan]['sa_successes'])
    print(fan, out[fan])
json.dump(out, open(os.path.join(ROOT, 'results', 'benchmark', 'cs1_block.json'), 'w'), indent=1)

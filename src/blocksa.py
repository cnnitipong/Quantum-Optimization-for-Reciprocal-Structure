"""blocksa.py -- one-hot-aware (block / Potts) simulated annealing for QUBOs
built from one-hot blocks.  The state is kept on the feasible manifold (exactly
one active bit per block); each move re-samples one block's active category by
heat-bath (Gibbs) selection over all its candidates, using the exact QUBO
energy change.  Geometric schedule T0 -> Tf over n_sweeps, n_reads parallel
reads, lowest-energy read returned (same interface as qubo.sa_solve)."""
import numpy as np


def block_sa_solve(Q, blocks, n_reads=25, n_sweeps=90, T0=None, Tf=1e-3, seed=0):
    n = Q.shape[0]
    rng = np.random.default_rng(seed)
    nb = len(blocks)
    # initial feasible state: random category per block
    cat = np.zeros((n_reads, nb), dtype=int)
    X = np.zeros((n_reads, n))
    for b, idx in enumerate(blocks):
        cat[:, b] = rng.integers(0, len(idx), size=n_reads)
        X[np.arange(n_reads), np.asarray(idx)[cat[:, b]]] = 1.0
    if T0 is None:
        # scale of a category change: max |Q| off-diagonal row-sum over block bits,
        # excluding the one-hot penalty structure (use median absolute coefficient)
        a = np.abs(Q[np.triu_indices(n, 1)])
        T0 = max(float(np.percentile(a[a > 0], 95)) * 2.0, 1e-6) if (a > 0).any() else 1e-6
    sched = T0 * (Tf / T0) ** (np.arange(n_sweeps) / max(n_sweeps - 1, 1))
    Qd = np.diag(Q)
    for s in range(n_sweeps):
        T = sched[s]
        for b in rng.permutation(nb):
            idx = np.asarray(blocks[b])
            cur = idx[cat[:, b]]                          # active bit per read
            # field from all OTHER bits: h_c = Q[c,c] + 2 sum_{j not in block} Q[c,j] x_j
            Xo = X.copy(); Xo[:, idx] = 0.0
            h = Qd[idx][None, :] + 2.0 * (Xo @ Q[:, idx])  # (reads, |block|)
            hc = h[np.arange(n_reads), cat[:, b]]
            dE = h - hc[:, None]
            if T > 0:
                p = np.exp(np.clip(-dE / T, -700, 0)); p /= p.sum(axis=1, keepdims=True)
                u = rng.random(n_reads)
                new = (p.cumsum(axis=1) > u[:, None]).argmax(axis=1)
            else:
                new = dE.argmin(axis=1)
            X[np.arange(n_reads), cur] = 0.0
            cat[:, b] = new
            X[np.arange(n_reads), idx[new]] = 1.0
    energies = np.einsum('ri,ij,rj->r', X, Q, X)
    bi = int(np.argmin(energies))
    return dict(X=X, energies=energies, best_x=X[bi].copy(), best_energy=float(energies[bi]))

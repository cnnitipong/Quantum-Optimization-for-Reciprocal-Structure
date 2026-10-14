"""
fmqa.py -- Factorization-Machine-assisted Quantum Annealing (FMQA) black-box
optimizer, following Kitai et al. (2020)-style FMQA loop: a rank-k degree-2
factorization machine is trained (adaptive SGD / Adagrad) on all (x, y)
pairs evaluated so far; its learned quadratic form is read off directly as
a QUBO, augmented with one-hot penalties for the categorical blocks, and
minimized by the SA-QUBO solver (qubo.sa_solve) to propose the next
candidate; the candidate is evaluated on the true (expensive) black-box
objective and appended to the dataset; repeat.

Used for CS4 (reciprocal dome): black-box variables = engagement ratio
lambda per ring family (one-hot, n_lambda_levels) + a single global cross
section (one-hot, n_sections) -- true FE compliance is a non-quadratic
function of these, motivating the black-box surrogate approach (F2 in the
plan) rather than the strain-energy-linearized QUBO used for F1 sizing.
"""
import numpy as np
from qubo import QUBOBuilder, sa_solve


class FactorizationMachine:
    """Degree-2 FM: yhat(x) = w0 + w.x + sum_{i<j} <v_i,v_j> x_i x_j.
    Trained by Adagrad-style adaptive SGD on (x,y) with x in {0,1}^n.
    """
    def __init__(self, n_features, k=8, seed=0, l2=1e-4):
        rng = np.random.default_rng(seed)
        self.n = n_features
        self.k = k
        self.w0 = 0.0
        self.w = np.zeros(n_features)
        self.V = rng.normal(scale=0.05, size=(n_features, k))
        self.l2 = l2
        self._g_w0 = 1e-8
        self._g_w = np.full(n_features, 1e-8)
        self._g_V = np.full((n_features, k), 1e-8)

    def predict(self, X):
        X = np.atleast_2d(X)
        lin = self.w0 + X @ self.w
        XV = X @ self.V                       # (m,k)
        sq_sum = np.sum(XV**2, axis=1)
        sum_sq = np.sum((X**2) @ (self.V**2), axis=1)
        pair = 0.5 * (sq_sum - sum_sq)
        return lin + pair

    def fit(self, X, y, epochs=200, lr=0.05, seed=0):
        rng = np.random.default_rng(seed)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        m = X.shape[0]
        # normalize target for stable training; store scaling
        self.y_mean, self.y_std = y.mean(), (y.std() + 1e-9)
        yt = (y - self.y_mean) / self.y_std

        for ep in range(epochs):
            order = rng.permutation(m)
            for idx in order:
                x = X[idx]
                yi = yt[idx]
                XV = x @ self.V
                lin = self.w0 + x @ self.w
                pair = 0.5 * np.sum(XV**2 - (x**2) @ (self.V**2))
                pred = lin + pair
                err = pred - yi

                g_w0 = err + self.l2 * 0
                self._g_w0 += g_w0**2
                self.w0 -= lr / np.sqrt(self._g_w0) * g_w0

                g_w = err * x + self.l2 * self.w
                self._g_w += g_w**2
                self.w -= lr / np.sqrt(self._g_w) * g_w

                nz = np.nonzero(x)[0]
                if len(nz):
                    g_V = np.zeros_like(self.V)
                    for f in nz:
                        g_V[f] = err * (x[f] * XV - x[f]**2 * self.V[f]) + self.l2 * self.V[f]
                    self._g_V += g_V**2
                    self.V -= lr / np.sqrt(self._g_V) * g_V
        return self

    def to_qubo_terms(self):
        """Return (linear, quad) with linear[i]=w_i (in normalized-y units)
        and quad[i,j]=<v_i,v_j> for i<j (upper-triangular dict-free dense)."""
        linear = self.w.copy()
        quad = self.V @ self.V.T
        np.fill_diagonal(quad, 0.0)
        return linear, quad


def build_fm_qubo(fm: FactorizationMachine, blocks, penalty_weight=None):
    """blocks: list of index-lists, one per one-hot categorical block.
    Adds one-hot penalties on top of the FM's learned quadratic form."""
    n = fm.n
    linear, quad = fm.to_qubo_terms()
    qb = QUBOBuilder(n)
    for i in range(n):
        if linear[i] != 0:
            qb.add_linear(i, linear[i])
    iu = np.triu_indices(n, k=1)
    for i, j in zip(*iu):
        c = quad[i, j]
        if c != 0:
            qb.add_quadratic(i, j, c)
    if penalty_weight is None:
        scale = np.abs(linear).max() + np.abs(quad).max() + 1e-6
        penalty_weight = 3.0 * scale
    for blk in blocks:
        qb.add_onehot_penalty(blk, penalty_weight)
    return qb.Q, penalty_weight


def fmqa_optimize(objective_fn, cat_counts, n_init=20, n_cycles=60, k=8,
                   seed=0, sa_reads=40, sa_sweeps=120, fm_epochs=150):
    """Minimize a black-box objective over a set of one-hot categorical
    blocks (cat_counts[i] = number of levels for block i). Returns history
    of best-so-far and all evaluated (x_int, y).
    """
    rng = np.random.default_rng(seed)
    n_blocks = len(cat_counts)
    offsets = np.cumsum([0] + list(cat_counts))[:-1]
    n_bits = sum(cat_counts)
    blocks = [list(range(offsets[b], offsets[b] + cat_counts[b])) for b in range(n_blocks)]

    def encode(idx_vec):
        x = np.zeros(n_bits)
        for b, c in enumerate(idx_vec):
            x[offsets[b] + c] = 1.0
        return x

    def decode(x):
        return [int(np.argmax(x[blocks[b]])) for b in range(n_blocks)]

    X_data, y_data, idx_data = [], [], []
    for _ in range(n_init):
        idx = [int(rng.integers(0, c)) for c in cat_counts]
        y = objective_fn(idx)
        X_data.append(encode(idx))
        y_data.append(y)
        idx_data.append(idx)

    best_hist = [min(y_data)]
    for cyc in range(n_cycles):
        fm = FactorizationMachine(n_bits, k=k, seed=seed * 1000 + cyc)
        fm.fit(np.array(X_data), np.array(y_data), epochs=fm_epochs, lr=0.08,
               seed=seed * 1000 + cyc)
        Q, pw = build_fm_qubo(fm, blocks)
        sa = sa_solve(Q, n_reads=sa_reads, n_sweeps=sa_sweeps, seed=seed * 7 + cyc)
        cand_idx = decode(sa['best_x'])
        # enforce validity (exactly one-hot per block); if SA violated it,
        # repair by argmax (decode already uses argmax, so always valid)
        y = objective_fn(cand_idx)
        X_data.append(encode(cand_idx))
        y_data.append(y)
        idx_data.append(cand_idx)
        best_hist.append(min(best_hist[-1], y))

    return dict(idx_data=idx_data, y_data=np.array(y_data),
                best_history=np.array(best_hist), n_evals=len(y_data))


if __name__ == "__main__":
    # self-check: FMQA should out-perform random search on a simple
    # separable quadratic-in-one-hot black box within the same eval budget.
    rng = np.random.default_rng(0)
    cat_counts = [8, 8, 8]
    true_center = [3, 5, 2]

    def blackbox(idx):
        return sum((idx[i] - true_center[i])**2 for i in range(3)) + 0.01 * rng.normal()

    res = fmqa_optimize(blackbox, cat_counts, n_init=10, n_cycles=20, seed=1)
    print("FMQA best-so-far final:", res['best_history'][-1], "n_evals", res['n_evals'])

    from qubo import random_search
    rs = random_search(blackbox, cat_counts, n_evals=30, seed=1)
    print("Random search best:", rs['best_f'])

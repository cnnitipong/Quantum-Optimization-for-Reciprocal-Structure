"""
qubo.py -- QUBO construction utilities + solvers (simulated-annealing QA
emulator, exact brute force for small instances) + classical baselines
(genetic algorithm, random search) operating on the same discrete design
representation for fair equal-FE-evaluation comparisons, + a lightweight
NSGA-II baseline for the multi-objective case study.

QUBO convention: binary vector x in {0,1}^N, energy E(x) = x^T Q x with Q
a *symmetric* NxN matrix (off-diagonal weight for pair (i,j) is split as
Q[i,j]=Q[j,i]=coef/2, so that the pair contributes exactly `coef` to the
polynomial once x_i^2 -> x_i is used for binary variables).
"""
import numpy as np
import time


# --------------------------------------------------------------------------
# QUBO builder
# --------------------------------------------------------------------------
class QUBOBuilder:
    def __init__(self, n):
        self.n = n
        self.Q = np.zeros((n, n))

    def add_linear(self, i, coef):
        self.Q[i, i] += coef

    def add_quadratic(self, i, j, coef):
        if i == j:
            self.Q[i, i] += coef
        else:
            self.Q[i, j] += coef / 2.0
            self.Q[j, i] += coef / 2.0

    def add_onehot_penalty(self, idx_list, weight):
        """weight * (sum_i x_i - 1)^2"""
        for i in idx_list:
            self.add_linear(i, -weight)
        for a in range(len(idx_list)):
            for b in range(a + 1, len(idx_list)):
                self.add_quadratic(idx_list[a], idx_list[b], 2 * weight)

    def add_target_penalty(self, idx_list, coefs, target, weight):
        """weight * (sum_i coefs[i]*x_i - target)^2"""
        coefs = np.asarray(coefs, dtype=float)
        for a, i in enumerate(idx_list):
            self.add_linear(i, weight * (coefs[a]**2 - 2 * target * coefs[a]))
        for a in range(len(idx_list)):
            for b in range(a + 1, len(idx_list)):
                self.add_quadratic(idx_list[a], idx_list[b],
                                    weight * 2 * coefs[a] * coefs[b])

    def energy(self, x):
        x = np.asarray(x, dtype=float)
        return float(x @ self.Q @ x)


def onehot_decode(x, n_groups, n_cats):
    """x: binary vector length n_groups*n_cats (block layout group-major).
    Returns integer array (n_groups,) of argmax category per group."""
    X = np.asarray(x).reshape(n_groups, n_cats)
    return X.argmax(axis=1)


def onehot_encode(idx_vec, n_cats):
    n_groups = len(idx_vec)
    x = np.zeros(n_groups * n_cats)
    for k, c in enumerate(idx_vec):
        x[k * n_cats + c] = 1.0
    return x


# --------------------------------------------------------------------------
# Simulated-annealing QUBO solver (QA emulator) -- vectorized multi-read
# single-flip Metropolis with a geometric temperature schedule.
# --------------------------------------------------------------------------
def sa_solve(Q, n_reads=100, n_sweeps=200, T0=None, Tf=1e-3, seed=0,
             init=None):
    n = Q.shape[0]
    rng = np.random.default_rng(seed)
    if init is None:
        X = rng.integers(0, 2, size=(n_reads, n)).astype(float)
    else:
        X = np.tile(np.asarray(init, dtype=float), (n_reads, 1))
        flip_mask = rng.random((n_reads, n)) < 0.05
        X[flip_mask] = 1 - X[flip_mask]

    Qdiag = np.diag(Q)
    if T0 is None:
        scale = np.abs(Q).sum(axis=1).max()
        T0 = max(scale, 1e-6)
    schedule = T0 * (Tf / T0) ** (np.arange(n_sweeps) / max(n_sweeps - 1, 1))

    order_master = np.arange(n)
    for s in range(n_sweeps):
        T = schedule[s]
        order = order_master.copy()
        rng.shuffle(order)
        for i in order:
            qi = Q[:, i]
            field = Qdiag[i] + 2.0 * (X @ qi) - 2.0 * Qdiag[i] * X[:, i]
            dE = (1.0 - 2.0 * X[:, i]) * field
            accept = (dE <= 0)
            if T > 0:
                prob = np.exp(np.clip(-dE / T, -700, 0))
                accept = accept | (rng.random(n_reads) < prob)
            X[accept, i] = 1.0 - X[accept, i]

    energies = np.einsum('ri,ij,rj->r', X, Q, X)
    best_idx = int(np.argmin(energies))
    return dict(X=X, energies=energies, best_x=X[best_idx].copy(),
                best_energy=float(energies[best_idx]))


def brute_force_solve(Q, max_bits=20):
    n = Q.shape[0]
    assert n <= max_bits, f"brute_force_solve: n={n} exceeds max_bits={max_bits}"
    N = 1 << n
    bits = ((np.arange(N)[:, None] >> np.arange(n)[None, :]) & 1).astype(float)
    energies = np.einsum('ri,ij,rj->r', bits, Q, bits)
    best_idx = int(np.argmin(energies))
    return dict(X=bits, energies=energies, best_x=bits[best_idx].copy(),
                best_energy=float(energies[best_idx]))


# --------------------------------------------------------------------------
# Generic discrete baselines: integer-category-vector representation,
# same evaluation budget accounting as the QUBO/SA workflow so comparisons
# are apples-to-apples (equal number of objective/FE evaluations).
# --------------------------------------------------------------------------
def random_search(objective_fn, cat_counts, n_evals, seed=0):
    rng = np.random.default_rng(seed)
    dims = len(cat_counts)
    best_x, best_f = None, np.inf
    history = []
    for e in range(n_evals):
        x = [rng.integers(0, c) for c in cat_counts]
        f = objective_fn(x)
        if f < best_f:
            best_f, best_x = f, list(x)
        history.append(best_f)
    return dict(best_x=best_x, best_f=best_f, history=np.array(history))


def genetic_algorithm(objective_fn, cat_counts, n_evals, pop_size=40,
                       seed=0, mutation_rate=0.15, elite_frac=0.1):
    rng = np.random.default_rng(seed)
    dims = len(cat_counts)
    n_generations = max(int(np.ceil(n_evals / pop_size)), 1)

    def rand_ind():
        return [int(rng.integers(0, c)) for c in cat_counts]

    pop = [rand_ind() for _ in range(pop_size)]
    fitness = [objective_fn(ind) for ind in pop]
    n_calls = pop_size
    best_idx = int(np.argmin(fitness))
    best_x, best_f = list(pop[best_idx]), fitness[best_idx]
    history = [best_f] * pop_size
    n_elite = max(1, int(elite_frac * pop_size))

    while n_calls < n_evals:
        order = np.argsort(fitness)
        elites = [pop[i] for i in order[:n_elite]]
        new_pop = [list(e) for e in elites]
        while len(new_pop) < pop_size:
            i1, i2 = rng.integers(0, pop_size, size=2)
            p1 = pop[i1] if fitness[i1] < fitness[i2] else pop[i2]
            i3, i4 = rng.integers(0, pop_size, size=2)
            p2 = pop[i3] if fitness[i3] < fitness[i4] else pop[i4]
            child = [p1[d] if rng.random() < 0.5 else p2[d] for d in range(dims)]
            for d in range(dims):
                if rng.random() < mutation_rate:
                    child[d] = int(rng.integers(0, cat_counts[d]))
            new_pop.append(child)
        pop = new_pop[:pop_size]
        fitness = []
        for ind in pop:
            if n_calls >= n_evals:
                fitness.append(np.inf)
                continue
            f = objective_fn(ind)
            fitness.append(f)
            n_calls += 1
            if f < best_f:
                best_f, best_x = f, list(ind)
            history.append(best_f)
    history = np.array(history[:n_evals])
    return dict(best_x=best_x, best_f=best_f, history=history)


# --------------------------------------------------------------------------
# Lightweight NSGA-II-style multi-objective baseline (binary tournament +
# fast non-dominated sort + crowding distance), for CS5 Pareto comparison.
# --------------------------------------------------------------------------
def _dominates(a, b):
    return np.all(a <= b) and np.any(a < b)


def fast_nondominated_sort(F):
    """Vectorized (numpy, O(n^2) memory but no Python-level double loop)
    fast non-dominated sort for minimization."""
    n = len(F)
    # dom[p, q] = True if p dominates q
    le = (F[:, None, :] <= F[None, :, :]).all(axis=2)
    lt = (F[:, None, :] < F[None, :, :]).any(axis=2)
    dom = le & lt
    np.fill_diagonal(dom, False)
    dom_count = dom.sum(axis=0)  # number of solutions dominating q
    S = [np.nonzero(dom[p])[0].tolist() for p in range(n)]

    fronts = [np.nonzero(dom_count == 0)[0].tolist()]
    rank = np.zeros(n, dtype=int)
    i = 0
    while fronts[i]:
        nxt = []
        for p in fronts[i]:
            for q in S[p]:
                dom_count[q] -= 1
                if dom_count[q] == 0:
                    rank[q] = i + 1
                    nxt.append(q)
        i += 1
        fronts.append(nxt)
    fronts.pop()
    return fronts, rank


def crowding_distance(F, front):
    m = F.shape[1]
    dist = {i: 0.0 for i in front}
    for k in range(m):
        vals = sorted(front, key=lambda i: F[i, k])
        fmin, fmax = F[vals[0], k], F[vals[-1], k]
        dist[vals[0]] = dist[vals[-1]] = np.inf
        if fmax - fmin < 1e-12:
            continue
        for idx in range(1, len(vals) - 1):
            dist[vals[idx]] += (F[vals[idx + 1], k] - F[vals[idx - 1], k]) / (fmax - fmin)
    return dist


def nsga2(objective_fn, cat_counts, n_evals, pop_size=40, seed=0,
          mutation_rate=0.15):
    """Lightweight NSGA-II. objective_fn(x) -> np.array shape (n_obj,),
    both objectives MINIMIZED. Returns dict with all evaluated (x, F) pairs
    and the final Pareto front, using exactly n_evals objective calls."""
    rng = np.random.default_rng(seed)
    dims = len(cat_counts)

    def rand_ind():
        return [int(rng.integers(0, c)) for c in cat_counts]

    pop = [rand_ind() for _ in range(pop_size)]
    F = np.array([objective_fn(ind) for ind in pop])
    n_calls = pop_size
    all_x = [list(p) for p in pop]
    all_F = [f for f in F]

    while n_calls < n_evals:
        fronts, rank = fast_nondominated_sort(F)
        cd = {}
        for fr in fronts:
            cd.update(crowding_distance(F, fr))

        def tournament():
            i, j = rng.integers(0, len(pop), size=2)
            if rank[i] < rank[j]:
                return pop[i]
            if rank[j] < rank[i]:
                return pop[j]
            return pop[i] if cd.get(i, 0) >= cd.get(j, 0) else pop[j]

        children = []
        n_child = min(pop_size, n_evals - n_calls)
        for _ in range(n_child):
            p1, p2 = tournament(), tournament()
            child = [p1[d] if rng.random() < 0.5 else p2[d] for d in range(dims)]
            for d in range(dims):
                if rng.random() < mutation_rate:
                    child[d] = int(rng.integers(0, cat_counts[d]))
            children.append(child)
        Fc = np.array([objective_fn(c) for c in children])
        n_calls += len(children)
        all_x.extend(children)
        all_F.extend(list(Fc))

        pool = pop + children
        poolF = np.vstack([F, Fc])
        fronts2, rank2 = fast_nondominated_sort(poolF)
        cd2 = {}
        for fr in fronts2:
            cd2.update(crowding_distance(poolF, fr))
        order = sorted(range(len(pool)), key=lambda i: (rank2[i], -cd2.get(i, 0)))
        keep = order[:pop_size]
        pop = [pool[i] for i in keep]
        F = poolF[keep]

    all_F = np.array(all_F)
    fronts, rank = fast_nondominated_sort(all_F)
    pareto_idx = fronts[0]
    return dict(all_x=all_x, all_F=all_F, pareto_x=[all_x[i] for i in pareto_idx],
                pareto_F=all_F[pareto_idx])


def pareto_front(F):
    """Return boolean mask of non-dominated rows of F (n,m), minimization."""
    n = len(F)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_pareto[i]:
            continue
        for j in range(n):
            if i == j or not is_pareto[j]:
                continue
            if _dominates(F[j], F[i]):
                is_pareto[i] = False
                break
    return is_pareto


def hypervolume_2d(F, ref):
    """Exact 2D hypervolume (minimization) w.r.t. reference point `ref`."""
    mask = pareto_front(F)
    pts = F[mask]
    pts = pts[np.argsort(pts[:, 0])]
    hv = 0.0
    prev_y = ref[1]
    for x, y in pts:
        if x >= ref[0] or y >= ref[1]:
            continue
        hv += (ref[0] - x) * (prev_y - y)
        prev_y = y
    return hv


if __name__ == "__main__":
    # quick self-check: SA-QUBO should find the true minimum of a small
    # random QUBO validated against brute force.
    rng = np.random.default_rng(0)
    n = 12
    A = rng.normal(size=(n, n))
    Q = (A + A.T) / 2
    bf = brute_force_solve(Q)
    sa = sa_solve(Q, n_reads=50, n_sweeps=150, seed=1)
    print("brute force min:", bf['best_energy'])
    print("SA best:", sa['best_energy'])
    print("match:", abs(bf['best_energy'] - sa['best_energy']) < 1e-6)

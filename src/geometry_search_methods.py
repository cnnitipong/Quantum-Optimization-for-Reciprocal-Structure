"""geometry_search_methods.py -- alternatives to FMQA for the engagement-ratio search (dome, vault).

Same objective, catalogue, 80-FE budget and 10 seeds as the FMQA / GA / random campaign
(cs4_dome2.py, cs_vault_hex2_geo.py).  Methods (all start from the same 10 seeded random designs):
  il       iterative linearisation: one-sided finite-difference slopes of log C per variable,
           linear (separable) model minimised inside a trust region, region shrinks on failure
  qubo2    Bayesian optimisation with an order-2 ANOVA kernel (a QUBO surrogate), acquisition
           (mean - std) minimised by a categorical block annealer -- the FMQA idea with a GP
  hobo3    same with an order-3 ANOVA kernel, i.e. a cubic PUBO/HOBO surrogate
  qml      same with a simulated quantum fidelity kernel (ZZ feature map, 2 reps, one qubit
           per variable, exact statevector) -- quantum-kernel surrogate
  hybrid   qubo2 for 50 evaluations, then classical FE first-improvement local search
Plus 'enum': exhaustive dome optimum (4096 lambda combinations at the largest section,
which is optimal for a compliance-only objective).
Usage: python3 -I src/geometry_search_methods.py [run|enum|report]
Output: results/geometry_search/geo_methods.json (checkpointed per task).
"""
import os, sys, json, time, itertools
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
import numpy as np
_argv = sys.argv; sys.argv = ['x']
import cs4_dome2 as DOME, cs_vault_hex2_geo as VAULT
sys.argv = _argv
OUT = os.path.join(ROOT, 'results', 'geometry_search'); os.makedirs(OUT, exist_ok=True)
PATH = os.path.join(OUT, 'geo_methods.json')
B, NI, L = 80, 10, 8
PROB = {'dome': (DOME.objective, 5), 'vault': (VAULT.objective, 7)}
METHODS = ['il', 'qubo2', 'hobo3', 'qml', 'hybrid']


class Run:
    def __init__(self, prob, seed):
        self.f, self.n = PROB[prob]; self.rng = np.random.default_rng(1000 + seed)
        self.X, self.y, self.cache = [], [], {}

    def ev(self, x):
        x = tuple(int(v) for v in x)
        if x not in self.cache:
            if len(self.y) >= B:
                return None
            c = float(self.f(list(x))); self.cache[x] = c; self.X.append(x); self.y.append(c)
        return self.cache[x]

    def left(self):
        return B - len(self.y)

    def best(self):
        i = int(np.argmin(self.y)); return np.array(self.X[i]), self.y[i]

    def init(self):
        while len(self.y) < NI:
            self.ev(self.rng.integers(0, L, self.n))


# ---------------------------------------------------------------- kernels
def anova(A, Bm, order):
    d = (A[:, None, :] == Bm[None, :, :]).astype(float)          # matches per variable
    e1 = d.sum(-1); k = 1 + e1
    if order >= 2:
        e2 = (e1 ** 2 - e1) / 2; k = k + e2                 # elementary symmetric e2
    if order >= 3:                                                 # d is 0/1, so power sums equal e1
        e3 = (e1 ** 3 - 3 * e1 ** 2 + 2 * e1) / 6; k = k + e3
    n = A.shape[1]; kself = 1 + n + (n * (n - 1) / 2 if order >= 2 else 0) + (n * (n - 1) * (n - 2) / 6 if order >= 3 else 0)
    return k / kself


_H = {}


def zz_states(X):
    """ZZ feature map (H, U(x), H, U(x)) on n qubits, exact statevectors, X levels 0..7."""
    X = np.atleast_2d(X); n = X.shape[1]; N = 2 ** n
    if n not in _H:
        h = np.array([[1, 1], [1, -1]]) / np.sqrt(2); Hn = h
        for _ in range(n - 1):
            Hn = np.kron(Hn, h)
        Z = np.array([[1 - 2 * ((k >> (n - 1 - q)) & 1) for q in range(n)] for k in range(N)], float)
        _H[n] = (Hn, Z)
    Hn, Z = _H[n]
    t = X / (L - 1) * np.pi                                         # angle in [0, pi]
    ph = t @ Z.T                                                    # sum_i t_i z_i
    for i in range(n):
        j = (i + 1) % n
        ph = ph + ((np.pi - t[:, i]) * (np.pi - t[:, j]))[:, None] * (Z[:, i] * Z[:, j])[None, :]
    U = np.exp(1j * ph)
    psi = U * (np.ones(N) / np.sqrt(N))[None, :]
    psi = (psi @ Hn.T) * U
    return psi


def qkernel(A, Bm):
    return np.abs(zz_states(A) @ zz_states(Bm).conj().T) ** 2


class GP:
    def __init__(self, kern, X, y):
        self.k, self.X = kern, np.array(X)
        z = np.log(np.array(y)); self.mu, self.sd = z.mean(), z.std() + 1e-12; t = (z - self.mu) / self.sd
        K = kern(self.X, self.X); best = None
        for s2 in (1e-4, 1e-3, 1e-2, 1e-1):                           # noise by leave-one-out error
            Ki = np.linalg.inv(K + s2 * np.eye(len(t))); a = Ki @ t
            loo = np.mean((a / np.diag(Ki)) ** 2)
            if best is None or loo < best[0]:
                best = (loo, Ki, a)
        _, self.Ki, self.a = best

    def lcb(self, Xc, kappa=1.0):
        Ks = self.k(np.atleast_2d(Xc), self.X)
        m = Ks @ self.a; v = np.maximum(1 - ((Ks @ self.Ki) * Ks).sum(1), 1e-12)
        return m - kappa * np.sqrt(v)


def block_anneal(fun, n, rng, chains=32, sweeps=20, seeds=()):
    S = rng.integers(0, L, (chains, n))
    for i, s in enumerate(seeds[:chains // 2]):
        S[i] = s
    e = fun(S); T0 = max(e.std(), 1e-3); Ts = T0 * (1e-3) ** (np.arange(sweeps) / (sweeps - 1))
    for T in Ts:
        for v in rng.permutation(n):
            C = np.repeat(S[:, None, :], L, 1); C[:, :, v] = np.arange(L)[None, :]
            E = fun(C.reshape(-1, n)).reshape(chains, L)
            P = np.exp(-(E - E.min(1, keepdims=True)) / T); P /= P.sum(1, keepdims=True)
            S[:, v] = (P.cumsum(1) < rng.random((chains, 1))).sum(1).clip(0, L - 1)
    e = fun(S); return S[np.argsort(e)], np.sort(e)


def surrogate_step(r, kern):
    gp = GP(kern, r.X, r.y)
    xb, _ = r.best()
    S, _ = block_anneal(gp.lcb, r.n, r.rng, seeds=[xb])
    for s in S:                                                     # best unevaluated proposal
        if tuple(int(v) for v in s) not in r.cache:
            return r.ev(s)
    while True:                                                     # all seen: random unseen mutation
        x = xb.copy(); x[r.rng.integers(r.n)] = r.rng.integers(L)
        if tuple(int(v) for v in x) not in r.cache or r.rng.random() < 0.01:
            return r.ev(x if tuple(int(v) for v in x) not in r.cache else r.rng.integers(0, L, r.n))


def local_search(r):
    while r.left() > 0:
        x, c = r.best(); improved = False
        for v in r.rng.permutation(r.n):
            for s in (-1, 1):
                z = x.copy(); z[v] += s
                if 0 <= z[v] < L and tuple(z) not in r.cache:
                    if r.ev(z) is None:
                        return
                    if r.cache[tuple(z)] < c:
                        improved = True; break
            if improved or r.left() == 0:
                break
        if not improved:                                            # local optimum: perturb two variables
            z = x.copy()
            for v in r.rng.choice(r.n, 2, replace=False):
                z[v] = r.rng.integers(L)
            r.ev(z)


def iterative_linearisation(r):
    """Trust-region iterative linearisation; every pass must spend at least one new FE evaluation."""
    delta = 3
    while r.left() > 0:
        n0 = len(r.y); x, c = r.best(); g = np.zeros(r.n)
        for v in range(r.n):                                        # one-sided slope of log C
            s = 1 if x[v] < L - 1 else -1
            z = x.copy(); z[v] += s; cz = r.ev(z)
            if cz is None:
                return
            g[v] = (np.log(cz) - np.log(c)) / s
        if r.best()[1] < c:                                         # a probe improved: recentre there
            continue
        step = np.clip(x - np.sign(g) * delta * (np.abs(g) > 1e-9), 0, L - 1).astype(int)
        new = tuple(int(v) for v in step) not in r.cache
        cs = r.ev(step)
        if cs is None:
            return
        if cs < c:
            delta = min(4, delta + 1)
        else:
            delta = max(1, delta - 1)
        if len(r.y) == n0:                                          # nothing new: random unseen design
            while r.left() > 0:
                z = r.rng.integers(0, L, r.n)
                if tuple(int(v) for v in z) not in r.cache:
                    r.ev(z); break


KERN = {'qubo2': lambda A, Bm: anova(A, Bm, 2), 'hobo3': lambda A, Bm: anova(A, Bm, 3), 'qml': qkernel}


def task(args):
    prob, method, seed = args
    r = Run(prob, seed); r.init(); t = time.time()
    if method == 'il':
        iterative_linearisation(r)
    elif method == 'hybrid':
        while len(r.y) < 50:
            surrogate_step(r, KERN['qubo2'])
        local_search(r)
    else:
        while r.left() > 0:
            surrogate_step(r, KERN[method])
    bsf = list(np.minimum.accumulate(r.y))
    x, c = r.best()
    return dict(prob=prob, method=method, seed=seed, best=c, best_x=[int(v) for v in x], bsf=bsf, time_s=time.time() - t)


def enum_task(lams):
    return lams, float(DOME.objective(list(lams) + [7]))


def main(mode):
    import multiprocessing as mp
    res = json.load(open(PATH)) if os.path.exists(PATH) else {}
    if mode == 'run':
        todo = [(p, m, s) for p in ('dome', 'vault') for m in METHODS for s in range(10) if f'{p}|{m}|{s}' not in res]
        with mp.Pool(2) as pool:
            for out in pool.imap_unordered(task, todo):
                res[f"{out['prob']}|{out['method']}|{out['seed']}"] = out
                json.dump(res, open(PATH, 'w')); print(out['prob'], out['method'], out['seed'], round(out['best'], 1), flush=True)
    elif mode == 'enum':
        allx = list(itertools.product(range(L), repeat=4))
        with mp.Pool(2) as pool:
            vals = pool.map(enum_task, allx, chunksize=64)
        i = int(np.argmin([v for _, v in vals]))
        json.dump(dict(x=[list(v[0]) for v in vals], C=[v[1] for v in vals]), open(os.path.join(OUT, 'dome_enum_all.json'), 'w'))
        res['dome_enum'] = dict(best=vals[i][1], best_x=list(vals[i][0]) + [7], n=len(vals),
                                q01=float(np.quantile([v for _, v in vals], 0.01)), median=float(np.median([v for _, v in vals])))
        json.dump(res, open(PATH, 'w')); print(res['dome_enum'])


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'run')


def report():
    from exact_mw import exact_mannwhitney
    from stats_utils import cliffs_delta
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    res = json.load(open(PATH)); out = {}
    STORED = {'dome': 'results/complex/cs4_complex.json', 'vault': 'results/complex/vault_hex_geo.json'}
    LAB = {'fmqa': 'FMQA (paper)', 'ga': 'GA (paper)', 'random': 'Random (paper)', 'il': 'Iterative linearisation',
           'qubo2': 'QUBO-kernel BO', 'hobo3': 'HOBO/PUBO-kernel BO', 'qml': 'Quantum-kernel BO', 'hybrid': 'Hybrid annealer + local search'}
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.6))
    sty = {'fmqa': ('-', 'o', 'white'), 'ga': ('--', 's', 'white'), 'random': (':', 'x', 'black'), 'il': ('-', None, None),
           'qubo2': ('--', None, None), 'hobo3': ('-.', '^', 'black'), 'qml': ('-', 'D', 'black'), 'hybrid': (':', 'v', 'black')}
    for ax, p in zip(axs, ('dome', 'vault')):
        d = json.load(open(os.path.join(ROOT, STORED[p])))['methods']; series = {}
        for k in ('fmqa', 'ga', 'random'):
            rr = d[k]; rr = list(rr.values()) if isinstance(rr, dict) else rr
            series[k] = np.array([np.minimum.accumulate(r['best_history'])[:B] for r in rr])
        for m in METHODS:
            series[m] = np.array([res[f'{p}|{m}|{s}']['bsf'][:B] for s in range(10) if f'{p}|{m}|{s}' in res])
        fin = {k: v[:, -1] for k, v in series.items() if len(v)}
        out[p] = {}
        for k, v in fin.items():
            row = dict(mean=float(v.mean()), std=float(v.std()), best=float(v.min()), n=int(len(v)))
            if k not in ('ga',):
                row['p_vs_ga'] = float(exact_mannwhitney(v, fin['ga'])['p_exact_two_sided'])
                row['delta_vs_ga'] = float(cliffs_delta(v, fin['ga']))
            if k not in ('fmqa',):
                row['p_vs_fmqa'] = float(exact_mannwhitney(v, fin['fmqa'])['p_exact_two_sided'])
            out[p][k] = row
        for k, v in series.items():
            if not len(v):
                continue
            ls, mk, fc = sty[k]; mn = v.mean(0)
            ax.plot(np.arange(1, len(mn) + 1), mn, ls=ls, marker=mk, markevery=(3, 10), ms=3.5, mfc=fc, color='black' if k in METHODS else '0.55',
                    lw=1.0, label=LAB[k])
        ax.set_yscale('log'); ax.set_xlabel('Real FE evaluations'); ax.set_ylabel('Best compliance [J]'); ax.set_title(p.capitalize(), fontsize=9)
        if p == 'dome' and 'dome_enum' in res:
            ax.axhline(res['dome_enum']['best'], color='0.3', lw=0.6); out[p]['exact_optimum'] = res['dome_enum']
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'geo_methods.png'), dpi=160, bbox_inches='tight')
    json.dump(out, open(os.path.join(OUT, 'geo_methods_summary.json'), 'w'), indent=1)
    for p, rows in out.items():
        print('==', p)
        for k, r in rows.items():
            if k == 'exact_optimum':
                print('   exact optimum', round(r['best'], 1)); continue
            print(f"   {LAB[k]:32s} {r['mean']:8.1f} ± {r['std']:6.1f}  best {r['best']:7.1f}  p_GA {r.get('p_vs_ga', float('nan')):.3g}  p_FMQA {r.get('p_vs_fmqa', float('nan')):.3g}")


if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == 'report':
    report()

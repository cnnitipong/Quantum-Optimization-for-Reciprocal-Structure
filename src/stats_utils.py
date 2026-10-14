"""
stats_utils.py -- pure numpy/math implementations of the nonparametric
statistical tests needed for this study (scipy is unavailable in this
sandbox). All three primitives below are standard textbook formulas,
implemented from scratch and self-tested against hand-computable cases at
the bottom of this file.

  - reg_incomplete_beta(x, a, b): regularized incomplete beta function
    I_x(a,b) via the Numerical-Recipes continued-fraction (Lentz) method.
    Used to get exact Student-t and F-distribution tail probabilities
    without scipy.
  - student_t_sf2(t, dof): two-sided Student-t p-value.
  - f_sf(x, d1, d2): upper-tail F-distribution p-value P(F_{d1,d2} > x).
  - mann_whitney_u(x, y): two-sided Mann-Whitney U test, normal
    approximation with tie correction. Returns U, z, p.
  - cliffs_delta(x, y): Cliff's delta effect size (in [-1,1]).
  - brown_forsythe(*groups): median-based Levene's test (Brown-Forsythe),
    returns F statistic and p-value (works for 2+ groups).
"""
import math
import numpy as np


# --------------------------------------------------------------------------
# regularized incomplete beta (continued fraction, Numerical Recipes 6.4)
# --------------------------------------------------------------------------
def _betacf(x, a, b, maxit=300, eps=3e-9, fpmin=1e-300):
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, maxit + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return h


def reg_incomplete_beta(x, a, b):
    """I_x(a,b), regularized incomplete beta, 0<=x<=1, a,b>0."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbt = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
           + a * math.log(x) + b * math.log(1.0 - x))
    bt = math.exp(lbt)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(x, a, b) / a
    else:
        return 1.0 - bt * _betacf(1.0 - x, b, a) / b


def student_t_sf2(t, dof):
    """Two-sided Student-t p-value for statistic t with `dof` degrees of
    freedom: p = I_{dof/(dof+t^2)}(dof/2, 1/2)."""
    t = abs(float(t))
    x = dof / (dof + t * t)
    return reg_incomplete_beta(x, dof / 2.0, 0.5)


def f_sf(x, d1, d2):
    """Upper-tail P(F_{d1,d2} > x) for the F distribution."""
    if x <= 0:
        return 1.0
    xb = d2 / (d2 + d1 * x)
    return reg_incomplete_beta(xb, d2 / 2.0, d1 / 2.0)


def norm_sf2(z):
    """Two-sided standard-normal tail probability for |z|."""
    z = abs(float(z))
    return 2.0 * (1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))


# --------------------------------------------------------------------------
# Mann-Whitney U (normal approximation, tie-corrected, two-sided)
# --------------------------------------------------------------------------
def mann_whitney_u(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n1, n2 = len(x), len(y)
    N = n1 + n2
    combined = np.concatenate([x, y])
    order = np.argsort(combined, kind='mergesort')
    ranks = np.empty(N, dtype=float)
    sorted_vals = combined[order]
    # average ranks for ties
    i = 0
    while i < N:
        j = i
        while j + 1 < N and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0  # ranks are 1-indexed
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1
    R1 = ranks[:n1].sum()
    U1 = R1 - n1 * (n1 + 1) / 2.0
    U2 = n1 * n2 - U1
    U = min(U1, U2)

    # tie correction term
    _, counts = np.unique(combined, return_counts=True)
    tie_term = np.sum(counts**3 - counts)
    mu_U = n1 * n2 / 2.0
    sigma2 = (n1 * n2 / 12.0) * ((N + 1) - tie_term / (N * (N - 1))) if N > 1 else 0.0
    sigma = math.sqrt(max(sigma2, 0.0))
    if sigma == 0:
        z = 0.0
    else:
        # continuity correction toward the mean
        diff = U1 - mu_U
        cc = 0.5 * np.sign(diff)
        z = (diff - cc) / sigma
    p = norm_sf2(z)
    return dict(U1=float(U1), U2=float(U2), U=float(U), z=float(z), p=float(p),
                n1=n1, n2=n2)


def cliffs_delta(x, y):
    """Cliff's delta effect size in [-1, 1]. Positive => x tends to be
    larger than y."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    diff = x[:, None] - y[None, :]
    more = np.sum(diff > 0)
    less = np.sum(diff < 0)
    n1n2 = len(x) * len(y)
    return float((more - less) / n1n2)


# --------------------------------------------------------------------------
# Brown-Forsythe (median-centered Levene) test, k>=2 groups
# --------------------------------------------------------------------------
def brown_forsythe(*groups):
    groups = [np.asarray(g, dtype=float) for g in groups]
    k = len(groups)
    N = sum(len(g) for g in groups)
    Z = [np.abs(g - np.median(g)) for g in groups]
    Zbar_i = [z.mean() for z in Z]
    Zbar = np.concatenate(Z).mean()
    ssb = sum(len(Z[i]) * (Zbar_i[i] - Zbar)**2 for i in range(k))
    ssw = sum(np.sum((Z[i] - Zbar_i[i])**2) for i in range(k))
    d1, d2 = k - 1, N - k
    if ssw <= 0 or d2 <= 0:
        return dict(F=float('nan'), p=float('nan'), d1=d1, d2=d2)
    F = (ssb / d1) / (ssw / d2)
    p = f_sf(F, d1, d2)
    return dict(F=float(F), p=float(p), d1=d1, d2=d2)


if __name__ == "__main__":
    # self-tests against known/hand-verifiable cases
    # 1) Student-t: for large dof, t=1.96 two-sided p ~ 0.05
    p = student_t_sf2(1.96, 1e6)
    print("t=1.96, dof=1e6 -> p=", p, "(expect ~0.05)")
    assert abs(p - 0.05) < 0.001

    # 2) Mann-Whitney on a clear-cut separated case: all of x > all of y
    x = [10, 11, 12, 13, 14]
    y = [1, 2, 3, 4, 5]
    r = mann_whitney_u(x, y)
    print("fully separated MWU:", r)
    assert r['U'] == 0.0
    d = cliffs_delta(x, y)
    print("cliffs delta (fully separated):", d)
    assert abs(d - 1.0) < 1e-9

    # 3) identical distributions -> U ~ n1*n2/2, p large
    rng = np.random.default_rng(0)
    a = rng.normal(size=200)
    b = rng.normal(size=200)
    r2 = mann_whitney_u(a, b)
    print("same-dist MWU p (expect large, not significant most of time):", r2['p'])

    # 4) Brown-Forsythe: group with much larger spread should give small p
    rng2 = np.random.default_rng(1)
    g1 = rng2.normal(0, 1.0, 200)
    g2 = rng2.normal(0, 5.0, 200)
    bf = brown_forsythe(g1, g2)
    print("Brown-Forsythe (different variances):", bf)
    assert bf['p'] < 0.01

    print("ALL stats_utils.py SELF-TESTS PASSED")

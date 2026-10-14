"""
exact_mw.py -- EXACT two-sided Mann-Whitney U p-value by full enumeration
over the rank-permutation distribution (numpy/itertools only, no scipy).

Method: combine the two samples, compute mid-ranks (ties get the average
rank of their tied block, exactly as in stats_utils.mann_whitney_u). Under
H0 (exchangeability of the N=n1+n2 observations), every one of the
C(N,n1) ways of choosing which n1 of the N rank values go to sample 1 is
equally likely. This is the exact permutation/randomization distribution
of the Wilcoxon rank-sum statistic T (equivalently the Mann-Whitney U),
and it is ties-correct by construction (tied observations keep their
averaged rank; enumerating over INDEX subsets, not value subsets, is
exactly what a network/DP algorithm for the tied exact Mann-Whitney test
also computes, just via a different bookkeeping route). Feasible for
n1,n2 <= 10 (N <= 20 => C(20,10) = 184,756 combinations, trivial to
enumerate exactly with plain Python/itertools in well under a second).

Two-sided exact p-value convention (matches R's wilcox.test(exact=TRUE)
and scipy's 'exact' method): p = min(1, 2*min(P(T<=T_obs), P(T>=T_obs))).
"""
import itertools
from math import comb
import numpy as np


def _midranks(combined):
    """Average ('mid') ranks for a 1D array, ties get the mean rank of
    their block. 1-indexed ranks, identical convention to
    stats_utils.mann_whitney_u."""
    combined = np.asarray(combined, dtype=float)
    N = len(combined)
    order = np.argsort(combined, kind='mergesort')
    sorted_vals = combined[order]
    ranks = np.empty(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j + 1 < N and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1
    return ranks


def exact_mannwhitney(x, y, max_N=20):
    """Exact two-sided Mann-Whitney U test via full enumeration of the
    rank-permutation null distribution. Returns dict with U (Mann-Whitney
    U for sample x), T (rank sum for sample x), the exact two-sided
    p-value, and the one-sided tail masses. Raises ValueError if
    n1+n2 > max_N (not attempted -- exponential enumeration cost)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n1, n2 = len(x), len(y)
    N = n1 + n2
    if N > max_N:
        raise ValueError(f"exact_mannwhitney: N={N} exceeds max_N={max_N}; "
                          f"full enumeration (C(N,n1)) not attempted.")
    if n1 == 0 or n2 == 0:
        return dict(U=float('nan'), T=float('nan'), n1=n1, n2=n2, N=N,
                     total_combinations=0, p_le=1.0, p_ge=1.0,
                     p_exact_two_sided=1.0)

    combined = np.concatenate([x, y])
    ranks = _midranks(combined)

    T_obs = float(ranks[:n1].sum())
    U_obs = T_obs - n1 * (n1 + 1) / 2.0

    total = comb(N, n1)
    eps = 1e-9
    count_le = 0
    count_ge = 0
    idx_range = range(N)
    for combo in itertools.combinations(idx_range, n1):
        T = ranks[list(combo)].sum()
        if T <= T_obs + eps:
            count_le += 1
        if T >= T_obs - eps:
            count_ge += 1

    p_le = count_le / total
    p_ge = count_ge / total
    p_two = min(1.0, 2.0 * min(p_le, p_ge))

    return dict(U=float(U_obs), T=float(T_obs), n1=n1, n2=n2, N=N,
                total_combinations=int(total), p_le=float(p_le),
                p_ge=float(p_ge), p_exact_two_sided=float(p_two))


def wilson_ci(successes, n, z=1.959963984540054):
    """Wilson score 95% CI for a binomial proportion (default z for 95%,
    exact standard-normal quantile via math.erf-based inverse is not
    needed here -- 1.959963984540054 is the well-known closed-form value
    for Phi^-1(0.975), used as a literal constant, consistent with using
    only numpy/math elsewhere in this project)."""
    if n == 0:
        return dict(phat=float('nan'), lo=float('nan'), hi=float('nan'), n=n,
                     successes=successes, z=z)
    phat = successes / n
    denom = 1.0 + z * z / n
    center = phat + z * z / (2 * n)
    half = z * np.sqrt((phat * (1 - phat) / n) + (z * z / (4 * n * n)))
    lo = (center - half) / denom
    hi = (center + half) / denom
    return dict(phat=float(phat), lo=float(max(0.0, lo)), hi=float(min(1.0, hi)),
                n=n, successes=successes, z=z)


if __name__ == "__main__":
    from stats_utils import mann_whitney_u

    # ---- Validation case 1: fully separated samples (textbook exact case)
    # n1=n2=4, all x > all y => U=0, exact two-sided p = 2/C(8,4) = 2/70
    x1 = [10, 11, 12, 13]
    y1 = [1, 2, 3, 4]
    r1 = exact_mannwhitney(x1, y1)
    approx1 = mann_whitney_u(x1, y1)
    print("Case 1 (fully separated, n1=n2=4):")
    print("  exact:", r1)
    print("  normal-approx:", approx1)
    expected_p1 = 2.0 / comb(8, 4)
    print(f"  expected exact p = 2/C(8,4) = {expected_p1:.6f}")
    assert abs(r1['p_exact_two_sided'] - expected_p1) < 1e-9
    # r1['U'] is U for sample 1 (the LARGER-valued sample here), so it
    # equals n1*n2=16, not 0; min(U1,U2)=0 is what the normal-approx
    # helper reports as its 'U'. Cross-check via the min-convention:
    assert min(r1['U'], r1['n1'] * r1['n2'] - r1['U']) == approx1['U']
    print("  PASS (matches textbook closed form; normal approx is more "
          f"conservative here: {approx1['p']:.6f} vs exact {r1['p_exact_two_sided']:.6f})")

    # ---- Validation case 2: known textbook table value, n1=3, n2=5, U=0
    # (all of the n1=3 group below all of the n2=5 group) exact
    # two-sided p = 2/C(8,3) = 2/56 = 0.035714...
    x2 = [1, 2, 3]
    y2 = [4, 5, 6, 7, 8]
    r2 = exact_mannwhitney(x2, y2)
    approx2 = mann_whitney_u(x2, y2)
    expected_p2 = 2.0 / comb(8, 3)
    print("\nCase 2 (n1=3,n2=5, fully separated the other way):")
    print("  exact:", r2)
    print(f"  expected exact p = 2/C(8,3) = {expected_p2:.6f}")
    assert abs(r2['p_exact_two_sided'] - expected_p2) < 1e-9
    print("  PASS")

    # ---- Validation case 3: ties present, sanity vs normal approx
    x3 = [1, 2, 2, 3, 4]
    y3 = [2, 3, 3, 4, 5]
    r3 = exact_mannwhitney(x3, y3)
    approx3 = mann_whitney_u(x3, y3)
    print("\nCase 3 (ties present, n1=n2=5):")
    print("  exact:", r3)
    print("  normal-approx (tie-corrected):", approx3)
    print("  both should be in the same statistical-significance ballpark "
          "(neither significant here) -- no assertion, just a sanity check")

    # ---- Wilson CI sanity checks
    w1 = wilson_ci(6, 10)
    w2 = wilson_ci(7, 10)
    w3 = wilson_ci(5, 10)
    print("\nWilson 95% CIs:")
    print("  6/10:", w1)
    print("  7/10:", w2)
    print("  5/10:", w3)
    assert 0.0 <= w1['lo'] < w1['phat'] < w1['hi'] <= 1.0

    print("\nALL exact_mw.py SELF-TESTS PASSED")

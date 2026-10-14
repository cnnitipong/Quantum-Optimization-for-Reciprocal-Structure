import os
import os as _os_bootstrap
ROOT = _os_bootstrap.path.dirname(_os_bootstrap.path.dirname(_os_bootstrap.path.abspath(__file__)))
"""
fastsolve.py -- exact banded-Cholesky fast path for Frame3D static solves.

WHY
---
frame3d.Frame3D.solve() forms the dense reduced stiffness matrix Kff and
calls np.linalg.solve (LAPACK dgesv, general LU, O(n^3)).  For the meshes
used by the redesigned vault (rf_vault_hex2: 4448 restrained DOF) and dome
(rf_dome2) that costs ~1.7 s (4 BLAS threads) to ~3.3 s (1 thread) PER
EVALUATION, which makes an 80-eval x 10-seed x 3-method geometry-search
campaign (2400 FE solves) cost ~2 CPU-hours per typology.  scipy is NOT
available in this sandbox, so scipy.sparse / solveh_banded / splu are not
options.

WHAT THIS DOES
--------------
Kff is symmetric positive definite (a properly restrained linear-elastic
stiffness matrix) and, after a reverse-Cuthill-McKee (RCM) renumbering of
the NODES, extremely narrowly banded: measured half-bandwidth is ~120 DOF
for rf_vault_hex2 (4448 DOF) and ~156 DOF for the dome family.  This
module therefore:

  1. computes an RCM node ordering from the element adjacency graph
     (pure-python BFS, ~1 ms),
  2. builds the free-DOF permutation induced by that node ordering,
  3. permutes Kff into block-tridiagonal form with block size s = the
     exact measured half-bandwidth (so block (I,J) is provably ZERO for
     |I-J| >= 2: the closest entry of such a block is (I-J-1)*s + 1 >
     halfband),
  4. runs a standard BLOCKED (block-tridiagonal) Cholesky factorisation
     IN PLACE, followed by blocked forward/back substitution.

This is the SAME linear system and the SAME (exact, direct, non-iterative)
class of factorisation -- not an approximation, preconditioner, reduced
model, or tolerance-based iterative scheme.  It only reorders the
elimination so work is proportional to n*s^2 instead of n^3/3.  Measured
agreement with the dense np.linalg.solve path is reported in
results/complex/banded_fastpath_validation.json (relative max-abs
displacement / reaction / compliance differences) and must be at machine
precision for the fast path to be used in any campaign.

USAGE
-----
    from fastsolve import fast_solve
    frm = build_frame(model, group_section, load_per_node=...)
    fast_solve(frm)              # sets frm.U, frm.R, frm.K, frm._elem_data
    c = frm.compliance()

`fast_solve` is a drop-in replacement for `frm.solve(check_cond=False)`
(the project's existing campaign-inner-loop fast path).  Like that path it
performs no cond() mechanism guard -- non-singularity of each geometry
CLASS is established separately by the validation gates (gate1) run with
check_cond=True on the safe dense path.  A genuinely singular Kff still
raises np.linalg.LinAlgError here (numpy's Cholesky fails on a
non-positive-definite block), so this is not a silent-failure path.
"""
import numpy as np
from collections import deque

DOF_PER_NODE = 6


# --------------------------------------------------------------------------
# assembly (vectorised, numerically identical to Frame3D.assemble)
# --------------------------------------------------------------------------
def assemble_fast(frm):
    """Same dense K as Frame3D.assemble(), but scattering each element's
    12x12 global stiffness with one fancy-indexed += instead of a 144-step
    python double loop.  Also populates frm._elem_data exactly as
    Frame3D.assemble() does (needed by element_end_forces_local)."""
    from frame3d import transformation_matrix, local_stiffness
    n = frm.ndof()
    K = np.zeros((n, n))
    elem_data = []
    for e in frm.elements:
        ni, nj = frm.nodes[e['i']], frm.nodes[e['j']]
        T, L = transformation_matrix(ni, nj, e.get('ref_vec'))
        k_local = local_stiffness(e['E'], e['G'], e['A'], e['Iy'], e['Iz'], e['J'], L)
        k_global = T.T @ k_local @ T
        dofs = np.array([frm.gdof(e['i'], d) for d in range(6)]
                        + [frm.gdof(e['j'], d) for d in range(6)])
        K[np.ix_(dofs, dofs)] += k_global
        elem_data.append(dict(L=L, T=T, k_local=k_local, dofs=list(dofs)))
    frm._elem_data = elem_data
    return K


# --------------------------------------------------------------------------
# RCM node ordering
# --------------------------------------------------------------------------
def rcm_order(frm):
    """Reverse Cuthill-McKee node ordering of the element adjacency graph."""
    n = len(frm.nodes)
    adj = [set() for _ in range(n)]
    for e in frm.elements:
        adj[e['i']].add(e['j'])
        adj[e['j']].add(e['i'])
    adj = [sorted(a) for a in adj]
    deg = [len(a) for a in adj]
    seen = [False] * n
    order = []
    for start in sorted(range(n), key=lambda i: deg[i]):
        if seen[start]:
            continue
        seen[start] = True
        q = deque([start])
        while q:
            v = q.popleft()
            order.append(v)
            for w in sorted(adj[v], key=lambda x: deg[x]):
                if not seen[w]:
                    seen[w] = True
                    q.append(w)
    return order[::-1]


def free_dof_permutation(frm, order=None):
    """Return (perm, halfband): perm is the list of FREE global DOF indices
    in RCM order; halfband is the exact half-bandwidth of the permuted Kff
    implied by the element connectivity (max spread of any single element's
    free DOFs in permuted position space)."""
    if order is None:
        order = rcm_order(frm)
    fixed = set(frm.gdof(ni, d) for (ni, d) in frm.fixed.keys())
    perm = [DOF_PER_NODE * nd + d for nd in order for d in range(DOF_PER_NODE)
            if DOF_PER_NODE * nd + d not in fixed]
    pos = {g: k for k, g in enumerate(perm)}
    halfband = 0
    for e in frm.elements:
        ps = [pos[g] for g in
              ([DOF_PER_NODE * e['i'] + d for d in range(DOF_PER_NODE)]
               + [DOF_PER_NODE * e['j'] + d for d in range(DOF_PER_NODE)])
              if g in pos]
        if len(ps) > 1:
            halfband = max(halfband, max(ps) - min(ps))
    return perm, int(halfband)


# --------------------------------------------------------------------------
# blocked (block-tridiagonal) Cholesky solve
# --------------------------------------------------------------------------
def band_chol_solve(A, b, halfband, inplace=True):
    """Solve A x = b for symmetric positive definite, banded A (half-
    bandwidth `halfband`) by blocked Cholesky.  A is overwritten with its
    Cholesky blocks when inplace=True.  Exact direct solve."""
    n = A.shape[0]
    if n == 0:
        return np.zeros(0)
    s = max(int(halfband), 1)
    s = min(s, n)
    if not inplace:
        A = A.copy()
    nb = (n + s - 1) // s
    sl = [slice(I * s, min((I + 1) * s, n)) for I in range(nb)]

    # --- factorise: A = L L^T, L block-bidiagonal ---
    for I in range(nb):
        r = sl[I]
        if I > 0:
            p = sl[I - 1]
            Lsub = A[r, p]
            A[r, r] -= Lsub @ Lsub.T
        try:
            L = np.linalg.cholesky(A[r, r])
        except np.linalg.LinAlgError as exc:
            raise np.linalg.LinAlgError(
                f"banded Cholesky failed on diagonal block {I} "
                f"(matrix not positive definite -- mechanism suspected): {exc}")
        A[r, r] = L
        if I + 1 < nb:
            q = sl[I + 1]
            A[q, r] = np.linalg.solve(L, A[q, r].T).T

    # --- forward substitution: L y = b ---
    y = np.array(b, dtype=float, copy=True)
    for I in range(nb):
        r = sl[I]
        if I > 0:
            p = sl[I - 1]
            y[r] -= A[r, p] @ y[p]
        y[r] = np.linalg.solve(A[r, r], y[r])

    # --- back substitution: L^T x = y ---
    x = y
    for I in range(nb - 1, -1, -1):
        r = sl[I]
        if I + 1 < nb:
            q = sl[I + 1]
            x[r] -= A[q, r].T @ x[q]
        x[r] = np.linalg.solve(A[r, r].T, x[r])
    return x


# --------------------------------------------------------------------------
# public entry point
# --------------------------------------------------------------------------
def fast_solve(frm, want_reactions=True):
    """Drop-in replacement for Frame3D.solve(check_cond=False) using the
    RCM + blocked-banded-Cholesky path.  Sets frm.U, frm.R, frm.K."""
    K = assemble_fast(frm)
    F = frm.load_vector()
    n = frm.ndof()
    perm, halfband = free_dof_permutation(frm)
    if len(perm) == 0:
        U = np.zeros(n)
        R = K @ U - F
        frm.U, frm.R, frm.K = U, R, K
        return U, R
    idx = np.asarray(perm)
    Kp = K[np.ix_(idx, idx)]
    Fp = F[idx]
    Up = band_chol_solve(Kp, Fp, halfband, inplace=True)
    U = np.zeros(n)
    U[idx] = Up
    R = (K @ U - F) if want_reactions else np.zeros(n)
    frm.U, frm.R, frm.K = U, R, K
    return U, R


def validate_against_dense(frm):
    """Solve the same frame both ways and return a dict of relative
    differences.  Used to produce banded_fastpath_validation.json."""
    import time
    t0 = time.time()
    U1, R1 = frm.solve(check_cond=False)
    t_dense = time.time() - t0
    c1 = frm.compliance()
    d1 = frm.max_abs_disp()

    t0 = time.time()
    U2, R2 = fast_solve(frm)
    t_band = time.time() - t0
    c2 = frm.compliance()
    d2 = frm.max_abs_disp()

    perm, hb = free_dof_permutation(frm)
    uscale = max(np.abs(U1).max(), 1e-300)
    rscale = max(np.abs(R1).max(), 1e-300)
    return dict(
        ndof_total=int(frm.ndof()), ndof_free=int(len(perm)), halfband_dof=int(hb),
        n_blocks=int((len(perm) + max(hb, 1) - 1) // max(hb, 1)),
        t_dense_s=float(t_dense), t_banded_s=float(t_band),
        speedup=float(t_dense / max(t_band, 1e-9)),
        rel_max_disp_diff=float(np.abs(U1 - U2).max() / uscale),
        rel_max_reaction_diff=float(np.abs(R1 - R2).max() / rscale),
        rel_compliance_diff=float(abs(c1 - c2) / max(abs(c1), 1e-300)),
        rel_dmax_diff=float(abs(d1 - d2) / max(abs(d1), 1e-300)),
        compliance_dense=float(c1), compliance_banded=float(c2),
    )


# --------------------------------------------------------------------------
# SPD condition-number estimate (fast replacement for np.linalg.cond)
# --------------------------------------------------------------------------
class BandedFactor:
    """Reusable blocked-banded Cholesky factorisation of a frame's Kff."""

    def __init__(self, frm):
        K = assemble_fast(frm)
        self.perm, self.halfband = free_dof_permutation(frm)
        idx = np.asarray(self.perm)
        self.idx = idx
        self.Kff = K[np.ix_(idx, idx)]
        self.A = self.Kff.copy()
        n = self.A.shape[0]
        s = max(int(self.halfband), 1)
        self.s = s
        self.nb = (n + s - 1) // s
        self.sl = [slice(I * s, min((I + 1) * s, n)) for I in range(self.nb)]
        A, sl, nb = self.A, self.sl, self.nb
        for I in range(nb):
            r = sl[I]
            if I > 0:
                p = sl[I - 1]
                L = A[r, p]
                A[r, r] -= L @ L.T
            L = np.linalg.cholesky(A[r, r])
            A[r, r] = L
            if I + 1 < nb:
                q = sl[I + 1]
                A[q, r] = np.linalg.solve(L, A[q, r].T).T

    def solve(self, b):
        A, sl, nb = self.A, self.sl, self.nb
        y = np.array(b, dtype=float, copy=True)
        for I in range(nb):
            r = sl[I]
            if I > 0:
                y[r] -= A[r, sl[I - 1]] @ y[sl[I - 1]]
            y[r] = np.linalg.solve(A[r, r], y[r])
        x = y
        for I in range(nb - 1, -1, -1):
            r = sl[I]
            if I + 1 < nb:
                x[r] -= A[sl[I + 1], r].T @ x[sl[I + 1]]
            x[r] = np.linalg.solve(A[r, r].T, x[r])
        return x


def cond_spd_estimate(frm, n_iter=60, seed=0):
    """2-norm condition number of the restrained stiffness matrix Kff.

    Kff is symmetric positive definite, so cond_2(Kff) = lambda_max/lambda_min
    exactly; lambda_max is obtained by power iteration on Kff and lambda_min by
    inverse iteration through the banded Cholesky factor (both with Rayleigh
    quotients).  This is mathematically the SAME quantity np.linalg.cond(Kff)
    returns for an SPD matrix, but costs O(n*halfband^2 + n_iter*n*halfband)
    instead of a full O(n^3) SVD -- np.linalg.cond on a 4k-DOF Kff does not fit
    in this sandbox's per-call wall-clock budget.  Agreement with
    np.linalg.cond is validated in banded_fastpath_validation.json.

    Returns dict(lambda_min, lambda_max, cond, singular).
    """
    fac = BandedFactor(frm)
    K = fac.Kff
    n = K.shape[0]
    if n == 0:
        return dict(lambda_min=0.0, lambda_max=0.0, cond=float('inf'), singular=True)
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(n)
    v /= np.linalg.norm(v)
    for _ in range(n_iter):
        v = K @ v
        nv = np.linalg.norm(v)
        if nv == 0:
            break
        v /= nv
    lam_max = float(v @ (K @ v))
    w = rng.standard_normal(n)
    w /= np.linalg.norm(w)
    for _ in range(n_iter):
        w = fac.solve(w)
        nw = np.linalg.norm(w)
        if nw == 0:
            break
        w /= nw
    lam_min = float(w @ (K @ w))
    cond = float(lam_max / lam_min) if lam_min > 0 else float('inf')
    return dict(lambda_min=lam_min, lambda_max=lam_max, cond=cond,
                singular=bool(not np.isfinite(cond) or cond > 1e14))

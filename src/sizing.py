import os
import os as _os_bootstrap
ROOT = _os_bootstrap.path.dirname(_os_bootstrap.path.dirname(_os_bootstrap.path.abspath(__file__)))
"""
sizing.py -- F1 iterative discrete sizing engine (strain-energy-linearized
QUBO + move-limit trust region + mass-target penalty + one-hot penalty),
plus the shared real-FE objective function used by the GA / random-search
baselines so that every method is compared at an EQUAL NUMBER OF REAL FE
EVALUATIONS.

Algorithm (per outer iteration t):
  1. Solve the FE model with the current per-group section assignment
     (1 real FE evaluation) -> global displacement U.
  2. For every group k and every candidate section c within the move-limit
     window around the current index, compute the local strain energy
     SE_k(c) that the group's elements WOULD have under the *current*
     displacement field U if they had section c (frozen-field / sequential
     linearization -- standard trust-region SQP-like sizing update).
  3. Build a QUBO: linear reward -SE_k(c) (maximize stiffness contribution)
     + quadratic penalty driving total mass to a target + one-hot penalty
     per group; candidates outside the move-limit window are excluded via
     a large linear penalty.
  4. Solve the QUBO with simulated annealing (qubo.sa_solve); decode the
     winning section per group; this becomes iteration t+1's assignment.
Convergence: stop early if the assignment stops changing.
"""
import numpy as np
from frame3d import transformation_matrix, local_stiffness
from reciprocal_geom import build_frame, model_mass, LINK_SECTION
from sections import CHS_CATALOG, N_SECTIONS, E_STEEL, G_STEEL
from qubo import QUBOBuilder, sa_solve

BIG_PENALTY = 1e6


def element_geom_cache(model):
    nodes = np.array(model.nodes)
    cache = []
    for e in model.elements:
        T, L = transformation_matrix(nodes[e['i']], nodes[e['j']])
        dofs = [e['i'] * 6 + d for d in range(6)] + [e['j'] * 6 + d for d in range(6)]
        cache.append(dict(T=T, L=L, dofs=dofs, group=e['group'], kind=e['kind']))
    return cache


def group_element_indices(model):
    groups = {}
    for idx, e in enumerate(model.elements):
        if e['kind'] == 'beam':
            groups.setdefault(e['group'], []).append(idx)
    return groups


def evaluate_design(model, idx_vec, load_per_node=-1000.0, check_cond=True):
    """Real FE evaluation: idx_vec[k] = section index (0..7) for group k.
    Returns (compliance, mass, Frame3D).

    check_cond: forwarded to Frame3D.solve(); default True (safe path,
    unchanged behaviour for all pre-existing campaigns). Pass False only
    inside an optimization inner loop for a geometry/section-assignment
    class whose non-singularity has ALREADY been verified once with
    check_cond=True (see frame3d.py's solve() docstring) -- used by the
    complex-geometry (rf_dome_complex/davinci_bridge_complex) campaigns to
    skip the expensive cond-number check on every inner-loop FE solve.

    When check_cond=False the solve is routed through
    fastsolve.fast_solve() -- the RCM + blocked-banded-Cholesky EXACT
    direct solve of the identical linear system (see fastsolve.py and
    results/complex/banded_fastpath_validation.json: agreement with the
    dense np.linalg.solve path at ~1e-11 relative, i.e. machine precision
    for these condition numbers).  check_cond=True keeps the original
    dense cond-guarded path unchanged."""
    gs = {k: CHS_CATALOG[idx_vec[k]] for k in range(model.n_groups)}
    frm = build_frame(model, gs, load_per_node=load_per_node)
    if check_cond:
        frm.solve(check_cond=True)
    else:
        from fastsolve import fast_solve
        fast_solve(frm)
    C = frm.compliance()
    M = model_mass(model, gs)
    return C, M, frm


def make_ga_objective(model, mass_target, w_mass=5.0, load_per_node=-1000.0,
                       compliance_scale=1.0, check_cond=True):
    """Scalar objective (to MINIMIZE) shared by GA / random-search
    baselines: normalized compliance + quadratic mass-target penalty. One
    real FE evaluation per call -- directly comparable to one SA-QUBO
    outer iteration. check_cond: see evaluate_design()."""
    counters = {'n_calls': 0}

    def obj(idx_vec):
        counters['n_calls'] += 1
        try:
            C, M, _ = evaluate_design(model, idx_vec, load_per_node, check_cond=check_cond)
        except np.linalg.LinAlgError:
            return 1e9
        rel_mass_err = (M - mass_target) / mass_target
        return C / compliance_scale + w_mass * rel_mass_err**2

    obj.counters = counters
    return obj


def iterative_qubo_sizing(model, mass_target, n_iters=80, w_stiff=1.0,
                           w_mass=5.0, w_onehot=None, move_limit=2,
                           sa_reads=40, sa_sweeps=120, seed=0,
                           init_idx=None, load_per_node=-1000.0,
                           compliance_scale=1.0, patience=8, check_cond=True):
    n_groups = model.n_groups
    n_sec = N_SECTIONS
    geom = element_geom_cache(model)
    groups = group_element_indices(model)
    mass_per_len = np.array([s['mass_per_len'] for s in CHS_CATALOG])
    elem_len = {i: geom[i]['L'] for i in range(len(geom))}

    if init_idx is None:
        init_idx = [n_sec // 2] * n_groups
    cur = list(init_idx)

    comp_hist, mass_hist, idx_hist, fe_calls = [], [], [], 0
    no_change = 0
    for it in range(n_iters):
        C, M, frm = evaluate_design(model, cur, load_per_node, check_cond=check_cond)
        fe_calls += 1
        comp_hist.append(C)
        mass_hist.append(M)
        idx_hist.append(list(cur))

        # strain energy of each group's elements for every candidate section
        SE = np.zeros((n_groups, n_sec))
        mass_coef = np.zeros((n_groups, n_sec))
        for k, elem_idxs in groups.items():
            for ei in elem_idxs:
                g = geom[ei]
                u_global = frm.U[g['dofs']]
                u_local = g['T'] @ u_global
                for c in range(n_sec):
                    sp = CHS_CATALOG[c]
                    kl = local_stiffness(E_STEEL, G_STEEL, sp['A'], sp['Iy'],
                                          sp['Iz'], sp['J'], g['L'])
                    SE[k, c] += 0.5 * u_local @ kl @ u_local
                    mass_coef[k, c] += sp['mass_per_len'] * g['L']

        n_bits = n_groups * n_sec
        qb = QUBOBuilder(n_bits)

        def vidx(k, c):
            return k * n_sec + c

        SEmax = SE.max() if SE.max() > 0 else 1.0
        for k in range(n_groups):
            lo, hi = max(0, cur[k] - move_limit), min(n_sec - 1, cur[k] + move_limit)
            for c in range(n_sec):
                if lo <= c <= hi:
                    qb.add_linear(vidx(k, c), -w_stiff * SE[k, c] / SEmax)
                else:
                    qb.add_linear(vidx(k, c), BIG_PENALTY)

        all_idx = [vidx(k, c) for k in range(n_groups) for c in range(n_sec)]
        all_coef = [mass_coef[k, c] for k in range(n_groups) for c in range(n_sec)]
        qb.add_target_penalty(all_idx, all_coef, mass_target,
                               w_mass / max(mass_target**2, 1e-9))

        if w_onehot is None:
            w_onehot_use = 3.0 * (w_stiff + w_mass)
        else:
            w_onehot_use = w_onehot
        for k in range(n_groups):
            qb.add_onehot_penalty([vidx(k, c) for c in range(n_sec)], w_onehot_use)

        sa = sa_solve(qb.Q, n_reads=sa_reads, n_sweeps=sa_sweeps,
                      seed=seed * 10000 + it)
        Xb = sa['best_x'].reshape(n_groups, n_sec)
        new_idx = list(Xb.argmax(axis=1))

        if new_idx == cur:
            no_change += 1
        else:
            no_change = 0
        cur = new_idx
        if no_change >= patience:
            break

    return dict(compliance_history=np.array(comp_hist),
                mass_history=np.array(mass_hist),
                idx_history=idx_hist, final_idx=cur, fe_calls=fe_calls,
                n_iters_run=len(comp_hist))


def weighted_sum_qubo_sizing(model, w, mass_scale, compliance_scale,
                              n_iters=12, move_limit=2, sa_reads=25,
                              sa_sweeps=90, seed=0, init_idx=None,
                              load_per_node=-1000.0):
    """F4 multi-objective weighted-sum QUBO: minimize
    w*compliance_norm + (1-w)*mass_norm via the SAME strain-energy
    linearization used by iterative_qubo_sizing, but with a purely LINEAR
    per-iteration objective (no quadratic mass-target penalty needed since
    mass enters linearly already). Returns the (mass, compliance) trace so
    a Pareto front can be assembled across a sweep of w values / seeds.
    """
    n_groups = model.n_groups
    n_sec = N_SECTIONS
    geom = element_geom_cache(model)
    groups = group_element_indices(model)

    if init_idx is None:
        init_idx = [n_sec // 2] * n_groups
    cur = list(init_idx)

    comp_hist, mass_hist, idx_hist = [], [], []
    for it in range(n_iters):
        C, M, frm = evaluate_design(model, cur, load_per_node)
        comp_hist.append(C)
        mass_hist.append(M)
        idx_hist.append(list(cur))

        SE = np.zeros((n_groups, n_sec))
        mass_coef = np.zeros((n_groups, n_sec))
        for k, elem_idxs in groups.items():
            for ei in elem_idxs:
                g = geom[ei]
                u_global = frm.U[g['dofs']]
                u_local = g['T'] @ u_global
                for c in range(n_sec):
                    sp = CHS_CATALOG[c]
                    kl = local_stiffness(E_STEEL, G_STEEL, sp['A'], sp['Iy'],
                                          sp['Iz'], sp['J'], g['L'])
                    SE[k, c] += 0.5 * u_local @ kl @ u_local
                    mass_coef[k, c] += sp['mass_per_len'] * g['L']

        n_bits = n_groups * n_sec
        qb = QUBOBuilder(n_bits)

        def vidx(k, c):
            return k * n_sec + c

        SEmax = SE.max() if SE.max() > 0 else 1.0
        for k in range(n_groups):
            lo, hi = max(0, cur[k] - move_limit), min(n_sec - 1, cur[k] + move_limit)
            for c in range(n_sec):
                if lo <= c <= hi:
                    stiff_term = -w * (SE[k, c] / SEmax)
                    mass_term = (1 - w) * (mass_coef[k, c] / mass_scale)
                    qb.add_linear(vidx(k, c), stiff_term + mass_term)
                else:
                    qb.add_linear(vidx(k, c), BIG_PENALTY)

        w_onehot = 3.0
        for k in range(n_groups):
            qb.add_onehot_penalty([vidx(k, c) for c in range(n_sec)], w_onehot)

        sa = sa_solve(qb.Q, n_reads=sa_reads, n_sweeps=sa_sweeps,
                      seed=seed * 10000 + it)
        Xb = sa['best_x'].reshape(n_groups, n_sec)
        cur = list(Xb.argmax(axis=1))

    return dict(compliance_history=np.array(comp_hist),
                mass_history=np.array(mass_hist), idx_history=idx_hist,
                final_idx=cur)

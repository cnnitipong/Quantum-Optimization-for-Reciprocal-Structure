"""
greedy_sizing.py -- R3 ablation: "greedy-linearized" classical baseline.

Identical F1 outer loop to sizing.iterative_qubo_sizing (same frozen-field
strain-energy objective, same mass-target penalty, same +-2 move-limit
trust region, same real-FE budget accounting: exactly one real FE
evaluation per outer iteration), but the per-iteration discrete
subproblem -- which iterative_qubo_sizing hands to sa_solve() (simulated
annealing on the QUBO) -- is instead solved by DETERMINISTIC greedy
coordinate descent: sweep the member groups in a fixed order, and for
each group pick (holding all other groups fixed at their current value)
the candidate section within the move-limit window that minimizes the
exact same objective (linear strain-energy reward + quadratic mass-target
penalty) that the QUBO encodes; repeat full sweeps until no group changes
or a sweep cap is hit. This isolates the annealer's (SA-QUBO's)
contribution from the strain-energy linearization + move-limit trust
region itself: if greedy-linearized performs nearly as well as SA-QUBO,
the annealer isn't doing much beyond the linearization; if SA-QUBO is
still clearly better, the (nonconvex, many-local-optima) combinatorial
search itself matters.

Randomness only enters through the SEED-DEPENDENT RANDOM INITIAL DESIGN
(mirroring the CS2 protocol's fixed FE-budget / mass-target / move-limit
setup, but using a random start per seed instead of CS2's fixed
mid-catalog start) -- the greedy solver itself is fully deterministic
given an initial design.
"""
import numpy as np
from frame3d import local_stiffness
from sizing import element_geom_cache, group_element_indices, evaluate_design
from sections import CHS_CATALOG, N_SECTIONS, E_STEEL, G_STEEL


def greedy_coordinate_sizing(model, mass_target, n_iters=50, w_stiff=1.0,
                              w_mass=5.0, move_limit=2, seed=0,
                              init_idx=None, load_per_node=-1000.0,
                              compliance_scale=1.0, patience=1000,
                              max_inner_sweeps=8, random_init=True,
                              check_cond=True):
    n_groups = model.n_groups
    n_sec = N_SECTIONS
    geom = element_geom_cache(model)
    groups = group_element_indices(model)
    rng = np.random.default_rng(seed)

    if init_idx is not None:
        cur = list(init_idx)
    elif random_init:
        cur = [int(rng.integers(0, n_sec)) for _ in range(n_groups)]
    else:
        cur = [n_sec // 2] * n_groups

    comp_hist, mass_hist, idx_hist, fe_calls = [], [], [], 0
    no_change = 0
    group_order = list(range(n_groups))  # fixed deterministic sweep order

    for it in range(n_iters):
        C, M, frm = evaluate_design(model, cur, load_per_node, check_cond=check_cond)
        fe_calls += 1
        comp_hist.append(C)
        mass_hist.append(M)
        idx_hist.append(list(cur))

        # frozen-field strain energy, identical computation to sizing.py
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

        SEmax = SE.max() if SE.max() > 0 else 1.0
        greedy_cur = list(cur)

        def total_mass_with(k, c):
            tot = 0.0
            for j in range(n_groups):
                cj = c if j == k else greedy_cur[j]
                tot += mass_coef[j, cj]
            return tot

        for sweep in range(max_inner_sweeps):
            changed = False
            for k in group_order:
                lo = max(0, cur[k] - move_limit)
                hi = min(n_sec - 1, cur[k] + move_limit)
                best_c, best_obj = greedy_cur[k], np.inf
                for c in range(lo, hi + 1):
                    tot_mass = total_mass_with(k, c)
                    rel = (tot_mass - mass_target) / mass_target
                    obj = -w_stiff * SE[k, c] / SEmax + w_mass * rel**2
                    if obj < best_obj - 1e-15:
                        best_obj = obj
                        best_c = c
                if best_c != greedy_cur[k]:
                    greedy_cur[k] = best_c
                    changed = True
            if not changed:
                break

        new_idx = greedy_cur
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


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from reciprocal_geom import square_grillage, model_mass, build_frame
    model = square_grillage(4, 4, cell=3.0, lam=0.25, n_rings=3)
    gs_max = {k: CHS_CATALOG[-1] for k in range(model.n_groups)}
    mass_max = model_mass(model, gs_max)
    mass_target = 0.6 * mass_max
    res = greedy_coordinate_sizing(model, mass_target, n_iters=50, w_mass=5.0,
                                    seed=0, load_per_node=-2000.0)
    print("final compliance:", res['compliance_history'][-1],
          "final mass:", res['mass_history'][-1],
          "n_iters_run:", res['n_iters_run'])

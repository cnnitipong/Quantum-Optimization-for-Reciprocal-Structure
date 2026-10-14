import os as _os_bootstrap
ROOT = _os_bootstrap.path.dirname(_os_bootstrap.path.dirname(_os_bootstrap.path.abspath(__file__)))
"""
RI (stretch) -- stress-constrained sizing QUBO on CS2 (square_nexorade).

Question under test: does joint SA-QUBO annealing beat greedy per-group
argmin decoding when MASS and STRESS constraints COUPLE in the same
discrete sizing objective (as opposed to the mass-only objective used
throughout CS2-CS6)?

Protocol
--------
1. Live-load scale: uniform mid-catalog CS2 (all 12 groups = CHS_CATALOG[4])
   at the nominal CS2 load (-450 N/node) has max per-element stress
   utilization 0.4038 (element 62). Since the FE solve is LINEAR in the
   applied nodal load, stress scales exactly linearly with load
   magnitude, so scale = 0.9 / 0.4038 = 2.229 gives max-utilization ~=0.9
   at the SAME uniform mid-catalog design. Scaled load_per_node =
   -450 * 2.229 = -1003.08 N/node. This is the load used for ALL of RI.

2. Stress-constrained F1 QUBO (SA-QUBO): identical outer loop to
   sizing.iterative_qubo_sizing (frozen-displacement strain-energy
   linearization + move-limit trust region + mass-target quadratic
   penalty + one-hot penalty), PLUS a per-group FROZEN-FORCE stress
   feasibility term: at each outer iteration, using the CURRENT real FE
   solve's actual member forces (N, My, Mz) HELD FIXED (a conservative
   simplification -- it ignores that switching a member's section would
   itself redistribute internal forces; freezing the force can only make
   a candidate section look WORSE than it truly would be, never better,
   hence "conservative"), compute utilization(c) = N/A_c + M*r_c/I_c,
   summed over My/Mz worst-case, divided by fy=355MPa, for every catalog
   section c=0..7. The smallest feasible c_min,g (utilization<=1) is
   recorded per group; every bit x_{g,c} with c < c_min,g gets a LINEAR
   QUBO penalty +w_stress (dominant, w_stress=500 vs SE terms O(1) and
   mass-penalty terms O(w_mass)=O(20)).
3. Greedy-linearized (ablation baseline): IDENTICAL per-candidate
   objective (same frozen SE reward + mass penalty + frozen-force stress
   threshold penalty) as the SA-QUBO above, but solved by deterministic
   coordinate-descent argmin instead of simulated annealing -- isolates
   the annealer's own contribution once mass AND stress compete.
4. GA baseline: standard genetic_algorithm (qubo.py) with fitness =
   compliance/compliance_scale + w_mass*rel_mass_err^2 + w_stress_ga *
   sum_g(hinge(util_g - 1))^2, where util_g is computed from GA's own
   REAL FE solve of the trial design (TRUE utilization, not a frozen-
   force estimate -- GA doesn't have an iterative frozen field to exploit,
   so it gets the more literal ground-truth penalty).

10 seeds/method, 50 real-FE budget/seed (same accounting convention as
every other case study: 1 outer iteration / 1 GA fitness call = 1 real FE
evaluation), w_mass=20.

Reporting: for SA-QUBO/greedy, the "final design" used for the post-hoc
report is each seed's BEST-COMPLIANCE iterate (argmin over that seed's
own compliance_history) -- the same "best_compliance" convention used
throughout summary_v2.md/cs2_v2.json etc, NOT necessarily literally the
last outer iteration (which is usually identical once the loop has
converged/plateaued). For GA, its own best_x/best_f output is used
(GA's natural best-of-run point). All three methods' final designs are
then RE-EVALUATED with one additional real FE solve to compute the
TRUE (non-frozen, non-hinge-approximated) post-hoc per-group utilization,
exactly as an reader would want -- this ground-truth check is
independent of whichever internal approximation each method's own search
used.

Resumable/checkpointed (per method,seed unit) -- mirrors
review_rb_ablation_cs6.py's pattern.
"""
import json
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(ROOT, "src"))
from reciprocal_geom_v2 import square_nexorade
from reciprocal_geom import build_frame, model_mass
from sections import CHS_CATALOG, N_SECTIONS, E_STEEL, G_STEEL
from frame3d import local_stiffness
from sizing import element_geom_cache, group_element_indices, evaluate_design
from qubo import QUBOBuilder, sa_solve, genetic_algorithm
from stats_utils import mann_whitney_u, cliffs_delta
from exact_mw import exact_mannwhitney

RESULTS = os.path.join(ROOT, "results")
REVIEW = f"{RESULTS}/review"
OUTFILE = f"{REVIEW}/ri_stress_qubo.json"

N_FE_BUDGET = 50
N_SEEDS = 10
W_MASS = 20.0
W_STRESS = 500.0
W_STRESS_GA = 500.0
FY = 355e6
MOVE_LIMIT = 2
BIG_PENALTY = 1e6
CHUNK_ITERS = 10
TIME_BUDGET_S = float(sys.argv[1]) if len(sys.argv) > 1 else 38.0


# --------------------------------------------------------------------
# setup / scale factor
# --------------------------------------------------------------------
def build_setup():
    model = square_nexorade(6, 6, 2.5, 0.25)
    load_nom = model.meta['load_per_node']
    gs_mid = {k: CHS_CATALOG[4] for k in range(model.n_groups)}
    frm = build_frame(model, gs_mid, load_per_node=load_nom)
    frm.solve()
    max_util = 0.0
    for ei, e in enumerate(model.elements):
        if e['kind'] != 'beam':
            continue
        sp = gs_mid[e['group']]
        r = sp['D'] / 2.0
        f = frm.element_end_forces_local(ei)
        N_abs = max(abs(f[0]), abs(f[6]))
        My_abs = max(abs(f[4]), abs(f[10]))
        Mz_abs = max(abs(f[5]), abs(f[11]))
        sigma = N_abs / sp['A'] + My_abs * r / sp['Iy'] + Mz_abs * r / sp['Iz']
        util = sigma / FY
        max_util = max(max_util, util)
    scale = 0.9 / max_util
    load_scaled = load_nom * scale

    gs_max = {k: CHS_CATALOG[-1] for k in range(model.n_groups)}
    mass_max = model_mass(model, gs_max)
    mass_target = 0.6 * mass_max
    frm_scale = build_frame(model, gs_mid, load_per_node=load_scaled)
    frm_scale.solve()
    compliance_scale = frm_scale.compliance()

    return dict(model=model, load_nom=load_nom, max_util_at_nom=max_util,
                scale_factor=scale, load_scaled=load_scaled,
                mass_target=mass_target, mass_max=mass_max,
                compliance_scale=compliance_scale)


# --------------------------------------------------------------------
# shared stress-utilization helpers (element-level, per group)
# --------------------------------------------------------------------
def group_forces(model, frm, groups):
    """{k: [(N_abs, My_abs, Mz_abs), ...]} per element in group k, from a
    real FE solve `frm`."""
    out = {}
    for k, elem_idxs in groups.items():
        vals = []
        for ei in elem_idxs:
            f = frm.element_end_forces_local(ei)
            N_abs = max(abs(f[0]), abs(f[6]))
            My_abs = max(abs(f[4]), abs(f[10]))
            Mz_abs = max(abs(f[5]), abs(f[11]))
            vals.append((N_abs, My_abs, Mz_abs))
        out[k] = vals
    return out


def util_of(sp, N_abs, My_abs, Mz_abs):
    r = sp['D'] / 2.0
    sigma = N_abs / sp['A'] + My_abs * r / sp['Iy'] + Mz_abs * r / sp['Iz']
    return sigma / FY


def frozen_force_cmin(gforces, n_groups, n_sec=N_SECTIONS):
    """Per group: utilization(c) for c=0..n_sec-1 using FROZEN (fixed)
    forces from the current real FE solve; c_min,g = smallest feasible c
    (n_sec if none feasible)."""
    util_table = np.zeros((n_groups, n_sec))
    for k in range(n_groups):
        for (N_abs, My_abs, Mz_abs) in gforces[k]:
            for c in range(n_sec):
                u = util_of(CHS_CATALOG[c], N_abs, My_abs, Mz_abs)
                util_table[k, c] = max(util_table[k, c], u)
    c_min = np.full(n_groups, n_sec, dtype=int)
    for k in range(n_groups):
        feas = np.where(util_table[k] <= 1.0)[0]
        if len(feas):
            c_min[k] = int(feas.min())
    return c_min, util_table


def real_group_utils(model, gs, frm, groups):
    """TRUE per-group utilization at the design's OWN currently assigned
    section (ground-truth post-hoc check, no frozen-force approximation)."""
    n_groups = model.n_groups
    out = np.zeros(n_groups)
    for k, elem_idxs in groups.items():
        sp = gs[k]
        for ei in elem_idxs:
            f = frm.element_end_forces_local(ei)
            N_abs = max(abs(f[0]), abs(f[6]))
            My_abs = max(abs(f[4]), abs(f[10]))
            Mz_abs = max(abs(f[5]), abs(f[11]))
            out[k] = max(out[k], util_of(sp, N_abs, My_abs, Mz_abs))
    return out


def posthoc_report(model, idx_vec, load_scaled, groups):
    gs = {k: CHS_CATALOG[idx_vec[k]] for k in range(model.n_groups)}
    frm = build_frame(model, gs, load_per_node=load_scaled)
    frm.solve()
    C = frm.compliance()
    M = model_mass(model, gs)
    utils = real_group_utils(model, gs, frm, groups)
    return dict(compliance=float(C), mass=float(M),
                max_utilization=float(utils.max()),
                mean_utilization=float(utils.mean()),
                feasible=bool(utils.max() <= 1.0),
                per_group_utilization=utils.tolist())


# --------------------------------------------------------------------
# SA-QUBO stress-constrained sizing (one chunk of iterations)
# --------------------------------------------------------------------
def sa_stress_chunk(model, mass_target, compliance_scale, load_scaled,
                     geom, groups, seed, state, chunk_iters):
    n_groups = model.n_groups
    n_sec = N_SECTIONS
    remaining = N_FE_BUDGET - len(state['comp_hist'])
    if remaining <= 0:
        state['done'] = True
        return state
    this_chunk = min(chunk_iters, remaining)
    cur = list(state['cur_idx'])
    n_done0 = len(state['comp_hist'])

    for local_it in range(this_chunk):
        it = n_done0 + local_it
        C, M, frm = evaluate_design(model, cur, load_scaled)
        state['comp_hist'].append(float(C))
        state['mass_hist'].append(float(M))
        state['idx_hist'].append(list(cur))

        gforces = group_forces(model, frm, groups)
        c_min, _ = frozen_force_cmin(gforces, n_groups, n_sec)

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

        qb = QUBOBuilder(n_groups * n_sec)

        def vidx(k, c):
            return k * n_sec + c

        SEmax = SE.max() if SE.max() > 0 else 1.0
        for k in range(n_groups):
            lo, hi = max(0, cur[k] - MOVE_LIMIT), min(n_sec - 1, cur[k] + MOVE_LIMIT)
            for c in range(n_sec):
                coef = 0.0
                if lo <= c <= hi:
                    coef += -SE[k, c] / SEmax
                else:
                    coef += BIG_PENALTY
                if c < c_min[k]:
                    coef += W_STRESS
                qb.add_linear(vidx(k, c), coef)

        all_idx = [vidx(k, c) for k in range(n_groups) for c in range(n_sec)]
        all_coef = [mass_coef[k, c] for k in range(n_groups) for c in range(n_sec)]
        qb.add_target_penalty(all_idx, all_coef, mass_target, W_MASS / max(mass_target**2, 1e-9))

        w_onehot = 3.0 * (1.0 + W_MASS)
        for k in range(n_groups):
            qb.add_onehot_penalty([vidx(k, c) for c in range(n_sec)], w_onehot)

        sa = sa_solve(qb.Q, n_reads=25, n_sweeps=90, seed=seed * 10000 + it)
        Xb = sa['best_x'].reshape(n_groups, n_sec)
        cur = [int(x) for x in Xb.argmax(axis=1)]

    state['cur_idx'] = cur
    if len(state['comp_hist']) >= N_FE_BUDGET:
        state['done'] = True
    return state


# --------------------------------------------------------------------
# greedy-linearized stress-constrained sizing (SAME per-candidate
# objective as SA-QUBO above, coordinate-descent argmin instead of SA)
# --------------------------------------------------------------------
def greedy_stress_chunk(model, mass_target, load_scaled, geom, groups,
                         state, chunk_iters, max_inner_sweeps=8):
    n_groups = model.n_groups
    n_sec = N_SECTIONS
    remaining = N_FE_BUDGET - len(state['comp_hist'])
    if remaining <= 0:
        state['done'] = True
        return state
    this_chunk = min(chunk_iters, remaining)
    cur = list(state['cur_idx'])
    n_done0 = len(state['comp_hist'])

    for local_it in range(this_chunk):
        it = n_done0 + local_it
        C, M, frm = evaluate_design(model, cur, load_scaled)
        state['comp_hist'].append(float(C))
        state['mass_hist'].append(float(M))
        state['idx_hist'].append(list(cur))

        gforces = group_forces(model, frm, groups)
        c_min, _ = frozen_force_cmin(gforces, n_groups, n_sec)

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
            for k in range(n_groups):
                lo = max(0, cur[k] - MOVE_LIMIT)
                hi = min(n_sec - 1, cur[k] + MOVE_LIMIT)
                best_c, best_obj = greedy_cur[k], np.inf
                for c in range(lo, hi + 1):
                    tot_mass = total_mass_with(k, c)
                    rel = (tot_mass - mass_target) / mass_target
                    obj = -SE[k, c] / SEmax + W_MASS * rel**2
                    if c < c_min[k]:
                        obj += W_STRESS
                    if obj < best_obj - 1e-15:
                        best_obj = obj
                        best_c = c
                if best_c != greedy_cur[k]:
                    greedy_cur[k] = best_c
                    changed = True
            if not changed:
                break
        cur = greedy_cur

    state['cur_idx'] = cur
    if len(state['comp_hist']) >= N_FE_BUDGET:
        state['done'] = True
    return state


# --------------------------------------------------------------------
# GA with mass+stress-penalized fitness (real per-trial FE + true util)
# --------------------------------------------------------------------
def run_ga_stress(model, mass_target, compliance_scale, load_scaled, groups, seed):
    log = {'C': [], 'M': [], 'obj': [], 'idx': []}

    def obj(idx_vec):
        try:
            C, M, frm = evaluate_design(model, idx_vec, load_scaled)
        except np.linalg.LinAlgError:
            log['C'].append(1e9); log['M'].append(1e9); log['obj'].append(1e9); log['idx'].append(list(idx_vec))
            return 1e9
        gs = {k: CHS_CATALOG[idx_vec[k]] for k in range(model.n_groups)}
        utils = real_group_utils(model, gs, frm, groups)
        hinge = np.maximum(0.0, utils - 1.0)
        stress_pen = W_STRESS_GA * float(np.sum(hinge**2))
        rel = (M - mass_target) / mass_target
        f = C / compliance_scale + W_MASS * rel**2 + stress_pen
        log['C'].append(float(C)); log['M'].append(float(M)); log['obj'].append(float(f))
        log['idx'].append(list(idx_vec))
        return f

    t0 = time.time()
    res = genetic_algorithm(obj, [N_SECTIONS] * model.n_groups, n_evals=N_FE_BUDGET,
                             pop_size=20, seed=seed)
    dt = time.time() - t0
    best_i = int(np.argmin(log['obj']))
    return dict(seed=seed, time_s=dt, best_idx=log['idx'][best_i],
                best_obj=log['obj'][best_i], best_compliance=log['C'][best_i],
                best_mass=log['M'][best_i], obj_hist=log['obj'], C_hist=log['C'],
                M_hist=log['M'])


# --------------------------------------------------------------------
# checkpoint machinery
# --------------------------------------------------------------------
def load_progress():
    if os.path.exists(OUTFILE):
        try:
            with open(OUTFILE) as f:
                d = json.load(f)
                if d and 'setup' in d:
                    return d
        except Exception:
            pass
    return None


def save(prog):
    tmp = OUTFILE + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(prog, f)
    os.replace(tmp, OUTFILE)


def main():
    setup = build_setup()
    model = setup['model']
    geom = element_geom_cache(model)
    groups = group_element_indices(model)

    prog = load_progress()
    if prog is None:
        prog = dict(
            setup=dict(load_nom=setup['load_nom'], max_util_at_nom=setup['max_util_at_nom'],
                       scale_factor=setup['scale_factor'], load_scaled=setup['load_scaled'],
                       mass_target=setup['mass_target'], mass_max=setup['mass_max'],
                       compliance_scale=setup['compliance_scale'], w_mass=W_MASS,
                       w_stress=W_STRESS, w_stress_ga=W_STRESS_GA, n_fe_budget=N_FE_BUDGET,
                       n_seeds=N_SEEDS, move_limit=MOVE_LIMIT, fy_Pa=FY),
            sa_states={}, greedy_states={}, ga_results={}, posthoc={}, done_flags={})
        for s in range(N_SEEDS):
            init = [N_SECTIONS // 2] * model.n_groups
            prog['sa_states'][str(s)] = dict(cur_idx=list(init), comp_hist=[], mass_hist=[], idx_hist=[], done=False)
            prog['greedy_states'][str(s)] = dict(cur_idx=list(init), comp_hist=[], mass_hist=[], idx_hist=[], done=False)
        save(prog)

    load_scaled = prog['setup']['load_scaled']
    mass_target = prog['setup']['mass_target']
    compliance_scale = prog['setup']['compliance_scale']

    t_start = time.time()

    # SA states
    for s in range(N_SEEDS):
        key = str(s)
        st = prog['sa_states'][key]
        while not st['done']:
            if time.time() - t_start > TIME_BUDGET_S:
                save(prog); print_status(prog); return
            st = sa_stress_chunk(model, mass_target, compliance_scale, load_scaled,
                                  geom, groups, s, st, CHUNK_ITERS)
            prog['sa_states'][key] = st
            save(prog)
            print(f"[sa seed={s}] {len(st['comp_hist'])}/{N_FE_BUDGET} ({time.time()-t_start:.1f}s)")

    # greedy states
    for s in range(N_SEEDS):
        key = str(s)
        st = prog['greedy_states'][key]
        while not st['done']:
            if time.time() - t_start > TIME_BUDGET_S:
                save(prog); print_status(prog); return
            st = greedy_stress_chunk(model, mass_target, load_scaled, geom, groups,
                                      st, CHUNK_ITERS)
            prog['greedy_states'][key] = st
            save(prog)
            print(f"[greedy seed={s}] {len(st['comp_hist'])}/{N_FE_BUDGET} ({time.time()-t_start:.1f}s)")

    # GA (one seed = one atomic unit, ~50 real FE evals)
    for s in range(N_SEEDS):
        key = str(s)
        if key in prog['ga_results']:
            continue
        if time.time() - t_start > TIME_BUDGET_S:
            save(prog); print_status(prog); return
        r = run_ga_stress(model, mass_target, compliance_scale, load_scaled, groups, s)
        prog['ga_results'][key] = r
        save(prog)
        print(f"[ga seed={s}] best_C={r['best_compliance']:.3f} best_M={r['best_mass']:.1f} ({time.time()-t_start:.1f}s)")

    # post-hoc report (cheap, one FE solve per seed per method)
    for s in range(N_SEEDS):
        key = str(s)
        if key in prog['posthoc']:
            continue
        if time.time() - t_start > TIME_BUDGET_S:
            save(prog); print_status(prog); return
        sa_hist = np.array(prog['sa_states'][key]['comp_hist'])
        sa_best_i = int(np.argmin(sa_hist))
        sa_idx = prog['sa_states'][key]['idx_hist'][sa_best_i]

        gr_hist = np.array(prog['greedy_states'][key]['comp_hist'])
        gr_best_i = int(np.argmin(gr_hist))
        gr_idx = prog['greedy_states'][key]['idx_hist'][gr_best_i]

        ga_idx = prog['ga_results'][key]['best_idx']

        prog['posthoc'][key] = dict(
            sa=posthoc_report(model, sa_idx, load_scaled, groups),
            greedy=posthoc_report(model, gr_idx, load_scaled, groups),
            ga=posthoc_report(model, ga_idx, load_scaled, groups),
        )
        save(prog)
        print(f"[posthoc seed={s}] sa_util={prog['posthoc'][key]['sa']['max_utilization']:.3f} "
              f"greedy_util={prog['posthoc'][key]['greedy']['max_utilization']:.3f} "
              f"ga_util={prog['posthoc'][key]['ga']['max_utilization']:.3f}")

    n_sa_done = sum(1 for st in prog['sa_states'].values() if st['done'])
    n_gr_done = sum(1 for st in prog['greedy_states'].values() if st['done'])
    n_ga_done = len(prog['ga_results'])
    n_ph_done = len(prog['posthoc'])
    if n_sa_done < N_SEEDS or n_gr_done < N_SEEDS or n_ga_done < N_SEEDS or n_ph_done < N_SEEDS:
        print_status(prog)
        return

    # ---- finalize summary ----
    sa_C = [prog['posthoc'][str(s)]['sa']['compliance'] for s in range(N_SEEDS)]
    gr_C = [prog['posthoc'][str(s)]['greedy']['compliance'] for s in range(N_SEEDS)]
    ga_C = [prog['posthoc'][str(s)]['ga']['compliance'] for s in range(N_SEEDS)]
    sa_M = [prog['posthoc'][str(s)]['sa']['mass'] for s in range(N_SEEDS)]
    gr_M = [prog['posthoc'][str(s)]['greedy']['mass'] for s in range(N_SEEDS)]
    ga_M = [prog['posthoc'][str(s)]['ga']['mass'] for s in range(N_SEEDS)]
    sa_U = [prog['posthoc'][str(s)]['sa']['max_utilization'] for s in range(N_SEEDS)]
    gr_U = [prog['posthoc'][str(s)]['greedy']['max_utilization'] for s in range(N_SEEDS)]
    ga_U = [prog['posthoc'][str(s)]['ga']['max_utilization'] for s in range(N_SEEDS)]
    sa_feas = [prog['posthoc'][str(s)]['sa']['feasible'] for s in range(N_SEEDS)]
    gr_feas = [prog['posthoc'][str(s)]['greedy']['feasible'] for s in range(N_SEEDS)]
    ga_feas = [prog['posthoc'][str(s)]['ga']['feasible'] for s in range(N_SEEDS)]

    def mass_err_pct(Ms):
        return [100.0 * (m - mass_target) / mass_target for m in Ms]

    mw_approx_sa_gr = mann_whitney_u(sa_C, gr_C)
    mw_exact_sa_gr = exact_mannwhitney(sa_C, gr_C)
    delta_sa_gr = cliffs_delta(sa_C, gr_C)
    mw_approx_sa_ga = mann_whitney_u(sa_C, ga_C)
    mw_exact_sa_ga = exact_mannwhitney(sa_C, ga_C)
    delta_sa_ga = cliffs_delta(sa_C, ga_C)

    summary = dict(
        sa=dict(compliance_mean=float(np.mean(sa_C)), compliance_std=float(np.std(sa_C)),
                mass_mean=float(np.mean(sa_M)), mass_std=float(np.std(sa_M)),
                mass_err_pct_mean=float(np.mean(mass_err_pct(sa_M))),
                max_util_mean=float(np.mean(sa_U)), max_util_max=float(np.max(sa_U)),
                feasibility_rate=float(np.mean(sa_feas))),
        greedy=dict(compliance_mean=float(np.mean(gr_C)), compliance_std=float(np.std(gr_C)),
                    mass_mean=float(np.mean(gr_M)), mass_std=float(np.std(gr_M)),
                    mass_err_pct_mean=float(np.mean(mass_err_pct(gr_M))),
                    max_util_mean=float(np.mean(gr_U)), max_util_max=float(np.max(gr_U)),
                    feasibility_rate=float(np.mean(gr_feas))),
        ga=dict(compliance_mean=float(np.mean(ga_C)), compliance_std=float(np.std(ga_C)),
                mass_mean=float(np.mean(ga_M)), mass_std=float(np.std(ga_M)),
                mass_err_pct_mean=float(np.mean(mass_err_pct(ga_M))),
                max_util_mean=float(np.mean(ga_U)), max_util_max=float(np.max(ga_U)),
                feasibility_rate=float(np.mean(ga_feas))),
        sa_vs_greedy=dict(pct_advantage=float(100.0 * (np.mean(gr_C) - np.mean(sa_C)) / np.mean(gr_C)),
                          mw_normal_approx_p=mw_approx_sa_gr['p'], mw_exact_p=mw_exact_sa_gr['p_exact_two_sided'],
                          cliffs_delta=delta_sa_gr),
        sa_vs_ga=dict(pct_advantage=float(100.0 * (np.mean(ga_C) - np.mean(sa_C)) / np.mean(ga_C)),
                      mw_normal_approx_p=mw_approx_sa_ga['p'], mw_exact_p=mw_exact_sa_ga['p_exact_two_sided'],
                      cliffs_delta=delta_sa_ga),
    )
    prog['summary'] = summary
    prog['done'] = True
    save(prog)
    print("\n=== RI SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print("ALL DONE")


def print_status(prog):
    n_sa = sum(1 for st in prog['sa_states'].values() if st['done'])
    n_gr = sum(1 for st in prog['greedy_states'].values() if st['done'])
    n_ga = len(prog['ga_results'])
    n_ph = len(prog['posthoc'])
    print(f"  sa: {n_sa}/{N_SEEDS}  greedy: {n_gr}/{N_SEEDS}  ga: {n_ga}/{N_SEEDS}  posthoc: {n_ph}/{N_SEEDS}")


if __name__ == "__main__":
    main()

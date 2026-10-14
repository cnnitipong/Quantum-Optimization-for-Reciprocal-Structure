"""sizing_benchmark.py -- sizing experiments .

For each sizing case study, on the SAME paired random initial design per seed
as the greedy-linearised ablation (random_init(seed)), run:
  sa_reduced_w5 / sa_reduced_w20 : F1 loop with the move-limit window applied
        by VARIABLE ELIMINATION (only in-window bits enter the QUBO) and the
        SA start temperature taken from the reduced matrix, 25 reads x 90 sweeps.
  sa_strong_w5  : original F1 QUBO (penalty exclusion) at 100 reads x 400 sweeps.
  ls_w5         : FE-evaluated first-improvement local search (+-1 catalogue
        step per move, random group order), real penalised objective.
  ga8_w5        : GA with population 8 (6 generations in 50 evaluations).
Usage: python3 sizing_benchmark.py <case> [n_seeds]
case in cs2, cs3, cs6, dome, vault.  Output results/benchmark/<case>.json
(checkpointed per (method, seed)).
"""
import os, sys, json, time
import os as _os_bootstrap
ROOT = _os_bootstrap.path.dirname(_os_bootstrap.path.dirname(_os_bootstrap.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
import numpy as np
from frame3d import local_stiffness
from reciprocal_geom import build_frame, model_mass
from sections import CHS_CATALOG, N_SECTIONS, E_STEEL, G_STEEL
from qubo import QUBOBuilder, sa_solve, genetic_algorithm
from sizing import evaluate_design, element_geom_cache, group_element_indices, iterative_qubo_sizing

OUT = os.path.join(ROOT, 'results', 'benchmark')
os.makedirs(OUT, exist_ok=True)
N_FE = 50
MASS_FRAC = 0.6


def make_case(case):
    if case == 'cs2':
        from reciprocal_geom_v2 import square_nexorade
        m = square_nexorade(6, 6, 2.5, 0.25); load = -450.0
    elif case == 'cs3':
        from reciprocal_geom_v2 import hex_nexorade
        m = hex_nexorade(3, 2.2, 0.25); load = -1400.0
    elif case == 'cs6':
        from reciprocal_geom_v2 import davinci_bridge2
        m = davinci_bridge2(n_bays=24, span=20.0, width=6.0, lam=0.25); load = m.meta['load_per_node']
    elif case == 'dome':
        import dome2_sizing; m = dome2_sizing.make_model(); load = m.meta['load_per_node']
    elif case == 'vault':
        import vault_hex2_sizing; m = vault_hex2_sizing.make_model(); load = m.meta['load_per_node']
    else:
        raise ValueError(case)
    return m, load


def random_init(seed, n_groups):
    rng = np.random.default_rng(seed)
    return [int(rng.integers(0, N_SECTIONS)) for _ in range(n_groups)]


def linearise(model, geom, groups, frm):
    n_groups, n_sec = model.n_groups, N_SECTIONS
    SE = np.zeros((n_groups, n_sec)); MC = np.zeros((n_groups, n_sec))
    for k, elem_idxs in groups.items():
        for ei in elem_idxs:
            g = geom[ei]
            u_local = g['T'] @ frm.U[g['dofs']]
            for c in range(n_sec):
                sp = CHS_CATALOG[c]
                kl = local_stiffness(E_STEEL, G_STEEL, sp['A'], sp['Iy'], sp['Iz'], sp['J'], g['L'])
                SE[k, c] += 0.5 * u_local @ kl @ u_local
                MC[k, c] += sp['mass_per_len'] * g['L']
    return SE, MC


def sizing_reduced(model, mass_target, load, w_mass, seed, init_idx, n_iters=N_FE,
                   move_limit=2, sa_reads=25, sa_sweeps=90, w_stiff=1.0):
    """F1 loop with variable elimination of out-of-window sections."""
    n_groups, n_sec = model.n_groups, N_SECTIONS
    geom = element_geom_cache(model); groups = group_element_indices(model)
    cur = list(init_idx); C_h, M_h, X_h, nbits_h = [], [], [], []
    for it in range(n_iters):
        C, M, frm = evaluate_design(model, cur, load, check_cond=False)
        C_h.append(C); M_h.append(M); X_h.append(list(cur))
        SE, MC = linearise(model, geom, groups, frm)
        SEmax = SE.max() if SE.max() > 0 else 1.0
        cand = [list(range(max(0, cur[k] - move_limit), min(n_sec - 1, cur[k] + move_limit) + 1)) for k in range(n_groups)]
        vid = {}; idx = 0
        for k in range(n_groups):
            for c in cand[k]:
                vid[(k, c)] = idx; idx += 1
        nb = idx; qb = QUBOBuilder(nb)
        for (k, c), i in vid.items():
            qb.add_linear(i, -w_stiff * SE[k, c] / SEmax)
        all_idx = [vid[(k, c)] for k in range(n_groups) for c in cand[k]]
        all_coef = [MC[k, c] for k in range(n_groups) for c in cand[k]]
        qb.add_target_penalty(all_idx, all_coef, mass_target, w_mass / max(mass_target ** 2, 1e-9))
        w_oh = 3.0 * (w_stiff + w_mass)
        for k in range(n_groups):
            qb.add_onehot_penalty([vid[(k, c)] for c in cand[k]], w_oh)
        sa = sa_solve(qb.Q, n_reads=sa_reads, n_sweeps=sa_sweeps, seed=seed * 10000 + it)
        x = sa['best_x']; new = []
        for k in range(n_groups):
            vals = [(x[vid[(k, c)]], c) for c in cand[k]]
            new.append(max(vals)[1])
        nbits_h.append(nb); cur = new
    return dict(compliance_history=C_h, mass_history=M_h, idx_history=X_h, final_idx=cur,
                n_bits=int(np.mean(nbits_h)))


def sizing_block(model, mass_target, load, w_mass, seed, init_idx, n_iters=N_FE,
                 move_limit=2, sa_reads=25, sa_sweeps=90, w_stiff=1.0):
    """F1 loop, QUBO identical to sizing.py (penalty exclusion, one-hot penalty),
    solved by the one-hot-aware block annealer of blocksa.py."""
    from blocksa import block_sa_solve
    from sizing import BIG_PENALTY
    n_groups, n_sec = model.n_groups, N_SECTIONS
    geom = element_geom_cache(model); groups = group_element_indices(model)
    blocks = [[k * n_sec + c for c in range(n_sec)] for k in range(n_groups)]
    cur = list(init_idx); C_h, M_h, X_h, E_h = [], [], [], []
    for it in range(n_iters):
        C, M, frm = evaluate_design(model, cur, load, check_cond=False)
        C_h.append(C); M_h.append(M); X_h.append(list(cur))
        SE, MC = linearise(model, geom, groups, frm)
        SEmax = SE.max() if SE.max() > 0 else 1.0
        qb = QUBOBuilder(n_groups * n_sec)
        for k in range(n_groups):
            lo, hi = max(0, cur[k] - move_limit), min(n_sec - 1, cur[k] + move_limit)
            for c in range(n_sec):
                qb.add_linear(k * n_sec + c, -w_stiff * SE[k, c] / SEmax if lo <= c <= hi else BIG_PENALTY)
        qb.add_target_penalty(list(range(n_groups * n_sec)), [MC[k, c] for k in range(n_groups) for c in range(n_sec)],
                              mass_target, w_mass / max(mass_target ** 2, 1e-9))
        for k in range(n_groups):
            qb.add_onehot_penalty([k * n_sec + c for c in range(n_sec)], 3.0 * (w_stiff + w_mass))
        sa = block_sa_solve(qb.Q, blocks, n_reads=sa_reads, n_sweeps=sa_sweeps, seed=seed * 10000 + it)
        cur = [int(v) for v in sa['best_x'].reshape(n_groups, n_sec).argmax(axis=1)]
        E_h.append(float(sa['best_energy']))
    return dict(compliance_history=C_h, mass_history=M_h, idx_history=X_h, final_idx=cur, qubo_energy_history=E_h)


def penalised(model, mass_target, load, w_mass, Cscale, log):
    def f(idx):
        try:
            C, M, _ = evaluate_design(model, idx, load, check_cond=False)
        except np.linalg.LinAlgError:
            C, M = 1e9, 1e9
        rel = (M - mass_target) / mass_target
        v = C / Cscale + w_mass * rel ** 2
        log['C'].append(C); log['M'].append(M); log['x'].append(list(idx)); log['f'].append(v)
        return v
    return f


def local_search(model, mass_target, load, w_mass, Cscale, seed, init_idx, budget=N_FE):
    rng = np.random.default_rng(seed + 991)
    log = dict(C=[], M=[], x=[], f=[]); f = penalised(model, mass_target, load, w_mass, Cscale, log)
    cur = list(init_idx); fcur = f(cur); n = model.n_groups
    while len(log['C']) < budget:
        improved = False
        moves = [(k, d) for k in range(n) for d in (-1, +1)]
        rng.shuffle(moves)
        for k, d in moves:
            if len(log['C']) >= budget:
                break
            c = cur[k] + d
            if c < 0 or c >= N_SECTIONS:
                continue
            trial = list(cur); trial[k] = c
            ft = f(trial)
            if ft < fcur - 1e-12:
                cur, fcur, improved = trial, ft, True
                break
        if not improved and len(log['C']) < budget:
            # random restart kick: perturb two groups
            trial = list(cur)
            for k in rng.choice(n, size=min(2, n), replace=False):
                trial[k] = int(np.clip(trial[k] + rng.choice([-2, -1, 1, 2]), 0, N_SECTIONS - 1))
            ft = f(trial)
            if ft < fcur:
                cur, fcur = trial, ft
    return finish(log)


def ga8(model, mass_target, load, w_mass, Cscale, seed, budget=N_FE):
    log = dict(C=[], M=[], x=[], f=[]); f = penalised(model, mass_target, load, w_mass, Cscale, log)
    genetic_algorithm(f, [N_SECTIONS] * model.n_groups, n_evals=budget, pop_size=8, seed=seed)
    return finish(log)


def finish(log):
    bi = int(np.argmin(log['f']))
    return dict(compliance_history=log['C'], mass_history=log['M'], idx_history=log['x'],
                final_idx=log['x'][bi], best_obj_C=log['C'][bi], best_obj_M=log['M'][bi])


def run(case, n_seeds):
    model, load = make_case(case)
    gs_max = {k: CHS_CATALOG[-1] for k in range(model.n_groups)}
    mass_max = model_mass(model, gs_max); mass_target = MASS_FRAC * mass_max
    gs_mid = {k: CHS_CATALOG[3] for k in range(model.n_groups)}
    frm = build_frame(model, gs_mid, load_per_node=load); frm.solve(check_cond=False)
    Cscale = frm.compliance()
    out_path = os.path.join(OUT, f'{case}.json')
    res = json.load(open(out_path)) if os.path.exists(out_path) else dict(
        case=case, mass_target=mass_target, mass_max=mass_max, compliance_scale=Cscale,
        load_per_node=load, n_groups=model.n_groups, n_fe_budget=N_FE, methods={})
    methods = [('sa_block_w5', 5.0), ('sa_block_w20', 20.0), ('ls_w5', 5.0), ('ga8_w5', 5.0)]
    if case == 'cs6':
        methods.append(('sa_strong_w5', 5.0))
    if os.environ.get('ARM_W100'):
        methods = [('sa_block_w100', 100.0), ('greedy_w100', 100.0), ('sa_single_w100', 100.0)]
    if os.environ.get('ARM_W1000'):
        methods = [('sa_block_w1000', 1000.0), ('greedy_w1000', 1000.0), ('sa_single_w1000', 1000.0)]
    if os.environ.get('ARM_LINKFIX'):
        methods = [('sa_block_w1000f', 1000.0), ('greedy_w1000f', 1000.0), ('sa_single_w1000f', 1000.0)]
    # connector (link) mass: the per-iteration QUBO sums beam-element mass only, so a target
    # expressed on the TOTAL mass must be reduced by the constant link mass (arms ending in 'f')
    from reciprocal_geom import LINK_SECTION
    nodes = np.array(model.nodes)
    link_mass = sum(LINK_SECTION['mass_per_len'] * np.linalg.norm(nodes[e['i']] - nodes[e['j']]) for e in model.elements if e['kind'] == 'link')
    res['link_mass_kg'] = float(link_mass)
    for meth, w in methods:
        res['methods'].setdefault(meth, [])
        for s in range(len(res['methods'][meth]), n_seeds):
            t0 = time.time(); init = random_init(s, model.n_groups)
            mt_q = mass_target - link_mass if meth.endswith('f') else mass_target
            if meth.startswith('sa_block'):
                r = sizing_block(model, mt_q, load, w, s, init)
            elif meth == 'sa_strong_w5':
                r = iterative_qubo_sizing(model, mass_target, n_iters=N_FE, w_mass=w, seed=s,
                                          init_idx=init, load_per_node=load, sa_reads=100,
                                          sa_sweeps=400, patience=1000, check_cond=False)
                r = dict(compliance_history=list(map(float, r['compliance_history'])),
                         mass_history=list(map(float, r['mass_history'])),
                         idx_history=[list(map(int, x)) for x in r['idx_history']],
                         final_idx=[int(x) for x in r['final_idx']])
            elif meth.startswith('greedy'):
                from greedy_sizing import greedy_coordinate_sizing
                r = greedy_coordinate_sizing(model, mt_q, n_iters=N_FE, w_mass=w, seed=s, init_idx=init,
                                             load_per_node=load, patience=1000, check_cond=False)
                r = dict(compliance_history=list(map(float, r['compliance_history'])),
                         mass_history=list(map(float, r['mass_history'])),
                         idx_history=[list(map(int, x)) for x in r['idx_history']],
                         final_idx=[int(x) for x in r['final_idx']])
            elif meth.startswith('sa_single'):
                r = iterative_qubo_sizing(model, mt_q, n_iters=N_FE, w_mass=w, seed=s,
                                          init_idx=init, load_per_node=load, sa_reads=25,
                                          sa_sweeps=90, patience=1000, check_cond=False)
                r = dict(compliance_history=list(map(float, r['compliance_history'])),
                         mass_history=list(map(float, r['mass_history'])),
                         idx_history=[list(map(int, x)) for x in r['idx_history']],
                         final_idx=[int(x) for x in r['final_idx']])
            elif meth == 'ls_w5':
                r = local_search(model, mass_target, load, w, Cscale, s, init)
            else:
                r = ga8(model, mass_target, load, w, Cscale, s)
            r['seed'] = s; r['time_s'] = time.time() - t0; r['init_idx'] = init
            r['best_compliance'] = float(min(r['compliance_history']))
            res['methods'][meth].append(r)
            json.dump(res, open(out_path, 'w'))
            print(f'{case} {meth} seed {s}: bestC {r["best_compliance"]:.2f} finalM {r["mass_history"][-1]:.1f} t {r["time_s"]:.1f}s', flush=True)
    print('ALL DONE', case)


if __name__ == '__main__':
    run(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 10)

"""bridge_two_load_cases.py -- two-load-case F1 sizing of the da Vinci bridge (full-span live load and
half-span live load), block annealer and greedy decode at w_mass = 100 (matched mass),
paired random initial designs; post-hoc evaluation of the resulting designs and of the
single-load-case w100 designs under self-weight + half-span live load.
Also writes the barrier diagnosis (QUBO energy per iteration for four solvers along the
greedy trajectory, seed 0, w_mass = 5)."""
import os, sys, json, copy
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src'))
import numpy as np
from reciprocal_geom import build_frame, model_mass
from reciprocal_geom_v2 import davinci_bridge2
from sections import CHS_CATALOG, N_SECTIONS
from qubo import QUBOBuilder, sa_solve
from sizing import evaluate_design, element_geom_cache, group_element_indices, BIG_PENALTY
from blocksa import block_sa_solve
from sizing_benchmark import random_init, linearise
from load_case_checks import frame, max_util

OUT = os.path.join(ROOT, 'results', 'benchmark'); os.makedirs(OUT, exist_ok=True)
N_FE = 50


def half_model(model):
    m = copy.copy(model); xs = np.array(model.nodes)[:, 0]; mid = 0.5 * (xs.min() + xs.max())
    m.load_idx = [n for n in model.load_idx if model.nodes[n][0] < mid]
    return m


def build_Q(model, cur, SE, MC, mass_target, w_mass, move_limit=2, w_stiff=1.0):
    n, ns = model.n_groups, N_SECTIONS; qb = QUBOBuilder(n * ns)
    SEmax = SE.max() if SE.max() > 0 else 1.0
    for k in range(n):
        lo, hi = max(0, cur[k] - move_limit), min(ns - 1, cur[k] + move_limit)
        for c in range(ns):
            qb.add_linear(k * ns + c, -w_stiff * SE[k, c] / SEmax if lo <= c <= hi else BIG_PENALTY)
    qb.add_target_penalty(list(range(n * ns)), [MC[k, c] for k in range(n) for c in range(ns)], mass_target, w_mass / mass_target ** 2)
    for k in range(n):
        qb.add_onehot_penalty([k * ns + c for c in range(ns)], 3.0 * (w_stiff + w_mass))
    return qb


def greedy_decode(qb, cur, SE, MC, mass_target, w_mass, n, move_limit=2, w_stiff=1.0):
    SEmax = SE.max() if SE.max() > 0 else 1.0; g = list(cur)
    for sweep in range(8):
        ch = False
        for k in range(n):
            lo, hi = max(0, cur[k] - move_limit), min(N_SECTIONS - 1, cur[k] + move_limit); best = None
            for c in range(lo, hi + 1):
                tot = sum(MC[j, c if j == k else g[j]] for j in range(n)); rel = (tot - mass_target) / mass_target
                o = -w_stiff * SE[k, c] / SEmax + w_mass * rel ** 2
                if best is None or o < best[0] - 1e-15: best = (o, c)
            if best[1] != g[k]: g[k] = best[1]; ch = True
        if not ch: break
    return g


def energy(qb, idx, n):
    x = np.zeros(n * N_SECTIONS)
    for k, c in enumerate(idx): x[k * N_SECTIONS + c] = 1
    return float(qb.energy(x))


def mlc_sizing(model, mhalf, load, mass_target, w_mass, seed, init, solver):
    n = model.n_groups; geom = element_geom_cache(model); groups = group_element_indices(model)
    blocks = [[k * N_SECTIONS + c for c in range(N_SECTIONS)] for k in range(n)]
    cur = list(init); C1h, C2h, Mh, Xh = [], [], [], []
    for it in range(N_FE):
        C1, M, f1 = evaluate_design(model, cur, load, check_cond=False)
        C2, _, f2 = evaluate_design(mhalf, cur, load, check_cond=False)
        C1h.append(C1); C2h.append(C2); Mh.append(M); Xh.append(list(cur))
        SE1, MC = linearise(model, geom, groups, f1); SE2, _ = linearise(mhalf, geom, groups, f2)
        SE = SE1 + SE2; qb = build_Q(model, cur, SE, MC, mass_target, w_mass)
        if solver == 'block':
            sa = block_sa_solve(qb.Q, blocks, n_reads=25, n_sweeps=90, seed=seed * 10000 + it)
            cur = [int(v) for v in sa['best_x'].reshape(n, N_SECTIONS).argmax(axis=1)]
        else:
            cur = greedy_decode(qb, cur, SE, MC, mass_target, w_mass, n)
    return dict(C_full=C1h, C_half=C2h, mass=Mh, idx=Xh)


def posthoc(model, idx, load):
    gs = {k: CHS_CATALOG[idx[k]] for k in range(model.n_groups)}; span = 20.0; rec = dict(mass=model_mass(model, gs))
    for key, sw, live in [('nominal', False, 'full'), ('sw_full', True, 'full'), ('sw_half', True, 'half')]:
        frm = frame(model, gs, load, sw, live)
        rec[key] = dict(C=frm.compliance(), d_mm=frm.max_abs_disp() * 1000, d_over_L250=frm.max_abs_disp() / (span / 250), util=max_util(model, gs, frm))
    return rec


def main():
    model = davinci_bridge2(n_bays=24, span=20.0, width=6.0, lam=0.25); load = model.meta['load_per_node']
    mhalf = half_model(model); n = model.n_groups
    mass_max = model_mass(model, {k: CHS_CATALOG[-1] for k in range(n)}); mt = 0.6 * mass_max
    from reciprocal_geom import LINK_SECTION
    nodes = np.array(model.nodes)
    link_mass = sum(LINK_SECTION['mass_per_len'] * np.linalg.norm(nodes[e['i']] - nodes[e['j']]) for e in model.elements if e['kind'] == 'link')
    mt_q = mt - link_mass
    out = dict(mass_target=mt, methods={})
    # ---- barrier diagnosis along the greedy trajectory, seed 0, w = 5 ----
    diag = []; cur = random_init(0, n); geom = element_geom_cache(model); groups = group_element_indices(model)
    blocks = [[k * N_SECTIONS + c for c in range(N_SECTIONS)] for k in range(n)]
    for it in range(N_FE):
        C, M, frm = evaluate_design(model, cur, load, check_cond=False)
        SE, MC = linearise(model, geom, groups, frm); qb = build_Q(model, cur, SE, MC, mt, 5.0)
        g = greedy_decode(qb, cur, SE, MC, mt, 5.0, n)
        s1 = sa_solve(qb.Q, 25, 90, seed=it); x1 = [int(v) for v in s1['best_x'].reshape(n, N_SECTIONS).argmax(1)]
        s2 = sa_solve(qb.Q, 100, 400, seed=it); x2 = [int(v) for v in s2['best_x'].reshape(n, N_SECTIONS).argmax(1)]
        b = block_sa_solve(qb.Q, blocks, 25, 90, seed=it); xb = [int(v) for v in b['best_x'].reshape(n, N_SECTIONS).argmax(1)]
        diag.append(dict(it=it, C=C, E_greedy=energy(qb, g, n), E_sa=energy(qb, x1, n), E_sa16=energy(qb, x2, n), E_block=energy(qb, xb, n)))
        cur = g
    out['barrier_diag_seed0_w5'] = diag
    e = np.array([[d['E_greedy'], d['E_sa'], d['E_sa16'], d['E_block']] for d in diag])
    print('barrier: greedy<=sa on', int((e[:, 0] <= e[:, 1] + 1e-9).sum()), '/50; greedy<=sa16', int((e[:, 0] <= e[:, 2] + 1e-9).sum()), '; block<=greedy', int((e[:, 3] <= e[:, 0] + 1e-9).sum()), '; mean gaps sa-greedy %.3f sa16-greedy %.3f block-greedy %.4f' % ((e[:, 1] - e[:, 0]).mean(), (e[:, 2] - e[:, 0]).mean(), (e[:, 3] - e[:, 0]).mean()), flush=True)
    # ---- two-load-case sizing, w = 100, paired seeds ----
    for solver in ('block', 'greedy'):
        key = f'mlc_{solver}_w1000f'; out['methods'][key] = []
        for s in range(10):
            init = random_init(s, n); r = mlc_sizing(model, mhalf, load, mt_q, 1000.0, s, init, solver)
            C = np.array(r['C_full']) + np.array(r['C_half']); M = np.array(r['mass']); ok = np.abs(M - mt) / mt <= 0.05
            bi = int(np.argmin(np.where(ok, C, np.inf))) if ok.any() else int(np.argmin(C))
            r['best_idx'] = r['idx'][bi]; r['best_feasible'] = bool(ok.any()); r['posthoc'] = posthoc(model, r['idx'][bi], load); r['seed'] = s
            out['methods'][key].append(r)
            p = r['posthoc']; print(f"{key} seed {s}: mass {p['mass']:.0f} Cfull {p['nominal']['C']:.0f} | SW+half d/(L/250) {p['sw_half']['d_over_L250']:.2f} util {p['sw_half']['util']:.2f}", flush=True)
            json.dump(out, open(os.path.join(OUT, 'cs6_mlc.json'), 'w'))
    # ---- single-load-case w100 designs under the same post-hoc cases ----
    rev = json.load(open(os.path.join(OUT, 'cs6.json')))
    for key in ('sa_block_w1000f', 'greedy_w1000f'):
        if key not in rev['methods']: continue
        out['methods'][key + '_posthoc'] = []
        for r in rev['methods'][key]:
            C = np.array(r['compliance_history']); M = np.array(r['mass_history']); ok = np.abs(M - mt) / mt <= 0.05
            bi = int(np.argmin(np.where(ok, C, np.inf))) if ok.any() else int(np.argmin(C))
            p = posthoc(model, r['idx_history'][bi], load); p['seed'] = r['seed']; p['best_idx'] = r['idx_history'][bi]
            out['methods'][key + '_posthoc'].append(p)
            print(f"{key} seed {r['seed']}: mass {p['mass']:.0f} Cfull {p['nominal']['C']:.0f} | SW+half d/(L/250) {p['sw_half']['d_over_L250']:.2f} util {p['sw_half']['util']:.2f}", flush=True)
    json.dump(out, open(os.path.join(OUT, 'cs6_mlc.json'), 'w'))
    print('ALL DONE')


if __name__ == '__main__':
    main()

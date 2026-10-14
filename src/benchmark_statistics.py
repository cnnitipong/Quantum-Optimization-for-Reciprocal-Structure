"""benchmark_statistics.py -- consolidated statistics for the revised manuscript.
Reads the original campaign JSONs plus results/benchmark/*.json and writes
results/benchmark/tables.json with every number quoted in the revised tables:
  * per case study and method: n seeds, best compliance (any mass), MATCHED-MASS
    best compliance (best evaluated design with |M - M*| <= 5 %), its mass error,
    the (C, M) of the best-penalised-objective design, wall-clock;
  * exact two-sided Mann-Whitney p against the reference arm, Cliff's delta,
    Holm-adjusted p within each table family; paired Wilcoxon signed-rank
    (exact, via permutation of signs) for the paired ablation arms.
"""
import os, sys, json, itertools
import os as _os_bootstrap
ROOT = _os_bootstrap.path.dirname(_os_bootstrap.path.dirname(_os_bootstrap.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
import numpy as np
from exact_mw import exact_mannwhitney
from stats_utils import cliffs_delta

R2 = os.path.join(ROOT, 'results', 'benchmark')
TOL = 0.05


def mw(a, b):
    r = exact_mannwhitney(np.asarray(a, float), np.asarray(b, float))
    return float(r.get('p_exact_two_sided', float('nan')))


def holm(pvals):
    idx = np.argsort(pvals); m = len(pvals); adj = np.empty(m); run = 0.0
    for rank, i in enumerate(idx):
        run = max(run, (m - rank) * pvals[i]); adj[i] = min(1.0, run)
    return adj


def wilcoxon_exact(d):
    """Exact two-sided sign-permutation p-value of the Wilcoxon signed-rank statistic."""
    d = np.asarray(d, float); d = d[d != 0]; n = len(d)
    if n == 0:
        return 1.0
    r = np.argsort(np.argsort(np.abs(d))) + 1.0
    w_obs = float(np.sum(r[d > 0]))
    ws = [float(np.sum(r[list(s)])) for k in range(n + 1) for s in itertools.combinations(range(n), k)]
    ws = np.array(ws); mu = ws.mean()
    return float(np.mean(np.abs(ws - mu) >= abs(w_obs - mu) - 1e-12))


def runs_of(d, key):
    m = d['methods'].get(key)
    if m is None:
        return None
    return list(m.values()) if isinstance(m, dict) else m


def summarise(runs, mt):
    out = dict(n=len(runs))
    best = [min(r['compliance_history']) for r in runs]
    mm = []; mmM = []; fin = []; nfeas = 0
    for r in runs:
        C = np.array(r['compliance_history']); M = np.array(r['mass_history'])
        ok = np.abs(M - mt) / mt <= TOL
        if ok.any():
            nfeas += 1; i = np.argmin(np.where(ok, C, np.inf)); mm.append(C[i]); mmM.append(M[i])
        else:
            i = np.argmin(np.abs(M - mt)); mm.append(np.nan); mmM.append(np.nan)
    bm = [r['mass_history'][int(np.argmin(r['compliance_history']))] for r in runs]
    out.update(best_mean=float(np.mean(best)), best_std=float(np.std(best)), best_mass_err_pct=float(np.mean((np.array(bm) - mt) / mt * 100)),
               mm_mean=float(np.nanmean(mm)), mm_std=float(np.nanstd(mm)), mm_vals=[float(x) for x in mm],
               mm_mass_err_pct=float(np.nanmean((np.array(mmM) - mt) / mt * 100)), n_feasible=nfeas,
               final_mass_err_pct=float(np.mean([(r['mass_history'][-1] - mt) / mt * 100 for r in runs])),
               time_s=float(np.mean([r.get('time_s', np.nan) for r in runs])) if any('time_s' in r for r in runs) else None,
               best_vals=[float(b) for b in best])
    return out


CASES = {
    'CS2': dict(base='results/cs2_v2.json', abl='results/r3_ablation_v2.json', w20='results/r2_matched_mass_v2.json', rev='cs2.json'),
    'CS3': dict(base='results/cs3_v2.json', abl='results/ablation/rb_ablation_cs3.json', w20='results/r2_cs3_w20_v2.json', rev='cs3.json'),
    'CS6': dict(base='results/complex/cs6_complex.json', rev='cs6.json'),
    'Dome': dict(base='results/complex/dome_sizing.json', rev='dome.json'),
    'Vault': dict(base='results/complex/vault_hex_sizing.json', rev='vault.json'),
}


def load(p):
    p = os.path.join(ROOT, p)
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    tables = {}
    for case, src in CASES.items():
        base = load(src['base']); mt = base['mass_target']
        arms = {}
        for key, label in [('sa_qubo', 'SA single-flip (w5, mid start)'), ('ga', 'GA pop 20'), ('random', 'Random'),
                           ('sa_qubo_w20', 'SA single-flip (w20)'), ('greedy_w5', 'Greedy (w5)'), ('sa_paired_w5', 'SA single-flip paired (w5)'),
                           ('greedy_w20', 'Greedy (w20)'), ('sa_paired_w20', 'SA single-flip paired (w20)')]:
            r = runs_of(base, key)
            if r:
                arms[key] = summarise(r, mt)
        # separate ablation / w20 files (CS2, CS3)
        if 'abl' in src and (abl := load(src['abl'])):
            if 'methods' in abl:
                for key, out in [('greedy_w5', 'greedy_w5'), ('sa_qubo_paired_w5', 'sa_paired_w5'), ('greedy_w20', 'greedy_w20'), ('sa_qubo_paired_w20', 'sa_paired_w20')]:
                    r = runs_of(abl, key)
                    if r:
                        arms[out] = summarise(r, abl.get('mass_target', mt))
            elif 'states' in abl:
                for pre, out in [('greedy_w5', 'greedy_w5'), ('sa_w5', 'sa_paired_w5'), ('greedy_w20', 'greedy_w20'), ('sa_w20', 'sa_paired_w20')]:
                    r = [dict(compliance_history=v['comp_hist'], mass_history=v['mass_hist']) for k, v in sorted(abl['states'].items(), key=lambda kv: int(kv[0].rsplit('_', 1)[1])) if k.rsplit('_', 1)[0] == pre and v.get('done')]
                    if r:
                        arms[out] = summarise(r, abl.get('mass_target', mt))
        if 'w20' in src and (w20 := load(src['w20'])):
            if 'methods' in w20:
                r = runs_of(w20, 'sa_qubo') or runs_of(w20, 'sa_qubo_w20')
                if r:
                    arms['sa_qubo_w20'] = summarise(r, w20.get('mass_target', mt))
            else:
                arms['_w20_keys'] = list(w20.keys())
        rev = load(os.path.join('results', 'benchmark', src['rev']))
        if rev:
            for key, r in rev['methods'].items():
                if r:
                    arms[key] = summarise(r, mt)
        # statistics: everything vs block SA w5 (new reference) and vs single-flip SA w5
        fam = []
        ref_keys = [k for k in ['sa_block_w1000f', 'sa_block_w5', 'sa_qubo'] if k in arms]
        for ref in ref_keys:
            for k, a in arms.items():
                if k.startswith('_') or k == ref or 'mm_vals' not in a:
                    continue
                x = [v for v in arms[ref]['mm_vals'] if not np.isnan(v)]; y = [v for v in a['mm_vals'] if not np.isnan(v)]
                if len(x) and len(y):
                    p = mw(x, y); d = float(cliffs_delta(np.array(x), np.array(y)))
                    fam.append(dict(ref=ref, arm=k, p_mm=p, cliff_mm=d, p_best=mw(arms[ref]['best_vals'], a['best_vals'])))
        if fam:
            adj = holm([f['p_mm'] for f in fam])
            for f, pa in zip(fam, adj):
                f['p_mm_holm'] = float(pa)
        # paired tests for arms sharing the random init
        paired = {}
        for a1, a2 in [('greedy_w5', 'sa_paired_w5'), ('greedy_w20', 'sa_paired_w20'), ('greedy_w5', 'sa_block_w5'), ('greedy_w20', 'sa_block_w20'),
                       ('sa_paired_w5', 'sa_block_w5'), ('sa_paired_w20', 'sa_block_w20'), ('greedy_w5', 'ls_w5'), ('greedy_w5', 'sa_strong_w5'), ('greedy_w1000f', 'sa_block_w1000f'), ('sa_single_w1000f', 'sa_block_w1000f'), ('greedy_w1000f', 'sa_single_w1000f')]:
            if a1 in arms and a2 in arms and 'best_vals' in arms[a1] and 'best_vals' in arms[a2]:
                n = min(len(arms[a1]['best_vals']), len(arms[a2]['best_vals']))
                d = np.array(arms[a1]['best_vals'][:n]) - np.array(arms[a2]['best_vals'][:n])
                dm = np.array(arms[a1]['mm_vals'][:n]) - np.array(arms[a2]['mm_vals'][:n])
                paired[f'{a1} vs {a2}'] = dict(n=n, mean_diff_best=float(d.mean()), p_wilcoxon_best=wilcoxon_exact(d),
                                              mean_diff_mm=float(np.nanmean(dm)), p_wilcoxon_mm=wilcoxon_exact(dm[~np.isnan(dm)]))
        tables[case] = dict(mass_target=mt, arms=arms, tests=fam, paired=paired)
        print(f'== {case} (M* = {mt:.1f} kg)')
        for k, a in arms.items():
            if k.startswith('_'):
                print('   ', k, a); continue
            print(f"  {k:28s} n={a['n']:2d} best {a['best_mean']:9.2f}±{a['best_std']:7.2f} | matched {a['mm_mean']:9.2f}±{a['mm_std']:7.2f} (feas {a['n_feasible']}/{a['n']}, massErr {a['mm_mass_err_pct']:+5.1f}%) | finalMassErr {a['final_mass_err_pct']:+6.1f}%")
        for f in fam:
            print(f"    {f['ref']} vs {f['arm']:22s}: p_mm={f['p_mm']:.4g} holm={f.get('p_mm_holm',np.nan):.4g} delta={f['cliff_mm']:+.2f} p_best={f['p_best']:.4g}")
        for k, v in paired.items():
            print(f"    paired {k}: n={v['n']} dBest={v['mean_diff_best']:+.1f} p={v['p_wilcoxon_best']:.4g} | dMM={v['mean_diff_mm']:+.1f} p={v['p_wilcoxon_mm']:.4g}")
    # ---- CS7 stress-constrained: matched-mass re-reporting ----
    cs7 = load('results/ablation/ri_stress_qubo.json')
    if cs7:
        mt7 = cs7['setup'].get('mass_target') if isinstance(cs7['setup'], dict) else None
        def st_runs(states, ck='comp_hist', mk='mass_hist'):
            return [dict(compliance_history=v[ck], mass_history=v[mk], seed=k) for k, v in states.items() if v.get(ck)]
        arms7 = {}
        if mt7:
            arms7['sa'] = summarise(st_runs(cs7['sa_states']), mt7)
            arms7['greedy'] = summarise(st_runs(cs7['greedy_states']), mt7)
            arms7['ga'] = summarise([dict(compliance_history=r['C_hist'], mass_history=r['M_hist']) for r in (cs7['ga_results'].values() if isinstance(cs7['ga_results'], dict) else cs7['ga_results'])], mt7)
            tests7 = {}
            for k in ('greedy', 'ga'):
                x = [v for v in arms7['sa']['mm_vals'] if not np.isnan(v)]; y = [v for v in arms7[k]['mm_vals'] if not np.isnan(v)]
                tests7[k] = dict(p_mm=mw(x, y) if x and y else float('nan'), cliff_mm=float(cliffs_delta(np.array(x), np.array(y))) if x and y else float('nan'))
            tables['CS7'] = dict(mass_target=mt7, arms=arms7, tests=tests7)
            print('== CS7 (M* = %.1f)' % mt7)
            for k, a in arms7.items():
                print(f"  {k:8s} n={a['n']} best {a['best_mean']:.1f}±{a['best_std']:.1f} (massErr {a['best_mass_err_pct']:+.1f}%) | matched {a['mm_mean']:.1f}±{a['mm_std']:.1f} feas {a['n_feasible']}/{a['n']}")
            print('  tests', tests7)
        else:
            print('CS7 setup keys', cs7['setup'] if not isinstance(cs7['setup'], dict) else list(cs7['setup'].keys()))
    json.dump(tables, open(os.path.join(R2, 'tables.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()

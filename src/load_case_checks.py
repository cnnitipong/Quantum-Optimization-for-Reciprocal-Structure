"""load_case_checks.py -- self-weight and asymmetric-live-load re-evaluation of
EVERY best-found design (CS2, CS3, CS6 bridge, dome geometry-best, dome sizing-best,
vault geometry-best, vault sizing-best), plus the ratio of total self-weight to total
nominal live load (load realism) and a simple S355 stress utilisation under each case.
Writes results/benchmark/loadcases.json.
"""
import os, sys, json
import os as _os_bootstrap
ROOT = _os_bootstrap.path.dirname(_os_bootstrap.path.dirname(_os_bootstrap.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
import numpy as np
from reciprocal_geom import build_frame, model_mass, LINK_SECTION
from reciprocal_geom_v2 import square_nexorade, hex_nexorade, davinci_bridge2, rf_dome2, rf_vault_hex2
from sections import CHS_CATALOG
import dome2_sizing, vault_hex2_sizing

G = 9.81; FY = 355e6
OUT = os.path.join(ROOT, 'results', 'benchmark'); os.makedirs(OUT, exist_ok=True)
LAM = np.linspace(0.05, 0.5, 8)


def designs():
    yield 'CS2 nexorade (SA-QUBO best, seed 6)', square_nexorade(6, 6, 2.5, 0.25), [6, 7, 5, 7, 6, 7, 5, 7, 7, 6, 2, 6], -450.0
    yield 'CS3 hexagonal nexorade (SA-QUBO best, seed 0)', hex_nexorade(3, 2.2, 0.25), [7, 7, 7, 5, 6, 6, 7, 7, 6], -1400.0
    m = davinci_bridge2(n_bays=24, span=20.0, width=6.0, lam=0.25)
    yield 'CS6 da Vinci bridge (SA-QUBO best, seed 8)', m, [7, 6, 7, 7, 7, 7, 7, 7, 7, 5, 5, 7, 3, 5, 5, 7], m.meta['load_per_node']
    m = dome2_sizing.make_model()
    yield 'Dome geometry-best (FMQA, seed 1, uniform D=244.5)', m, [7] * m.n_groups, m.meta['load_per_node']
    yield 'Dome sizing-best (SA-QUBO, seed 2)', m, [1, 5, 7, 7, 7, 6, 6, 7, 7, 6, 7, 5], m.meta['load_per_node']
    m = vault_hex2_sizing.make_model()
    yield 'Vault geometry-best (FMQA, seed 2, uniform D=244.5)', m, [7] * m.n_groups, m.meta['load_per_node']
    yield 'Vault sizing-best (SA-QUBO, seed 3)', m, [7, 7, 7, 6, 7, 6, 6, 6, 7, 6, 6, 7], m.meta['load_per_node']


def frame(model, gs, load, self_weight, live):
    frm = build_frame(model, gs, load_per_node=0.0)
    if self_weight:
        for e in model.elements:
            ni = np.array(model.nodes[e['i']]); nj = np.array(model.nodes[e['j']])
            L = float(np.linalg.norm(nj - ni)); sp = LINK_SECTION if e['kind'] == 'link' else gs[e['group']]
            w = -sp['mass_per_len'] * L * G / 2.0
            frm.add_load(e['i'], fz=w); frm.add_load(e['j'], fz=w)
    xs = np.array(model.nodes)[:, 0]; mid = 0.5 * (xs.min() + xs.max())
    if live != 'none':
        for ni in model.load_idx:
            if live == 'full' or model.nodes[ni][0] < mid:
                frm.add_load(ni, fz=load)
    frm.solve(check_cond=False)
    return frm


def max_util(model, gs, frm):
    """S355 elastic utilisation |N|/A + |M_res| r / I over beam elements (links excluded)."""
    u = 0.0
    for ei, e in enumerate(model.elements):
        if e['kind'] == 'link':
            continue
        f = frm.element_end_forces_local(ei); sp = gs[e['group']]
        r = sp['D'] / 2.0
        for end in (0, 6):
            N = abs(f[end]); M = np.hypot(f[end + 4], f[end + 5])
            u = max(u, (N / sp['A'] + M * r / sp['Iy']) / FY)
    return float(u)


def main():
    out = {}
    for name, model, idx, load in designs():
        gs = {k: CHS_CATALOG[idx[k]] for k in range(model.n_groups)}
        span = model.meta.get('span', None)
        if span is None:
            xs = np.array(model.nodes); span = float(xs[:, 0].max() - xs[:, 0].min())
        if 'vault' in name.lower():
            span = 14.0
        mass = model_mass(model, gs)
        G_total = mass * G; Q_total = abs(load) * len(model.load_idx)
        rec = dict(idx=idx, mass_kg=mass, span_m=span, n_load_nodes=len(model.load_idx), load_per_node=load,
                   selfweight_over_live=G_total / Q_total)
        for key, sw, live in [('nominal', False, 'full'), ('selfweight_only', True, 'none'),
                              ('sw_plus_full_live', True, 'full'), ('sw_plus_half_live', True, 'half')]:
            frm = frame(model, gs, load, sw, live)
            rec[key] = dict(compliance=frm.compliance(), d_max=frm.max_abs_disp(),
                            d_over_span250=frm.max_abs_disp() / (span / 250.0), max_util=max_util(model, gs, frm))
        out[name] = rec
        print(f"{name}: G/Q={rec['selfweight_over_live']:.2f} | nominal d/(L/250)={rec['nominal']['d_over_span250']:.3f} u={rec['nominal']['max_util']:.3f} | SW+full d/(L/250)={rec['sw_plus_full_live']['d_over_span250']:.3f} u={rec['sw_plus_full_live']['max_util']:.3f} dC={100*(rec['sw_plus_full_live']['compliance']/rec['nominal']['compliance']-1):.0f}% | SW+half d/(L/250)={rec['sw_plus_half_live']['d_over_span250']:.3f}", flush=True)
    json.dump(out, open(os.path.join(OUT, 'loadcases.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()

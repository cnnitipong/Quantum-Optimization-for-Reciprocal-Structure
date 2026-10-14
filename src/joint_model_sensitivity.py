import os
import os as _os_bootstrap
ROOT = _os_bootstrap.path.dirname(_os_bootstrap.path.dirname(_os_bootstrap.path.abspath(__file__)))
"""
RD -- joint-model checks on current geometries.

(1) Eccentricity x link-stiffness 3x3 sweep (ecc mult in {0.5,1,2}, link
    stiffness mult in {0.1,1,10}) re-evaluating the best CS2 and CS3
    designs (section assignment held FIXED, no re-optimization -- same
    sensitivity-not-re-optimization protocol as src/r7_sensitivity.py),
    reporting %compliance change vs nominal (1,1) and whether SA-QUBO's
    best design still beats GA's best design in all 9 cells.

(2) Link axial-force SIGN check for the best CS2/CS3/CS4/CS6 designs:
    recover local-frame end forces in every eccentricity-connector LINK
    element and report the fraction in TENSION (reader's point: a real
    bearing/contact joint cannot resist tension, so this is reported
    plainly regardless of the outcome).
"""
import json
import sys
import numpy as np

sys.path.insert(0, os.path.join(ROOT, "src"))
from reciprocal_geom_v2 import square_nexorade, hex_nexorade, rf_dome, davinci_bridge
from reciprocal_geom import build_frame, model_mass, LINK_SECTION
from sections import CHS_CATALOG, RHO_STEEL
from frame3d import Frame3D

RESULTS = os.path.join(ROOT, "results")
OUTFILE = f"{RESULTS}/review/rd_joint_model.json"
CACHE = f"{RESULTS}/review/recovered_best_designs.json"

ECC_MULTS = [0.5, 1.0, 2.0]
LINK_MULTS = [0.1, 1.0, 10.0]
NOMINAL_ECC_RATIO_CS2 = 0.07
NOMINAL_ECC_RATIO_CS3 = 0.07


def scaled_link_section(mult):
    return dict(A=LINK_SECTION['A'] * mult, Iy=LINK_SECTION['Iy'] * mult,
                Iz=LINK_SECTION['Iz'] * mult, J=LINK_SECTION['J'] * mult,
                mass_per_len=RHO_STEEL * LINK_SECTION['A'] * mult)


def axial_sign_convention():
    frm = Frame3D()
    frm.add_node(0, 0, 0)
    frm.add_node(1, 0, 0)
    frm.add_element(0, 1, 210e9, 80e9, 1e-3, 1e-6, 1e-6, 2e-6, ref_vec=[0, 0, 1])
    frm.fix(0, dofs=(0, 1, 2, 3, 4, 5))
    frm.add_load(1, fx=+1000.0)
    frm.solve()
    f0 = frm.element_end_forces_local(0)
    return float(f0[0])  # positive => TENSION convention


def eval_cs2_cs3(builder, args, idx_vec, load_per_node, ecc_mult, link_mult, ecc_kw):
    kw = dict(ecc_kw)
    kw['ecc_ratio'] = kw['ecc_ratio'] * ecc_mult
    model = builder(*args, **kw)
    gs = {k: CHS_CATALOG[idx_vec[k]] for k in range(model.n_groups)}
    ls = scaled_link_section(link_mult)
    frm = build_frame(model, gs, link_section=ls, load_per_node=load_per_node)
    frm.solve()
    return float(frm.compliance()), model_mass(model, gs, link_section=ls)


def part1_ecc_link_sweep():
    cache = json.load(open(CACHE))
    out = {}
    builders = dict(cs2=(square_nexorade, dict(ecc_ratio=NOMINAL_ECC_RATIO_CS2)),
                     cs3=(hex_nexorade, dict(ecc_ratio=NOMINAL_ECC_RATIO_CS3)))
    for case in ('cs2', 'cs3'):
        builder_fn = {'cs2': square_nexorade, 'cs3': hex_nexorade}[case]
        ecc_kw = builders[case][1]
        entry = cache[case]
        args = entry['args'][:-1]  # drop lam (last positional -> pass as lam kw is 4th positional actually)
        # args stored as [nx,ny,cell,lam] for cs2 or [rings,edge,lam] for cs3; builder signature
        # (nx,ny,cell,lam,ecc_ratio=...) / (rings,edge,lam,ecc_ratio=...) so pass ALL stored args positionally
        args = entry['args']
        idx_vec = entry['idx_vec']
        load_per_node = entry['load_per_node']

        sa_idx = idx_vec
        # GA best idx for this case
        dcase = json.load(open(f"{RESULTS}/{case}_v2.json"))
        ga_runs = dcase['methods']['ga']
        if isinstance(ga_runs, dict):
            ga_runs = list(ga_runs.values())
        ga_best = min(ga_runs, key=lambda r: r['best_compliance'])
        ga_idx = ga_best['final_idx']

        results = {}
        for label, idxv in (('sa_qubo', sa_idx), ('ga', ga_idx)):
            table = {}
            for em in ECC_MULTS:
                for lm in LINK_MULTS:
                    C, M = eval_cs2_cs3(builder_fn, args, idxv, load_per_node, em, lm, ecc_kw)
                    table[f"ecc x{em}_link x{lm}"] = dict(ecc_mult=em, link_mult=lm,
                                                            compliance=C, mass=M)
            nominal = table["ecc x1.0_link x1.0"]['compliance']
            for k, v in table.items():
                v['pct_change_vs_nominal'] = 100.0 * (v['compliance'] - nominal) / nominal
            results[label] = dict(idx_vec=[int(x) for x in idxv], nominal_compliance=nominal, table=table)

        def matrix(label):
            m = np.zeros((3, 3))
            for i, em in enumerate(ECC_MULTS):
                for j, lm in enumerate(LINK_MULTS):
                    m[i, j] = results[label]['table'][f"ecc x{em}_link x{lm}"]['pct_change_vs_nominal']
            return m.tolist()

        results['sa_qubo']['pct_change_matrix_rows_ecc_cols_link'] = matrix('sa_qubo')
        results['ga']['pct_change_matrix_rows_ecc_cols_link'] = matrix('ga')

        sa_wins = {}
        n_sa_wins = 0
        for em in ECC_MULTS:
            for lm in LINK_MULTS:
                key = f"ecc x{em}_link x{lm}"
                c_sa = results['sa_qubo']['table'][key]['compliance']
                c_ga = results['ga']['table'][key]['compliance']
                win = bool(c_sa < c_ga)
                sa_wins[key] = dict(sa_compliance=c_sa, ga_compliance=c_ga, sa_beats_ga=win,
                                     pct_sa_better=100.0 * (c_ga - c_sa) / c_ga)
                n_sa_wins += int(win)

        out[case] = dict(ecc_mults=ECC_MULTS, link_mults=LINK_MULTS,
                          sa_qubo=results['sa_qubo'], ga=results['ga'],
                          sa_vs_ga_per_variant=sa_wins, n_variants_sa_beats_ga=n_sa_wins,
                          n_variants_total=9, sa_beats_ga_in_all_variants=bool(n_sa_wins == 9))
        print(f"[{case}] SA beats GA in {n_sa_wins}/9 ecc x link-stiffness variants")
    return out


def build_from_cache(entry):
    case = entry['case']
    if case == 'cs2':
        model = square_nexorade(*entry['args'])
    elif case == 'cs3':
        model = hex_nexorade(*entry['args'])
    elif case == 'cs4':
        model = rf_dome(*entry['args'], lam_per_ring=entry['lam_per_ring'],
                         load_per_node=entry['load_per_node'])
    elif case == 'cs6':
        model = davinci_bridge(*entry['args'])
    else:
        raise ValueError(case)
    gs = {k: CHS_CATALOG[entry['idx_vec'][k]] for k in range(model.n_groups)}
    return model, gs, entry['load_per_node']


def part2_link_tension_check():
    cache = json.load(open(CACHE))
    axial_sign = axial_sign_convention()
    out = {'axial_sign_convention_test_Ni_positive_is_tension': axial_sign}
    for case in ('cs2', 'cs3', 'cs4', 'cs6'):
        model, gs, load_per_node = build_from_cache(cache[case])
        frm = build_frame(model, gs, load_per_node=load_per_node)
        frm.solve()
        link_forces = []
        for ei, e in enumerate(model.elements):
            if e['kind'] != 'link':
                continue
            f = frm.element_end_forces_local(ei)
            Ni = float(f[0])
            is_tension = bool(np.sign(Ni) == np.sign(axial_sign)) if Ni != 0 else False
            link_forces.append(dict(elem_idx=ei, Ni=Ni, is_tension=is_tension))
        n_links = len(link_forces)
        n_tension = sum(1 for lf in link_forces if lf['is_tension'])
        frac_tension = n_tension / n_links if n_links > 0 else None
        n_abs = np.array([abs(lf['Ni']) for lf in link_forces])
        out[case] = dict(n_links=n_links, n_tension=n_tension, n_compression=n_links - n_tension,
                          fraction_tension=frac_tension,
                          max_abs_link_axial_N=float(n_abs.max()) if n_links else None,
                          mean_abs_link_axial_N=float(n_abs.mean()) if n_links else None,
                          worst_5_tension=sorted([lf for lf in link_forces if lf['is_tension']],
                                                  key=lambda x: -abs(x['Ni']))[:5])
        print(f"[{case}] links={n_links} tension={n_tension} ({100*frac_tension:.1f}%) "
              f"compression={n_links - n_tension}")
    return out


def main():
    out = dict(
        part1_ecc_link_sensitivity=part1_ecc_link_sweep(),
        part2_link_axial_tension_check=part2_link_tension_check(),
        note="Part 1: section assignment held FIXED at the best existing SA-QUBO/GA "
             "design for CS2/CS3 (no re-optimization) while sweeping ecc_ratio and "
             "link-connector stiffness; isolates joint-modelling sensitivity of an "
             "already-chosen design. Part 2: LINK elements are the eccentric "
             "master-slave bearing connectors (reciprocal_geom.LINK_SECTION); a real "
             "bearing/contact joint physically cannot resist tension, so any nonzero "
             "tension fraction here is a genuine modelling-idealisation caveat, "
             "reported plainly regardless of outcome.",
    )
    with open(OUTFILE, 'w') as f:
        json.dump(out, f, indent=2)
    print("WROTE", OUTFILE)
    print("ALL DONE")


if __name__ == "__main__":
    main()

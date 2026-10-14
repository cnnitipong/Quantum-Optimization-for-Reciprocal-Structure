# Quantum Optimization for Reciprocal Structures

Source code, raw results and figure scripts accompanying

> N. Praphaphankul. *Annealing-ready QUBO formulations for the discrete design of reciprocal frame structures.* Submitted to *Engineering Structures*, 2026.

Reciprocal frames, or nexorades, are assemblies of mutually supporting beams. Their design variables are discrete: a catalogue section for every member group and an engagement ratio for every fan. This repository formulates three design problems on these structures as quadratic unconstrained binary optimisation (QUBO) problems: member sizing, engagement-ratio selection and the mass versus compliance trade-off. The structures are modelled with a three-dimensional Euler-Bernoulli frame with eccentric bearing connectors, for six typologies: a fan unit, a rotated-square nexorade, a hexagonal nexorade, a reciprocal dome, a barrel vault and a da Vinci bridge.

All QUBO instances are minimised by classical simulated annealing (single-flip in `src/qubo.py`, one-hot block annealing in `src/blocksa.py`). The quantum kernel in `src/geometry_search_methods.py` is simulated exactly on a classical computer. No quantum hardware is used. The QUBO matrices can be submitted to an annealer without modification.

## Requirements

Python 3.10 or later with NumPy and Matplotlib (see `requirements.txt`). SciPy is not used: the frame solver, the statistics and the samplers are implemented with NumPy only. The study ran on a two-core CPU. Every driver fixes its random seeds.

```bash
python3 -m pip install -r requirements.txt
```

## Repository structure

```
src/                       source code
  frame3d.py, fastsolve.py         3D frame finite elements; banded Cholesky solver
  sections.py                      EN 10210 circular hollow section catalogue
  reciprocal_geom.py               fan construction, bearing model, frame assembly
  reciprocal_geom_v2.py            typology generators (nexorades, dome, vault, bridge)
  qubo.py, blocksa.py              QUBO construction; single-flip and block annealing
  sizing.py, greedy_sizing.py      strain-energy-linearised sizing loop (F1); coordinate-descent decode
  fmqa.py, resumable_v2.py         factorization-machine surrogate search (F2); GA and random search
  stats_utils.py, exact_mw.py      exact Mann-Whitney test, Wilson intervals, Cliff's delta
  ...                              case-study drivers, statistics, tables and figures (see below)
results/                   raw results (JSON) of every run reported in the paper
figures/                   figure panels of the paper, one PDF per panel
tables/                    LaTeX tables written by src/latex_tables.py
```

## Reproducing the paper

Commands run from the repository root. Each step reads its inputs from `results/` and writes to the same folder, so any step can be repeated on its own. The approximate run times below are for two cores.

| Paper item | Script | Output | Time |
|---|---|---|---|
| Typology checks, Table 3, Section 2.5 | `qa_dome2.py`, `qa_vault_hex2.py`, `qa_bridge2.py`, `validation_gates.py` | `results/*verification.json`, `results/benchmark/gates.json` | 5 min |
| CS1 enumeration and samplers, Fig. 11 | `cs1_fans_456.py`, `cs1_block_annealer.py`, `r8_f1_validation_v2.py` | `results/cs1_fans_456.json`, `results/benchmark/cs1_block.json` | 10 min |
| Engagement-ratio search, Table 5, Fig. 12 | `cs4_dome2.py`, `cs_vault_hex2_geo_par.py` | `results/complex/cs4_complex.json`, `results/complex/vault_hex_geo.json` | 1 h |
| Alternative search methods, Table 6, Figs. 13 to 16 | `geometry_search_methods.py run`, `geometry_search_methods.py enum`, `figures_geometry_search.py` | `results/geometry_search/` | 1.5 h |
| Sizing base runs (CS2 to CS6) | `cs2_v2.py`, `cs3_v2.py`, `cs6_complex_bridge2.py`, `dome2_sizing.py`, `vault_hex2_sizing.py`, `r2_matched_mass_v2.py`, `r2_cs3_w20_v2.py`, `r3_ablation_v2.py`, `ablation_cs3.py` | `results/*.json`, `results/complex/`, `results/ablation/` | 4 h |
| Block annealer, matched-mass arms, Tables 7 and 8 | `sizing_benchmark.py <case> 10`, with `ARM_W100=1`, `ARM_W1000=1` or `ARM_LINKFIX=1` | `results/benchmark/<case>.json` | 10 to 60 min per case |
| Statistics and tables | `benchmark_statistics.py`, `latex_tables.py tables` | `results/benchmark/tables.json`, `tables/` | 5 min |
| Stress-constrained sizing CS7, Table 9 | `cs7_stress_constrained.py` | `results/ablation/ri_stress_qubo.json` | 30 min |
| Multi-objective sizing, Fig. 24 | `cs5_v2.py`, `cs5_analysis_v2.py` | `results/cs5_v2.json`, `results/cs5_analysis_v2.json` | 1 h |
| Penalty, move limit and joint model, Section 5.8 | `cs_penalty_sensitivity_v2.py`, `r9_move_limit_v2.py`, `joint_model_sensitivity.py` | `results/` | 30 min |
| Engineering checks, Table 10 | `load_case_checks.py`, `bridge_two_load_cases.py`, `r6_uls_v2.py`, `bridge2_posthoc.py`, `dome2_posthoc.py`, `vault_hex_posthoc.py` | `results/benchmark/`, `results/complex/` | 20 min |
| Figs. 1 and 2 | `figures_schematics.py` | `figures/` | seconds |
| Figs. 4, 12, 17, 23 and 24 | `figures_results.py` | `figures/panels/` | 1 min |
| Figs. 3, 5 to 8, 10, 11, 18 to 22 and 25 to 28 | `figures_structures.py` | `figures/panels/`, `results/structures/figdata.json` | 1 min |

`run_all.sh` runs the analysis, statistics and figure steps in this order. In the sizing results, arms whose names end in `f` use the mass target net of the constant connector mass. These are the matched-mass arms of the paper.

## Citation

Please cite the paper if you use this code. Citation metadata is given in `CITATION.cff`.

## Licence

MIT licence; see `LICENSE`.

## Contact

Nitipong Praphaphankul, Department of Architecture, Faculty of Architecture, Chulalongkorn University, Bangkok, Thailand.

#!/usr/bin/env bash
# Reproduces the analyses, statistics, tables and figures of the paper.
# Run from the repository root. Total run time is several hours on two cores.
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
P="python3 -I"

# typologies and validation
$P src/qa_dome2.py; $P src/qa_vault_hex2.py; $P src/qa_bridge2.py; $P src/validation_gates.py
# CS1: enumerable fan units
$P src/cs1_fans_456.py; $P src/cs1_block_annealer.py; $P src/r8_f1_validation_v2.py
# engagement-ratio search
$P src/cs4_dome2.py; $P src/cs_vault_hex2_geo_par.py
$P src/geometry_search_methods.py run; $P src/geometry_search_methods.py enum
# sizing
$P src/cs2_v2.py; $P src/cs3_v2.py; $P src/cs6_complex_bridge2.py; $P src/dome2_sizing.py; $P src/vault_hex2_sizing.py
$P src/r2_matched_mass_v2.py; $P src/r2_cs3_w20_v2.py; $P src/r3_ablation_v2.py; $P src/ablation_cs3.py
for c in cs2 cs3 cs6 dome vault; do
  $P src/sizing_benchmark.py $c 10
  ARM_W100=1 $P src/sizing_benchmark.py $c 10
  ARM_W1000=1 $P src/sizing_benchmark.py $c 10
  ARM_LINKFIX=1 $P src/sizing_benchmark.py $c 10
done
$P src/cs7_stress_constrained.py
$P src/cs5_v2.py; $P src/cs5_analysis_v2.py
$P src/cs_penalty_sensitivity_v2.py; $P src/r9_move_limit_v2.py; $P src/joint_model_sensitivity.py
# engineering checks
$P src/load_case_checks.py; $P src/bridge_two_load_cases.py; $P src/r6_uls_v2.py
$P src/bridge2_posthoc.py; $P src/dome2_posthoc.py; $P src/vault_hex_posthoc.py
# statistics, tables and figures
$P src/benchmark_statistics.py; $P src/latex_tables.py tables
$P src/figures_schematics.py; $P src/figures_results.py; $P src/figures_structures.py; $P src/figures_geometry_search.py

"""figures_schematics.py -- construction schematics of the paper (Figs. 1 and 2).

Fig. 1: tangent-circle fan construction at two engagement ratios.
Fig. 2: rotation-method construction of a nexorade patch.
Output: figures/fig_fan_geometry.pdf and figures/fig_rotation_method.pdf.
Run from the repository root:  python3 -I src/figures_schematics.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
os.makedirs(os.path.join(ROOT, 'figures'), exist_ok=True)

import schematics_fan
import schematics_nexorade

schematics_fan.FIG = os.path.join(ROOT, 'figures')
schematics_nexorade.FIG = os.path.join(ROOT, 'figures')
schematics_fan.fig_fan_geometry()
schematics_nexorade.fig_rotation_method()

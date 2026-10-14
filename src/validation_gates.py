"""validation_gates.py -- validation gates 1-4 at lambda in {0.15, 0.25, 0.4} for ALL six
typologies used in the revised manuscript (fan4, square_nexorade, hex_nexorade,
rf_dome2, rf_vault_hex2, davinci_bridge2); gate 1 via the banded solve (non-singular
if the factorisation succeeds and the compliance is finite), gate 4 under the uniform
mid-catalogue section (index 4) and the typology's own nominal load."""
import os, sys, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src'))
import numpy as np
from reciprocal_geom import build_frame
from reciprocal_geom_v2 import (fan_unit, square_nexorade, hex_nexorade, rf_dome2, rf_vault_hex2, davinci_bridge2,
                                verify_no_interior_supports, verify_member_bearings)
from sections import CHS_CATALOG
try:
    from reciprocal_geom_v2 import verify_bridge_bearings
except ImportError:
    verify_bridge_bearings = None
try:
    from reciprocal_geom_v2 import verify_vault_hex2_bearings
except ImportError:
    verify_vault_hex2_bearings = None
try:
    from reciprocal_geom_v2 import verify_bridge2_bearings
except ImportError:
    verify_bridge2_bearings = None

B = {
 'fan4': (lambda lam: fan_unit(4, R=3.0, lam=lam), -2200.0, 8.49),
 'square_nexorade': (lambda lam: square_nexorade(6, 6, 2.5, lam), -450.0, 15.0),
 'hex_nexorade': (lambda lam: hex_nexorade(3, 2.2, lam), -1400.0, 13.2),
 'rf_dome2': (lambda lam: rf_dome2(n_sect=7, n_ring=4, span=18.5, rise=3.15, lam=lam, max_group_rings=4), -3500.0, 18.5),
 'rf_vault_hex2': (lambda lam: rf_vault_hex2(lam_bands=(lam,) * 6), -2200.0, 14.0),
 'davinci_bridge2': (lambda lam: davinci_bridge2(n_bays=24, span=20.0, width=6.0, lam=lam), -5500.0, 20.0),
}
out = {}
for name, (build, load, span) in B.items():
    out[name] = {}
    for lam in (0.15, 0.25, 0.4):
        m = build(lam); gs = {k: CHS_CATALOG[4] for k in range(m.n_groups)}
        try:
            frm = build_frame(m, gs, load_per_node=load); frm.solve(check_cond=False); C = frm.compliance(); d = frm.max_abs_disp(); g1 = bool(np.isfinite(C) and C > 0)
        except Exception as e:
            C, d, g1 = float('nan'), float('nan'), False
        g2 = verify_no_interior_supports(m, label=name)['ok']
        try:
            if 'bridge' in name and verify_bridge2_bearings: g3 = verify_bridge2_bearings(m, label=name)['ok']
            elif 'bridge' in name and verify_bridge_bearings: g3 = verify_bridge_bearings(m, label=name)['ok']
            elif 'vault' in name and verify_vault_hex2_bearings: g3 = verify_vault_hex2_bearings(m, label=name)['ok']
            else: g3 = verify_member_bearings(m, label=name)['ok']
        except Exception as e:
            g3 = f'error: {e}'
        lo, hi = span / 2000, span / 100; g4 = bool(lo <= d <= hi)
        out[name][str(lam)] = dict(gate1=g1, gate2=bool(g2), gate3=g3, gate4=g4, dmax=d, lo=lo, hi=hi, C=C)
        print(f"{name:16s} lam={lam}: g1={g1} g2={g2} g3={g3} g4={g4} dmax={d*1000:.1f} mm (window {lo*1000:.1f}-{hi*1000:.0f})", flush=True)
json.dump(out, open(os.path.join(ROOT, 'results', 'benchmark', 'gates.json'), 'w'), indent=1)

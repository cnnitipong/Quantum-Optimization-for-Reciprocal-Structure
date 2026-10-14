"""
sections.py -- Circular Hollow Section (CHS) catalog, per EN 10210 hot-finished
structural hollow sections (nominal outside diameter D x wall thickness t).
8 sections spanning D = 60.3 .. 244.5 mm, representative of common EN 10210-2
listed sizes. Section properties (A, I, J=2I, elastic modulus etc.) computed
analytically from D, t (thin/thick-wall exact annulus formulas) -- these are
geometric properties computed directly, not looked up, so they are exact
for the stated D, t regardless of rounding conventions in mill tables.

Steel: E = 210 GPa, G = E/(2(1+nu)) with nu = 0.3, rho = 7850 kg/m^3.
"""
import numpy as np

E_STEEL = 210e9      # Pa
NU_STEEL = 0.3
G_STEEL = E_STEEL / (2 * (1 + NU_STEEL))
RHO_STEEL = 7850.0   # kg/m^3

# (D_mm, t_mm) -- 8 representative EN 10210 CHS sizes, ascending
CHS_CATALOG_MM = [
    (60.3, 3.6),
    (76.1, 4.0),
    (88.9, 4.5),
    (114.3, 5.0),
    (139.7, 6.0),
    (168.3, 6.5),
    (193.7, 8.0),
    (244.5, 10.0),
]


def chs_props(D_mm, t_mm):
    D = D_mm / 1000.0
    t = t_mm / 1000.0
    d = D - 2 * t  # inner diameter
    A = np.pi / 4 * (D**2 - d**2)
    I = np.pi / 64 * (D**4 - d**4)   # second moment of area (Iy = Iz for CHS)
    J = 2 * I                         # polar moment = Iy+Iz for CHS
    mass_per_len = A * RHO_STEEL
    return dict(D=D, t=t, A=A, Iy=I, Iz=I, J=J, mass_per_len=mass_per_len,
                D_mm=D_mm, t_mm=t_mm)


CHS_CATALOG = [chs_props(D, t) for D, t in CHS_CATALOG_MM]
N_SECTIONS = len(CHS_CATALOG)


def section_array(key):
    return np.array([s[key] for s in CHS_CATALOG])


if __name__ == "__main__":
    print(f"{'D(mm)':>8} {'t(mm)':>8} {'A(mm2)':>10} {'I(cm4)':>10} {'J(cm4)':>10} {'kg/m':>8}")
    for s in CHS_CATALOG:
        print(f"{s['D_mm']:8.1f} {s['t_mm']:8.1f} {s['A']*1e6:10.1f} "
              f"{s['Iy']*1e8:10.2f} {s['J']*1e8:10.2f} {s['mass_per_len']:8.2f}")

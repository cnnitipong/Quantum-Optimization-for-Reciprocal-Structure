"""
frame3d.py -- 3D Euler-Bernoulli frame finite element solver.

12-DOF-per-element (2 nodes x 6 DOF: ux,uy,uz,rx,ry,rz), standard local
stiffness matrix (A, Iy, Iz, J, E, G), rotation to global frame,
direct-stiffness assembly, boundary condition application, dense linear
solve (numpy -- scipy not available in this environment).

Self-tests at the bottom (run as __main__): simply supported beam under
midspan point load vs P L^3/(48 E I), and a cantilever tip point load vs
P L^3/(3 E I). Both must match analytical solutions to <1% error.
"""
import numpy as np

DOF_PER_NODE = 6


def local_stiffness(E, G, A, Iy, Iz, J, L):
    """Standard 12x12 local stiffness matrix for a 3D Euler-Bernoulli beam.

    Local axis convention: x along the beam, y and z local transverse axes.
    DOF order per node: [ux, uy, uz, rx, ry, rz].
    """
    k = np.zeros((12, 12))

    EA_L = E * A / L
    GJ_L = G * J / L

    EIz = E * Iz
    EIy = E * Iy

    # Axial
    k[0, 0] = k[6, 6] = EA_L
    k[0, 6] = k[6, 0] = -EA_L

    # Torsion
    k[3, 3] = k[9, 9] = GJ_L
    k[3, 9] = k[9, 3] = -GJ_L

    # Bending about local z (transverse displacement uy, rotation rz)
    # DOFs: uy1(1), rz1(5), uy2(7), rz2(11)
    c = EIz / L**3
    k[1, 1] += 12 * c
    k[1, 5] += 6 * L * c
    k[1, 7] += -12 * c
    k[1, 11] += 6 * L * c

    k[5, 1] += 6 * L * c
    k[5, 5] += 4 * L**2 * c
    k[5, 7] += -6 * L * c
    k[5, 11] += 2 * L**2 * c

    k[7, 1] += -12 * c
    k[7, 5] += -6 * L * c
    k[7, 7] += 12 * c
    k[7, 11] += -6 * L * c

    k[11, 1] += 6 * L * c
    k[11, 5] += 2 * L**2 * c
    k[11, 7] += -6 * L * c
    k[11, 11] += 4 * L**2 * c

    # Bending about local y (transverse displacement uz, rotation ry)
    # DOFs: uz1(2), ry1(4), uz2(8), ry2(10)
    # Sign convention chosen so both bending blocks have identical
    # structure (equivalent to using a right-handed local frame consistently).
    d = EIy / L**3
    k[2, 2] += 12 * d
    k[2, 4] += -6 * L * d
    k[2, 8] += -12 * d
    k[2, 10] += -6 * L * d

    k[4, 2] += -6 * L * d
    k[4, 4] += 4 * L**2 * d
    k[4, 8] += 6 * L * d
    k[4, 10] += 2 * L**2 * d

    k[8, 2] += -12 * d
    k[8, 4] += 6 * L * d
    k[8, 8] += 12 * d
    k[8, 10] += 6 * L * d

    k[10, 2] += -6 * L * d
    k[10, 4] += 2 * L**2 * d
    k[10, 8] += 6 * L * d
    k[10, 10] += 4 * L**2 * d

    return k


def transformation_matrix(node_i, node_j, ref_vec=None):
    """12x12 rotation matrix from local to global for a beam element.

    ref_vec: an approximate "up"/reference vector used to fix orientation of
    local y,z axes (must not be parallel to element axis). Defaults chosen
    automatically if not given.
    """
    xi, yi, zi = node_i
    xj, yj, zj = node_j
    dx, dy, dz = xj - xi, yj - yi, zj - zi
    L = np.sqrt(dx**2 + dy**2 + dz**2)
    ex = np.array([dx, dy, dz]) / L

    if ref_vec is None:
        # pick global Z as reference unless element is (near) vertical
        if abs(ex[0]) < 1e-6 and abs(ex[1]) < 1e-6:
            ref_vec = np.array([0.0, 1.0, 0.0])
        else:
            ref_vec = np.array([0.0, 0.0, 1.0])
    ref_vec = np.asarray(ref_vec, dtype=float)

    ez = np.cross(ex, ref_vec)
    nz = np.linalg.norm(ez)
    if nz < 1e-9:
        # ref_vec parallel to ex: pick another
        ref_vec = np.array([1.0, 0.0, 0.0])
        ez = np.cross(ex, ref_vec)
        nz = np.linalg.norm(ez)
    ez = ez / nz
    ey = np.cross(ez, ex)
    ey = ey / np.linalg.norm(ey)

    R = np.vstack([ex, ey, ez])  # 3x3: rows are local axes in global coords
    T = np.zeros((12, 12))
    for i in range(4):
        T[3 * i:3 * i + 3, 3 * i:3 * i + 3] = R
    return T, L


class Frame3D:
    def __init__(self):
        self.nodes = []          # list of [x,y,z]
        self.elements = []       # list of dict: i,j,E,G,A,Iy,Iz,J,ref_vec,tag
        self.loads = {}          # node_idx -> 6-vector
        self.fixed = {}          # (node_idx, dof_local 0..5) -> prescribed value (0 for support)

    def add_node(self, x, y, z):
        self.nodes.append([x, y, z])
        return len(self.nodes) - 1

    def add_element(self, i, j, E, G, A, Iy, Iz, J, ref_vec=None, tag=None):
        self.elements.append(dict(i=i, j=j, E=E, G=G, A=A, Iy=Iy, Iz=Iz, J=J,
                                   ref_vec=ref_vec, tag=tag))
        return len(self.elements) - 1

    def add_load(self, node_idx, fx=0, fy=0, fz=0, mx=0, my=0, mz=0):
        v = self.loads.get(node_idx, np.zeros(6))
        v = v + np.array([fx, fy, fz, mx, my, mz], dtype=float)
        self.loads[node_idx] = v

    def fix(self, node_idx, dofs=(0, 1, 2, 3, 4, 5)):
        for d in dofs:
            self.fixed[(node_idx, d)] = 0.0

    def ndof(self):
        return len(self.nodes) * DOF_PER_NODE

    def gdof(self, node_idx, local_dof):
        return node_idx * DOF_PER_NODE + local_dof

    def assemble(self):
        n = self.ndof()
        K = np.zeros((n, n))
        elem_data = []
        for e in self.elements:
            ni, nj = self.nodes[e['i']], self.nodes[e['j']]
            T, L = transformation_matrix(ni, nj, e.get('ref_vec'))
            k_local = local_stiffness(e['E'], e['G'], e['A'], e['Iy'], e['Iz'], e['J'], L)
            k_global = T.T @ k_local @ T
            dofs = [self.gdof(e['i'], d) for d in range(6)] + [self.gdof(e['j'], d) for d in range(6)]
            for a in range(12):
                for b in range(12):
                    K[dofs[a], dofs[b]] += k_global[a, b]
            elem_data.append(dict(L=L, T=T, k_local=k_local, dofs=dofs))
        self._elem_data = elem_data
        return K

    def load_vector(self):
        n = self.ndof()
        F = np.zeros(n)
        for node_idx, v in self.loads.items():
            for d in range(6):
                F[self.gdof(node_idx, d)] += v[d]
        return F

    def solve(self, check_cond=True):
        """Static solve. Returns (U, R) global displacement & reaction vectors.

        check_cond: if True (default), runs np.linalg.cond(Kff) as a
        mechanism/singularity guard before solving and raises LinAlgError
        on a (near-)singular system -- this is the SAFE, always-correct
        path and should be used any time a NEW geometry/section-assignment
        CLASS is being solved for the first time (e.g. validation gates,
        one-off recovery/ULS solves).

        If False, the cond() check is skipped entirely and only
        np.linalg.solve is called -- this is a FAST PATH intended ONLY for
        the inner loop of an optimization campaign where the same model
        topology has ALREADY been verified non-singular (by an earlier
        check_cond=True gate call on a representative member of the same
        geometry/assignment class), so every subsequent per-eval solve in
        that campaign does not need to re-pay the O(n^3)-ish cond-number
        computation. np.linalg.solve itself will still raise a numpy
        LinAlgError (singular matrix) on a genuinely singular Kff even
        with check_cond=False, so this is not a silent-failure path -- it
        just skips the *early, more conservative* cond>1e14 threshold
        check. See fast_path_validation.json for allclose validation of
        this fast path against the checked path.
        """
        K = self.assemble()
        F = self.load_vector()
        n = self.ndof()
        fixed_dofs = sorted(self.gdof(ni, d) for (ni, d) in self.fixed.keys())
        free_dofs = [d for d in range(n) if d not in set(fixed_dofs)]

        if len(free_dofs) == 0:
            U = np.zeros(n)
            R = K @ U - F
            self.U = U
            self.R = R
            return U, R

        Kff = K[np.ix_(free_dofs, free_dofs)]
        Ff = F[free_dofs]

        if check_cond:
            # singularity check
            cond = np.linalg.cond(Kff)
            if not np.isfinite(cond) or cond > 1e14:
                raise np.linalg.LinAlgError(
                    f"Stiffness matrix is (near-)singular, cond={cond:.3e}. Mechanism suspected.")

        Uf = np.linalg.solve(Kff, Ff)
        U = np.zeros(n)
        U[free_dofs] = Uf
        R = K @ U - F
        self.U = U
        self.R = R
        self.K = K
        return U, R

    def node_disp(self, node_idx):
        return self.U[node_idx * 6:(node_idx + 1) * 6]

    def element_end_forces_local(self, elem_idx):
        """Local-frame end forces/moments for element elem_idx (12-vector)."""
        ed = self._elem_data[elem_idx]
        u_global = self.U[ed['dofs']]
        u_local = ed['T'] @ u_global
        f_local = ed['k_local'] @ u_local
        return f_local

    def element_strain_energy(self, elem_idx):
        ed = self._elem_data[elem_idx]
        u_global = self.U[ed['dofs']]
        u_local = ed['T'] @ u_global
        return 0.5 * u_local @ ed['k_local'] @ u_local

    def compliance(self):
        """Total external work = F^T U (a standard global stiffness proxy)."""
        F = self.load_vector()
        return float(F @ self.U)

    def max_abs_disp(self):
        n_nodes = len(self.nodes)
        disp = self.U.reshape(n_nodes, 6)[:, 0:3]
        mag = np.linalg.norm(disp, axis=1)
        return float(mag.max())


# ----------------------------- self tests -----------------------------
def _self_test():
    E = 210e9
    G = 80.77e9
    A = 1.0e-3
    Iy = 8.0e-6
    Iz = 8.0e-6
    J = 1.6e-5
    L = 4.0
    P = 1000.0

    tol = 0.01  # 1%

    # --- Test 1: simply supported beam, point load at midspan ---
    n_elem = 10
    frm = Frame3D()
    xs = np.linspace(0, L, n_elem + 1)
    for x in xs:
        frm.add_node(x, 0.0, 0.0)
    for e in range(n_elem):
        frm.add_element(e, e + 1, E, G, A, Iy, Iz, J, ref_vec=[0, 0, 1])
    # supports: node0 pinned (all translations + torsional restraint),
    # last node roller (uy,uz fixed only, allow ux). Torsion (rx) must be
    # restrained at least once along a collinear chain or the whole line
    # has a zero-stiffness rigid-body twist mode (mechanism).
    frm.fix(0, dofs=(0, 1, 2, 3))
    frm.fix(n_elem, dofs=(1, 2))
    mid = n_elem // 2
    frm.add_load(mid, fz=-P)
    frm.solve()
    w_num = frm.node_disp(mid)[2]
    w_analytical = -P * L**3 / (48 * E * Iy)
    err1 = abs((w_num - w_analytical) / w_analytical)
    print(f"[Test1: SS beam midspan] numeric={w_num:.6e} analytical={w_analytical:.6e} err={err1*100:.4f}%")
    assert err1 < tol, "Simply supported beam self-test FAILED"

    # --- Test 2: cantilever, tip point load ---
    frm2 = Frame3D()
    n_elem2 = 10
    xs2 = np.linspace(0, L, n_elem2 + 1)
    for x in xs2:
        frm2.add_node(x, 0.0, 0.0)
    for e in range(n_elem2):
        frm2.add_element(e, e + 1, E, G, A, Iy, Iz, J, ref_vec=[0, 0, 1])
    frm2.fix(0, dofs=(0, 1, 2, 3, 4, 5))
    frm2.add_load(n_elem2, fz=-P)
    frm2.solve()
    w_tip = frm2.node_disp(n_elem2)[2]
    w_tip_analytical = -P * L**3 / (3 * E * Iy)
    err2 = abs((w_tip - w_tip_analytical) / w_tip_analytical)
    print(f"[Test2: cantilever tip] numeric={w_tip:.6e} analytical={w_tip_analytical:.6e} err={err2*100:.4f}%")
    assert err2 < tol, "Cantilever self-test FAILED"

    # --- Test 3: singular / mechanism detection ---
    frm3 = Frame3D()
    frm3.add_node(0, 0, 0)
    frm3.add_node(1, 0, 0)
    frm3.add_element(0, 1, E, G, A, Iy, Iz, J, ref_vec=[0, 0, 1])
    frm3.add_load(1, fz=-P)
    # no supports at all -> singular
    try:
        frm3.solve()
        raised = False
    except np.linalg.LinAlgError:
        raised = True
    assert raised, "Mechanism-detection self-test FAILED (should have raised on singular K)"
    print("[Test3: mechanism detection] correctly raised LinAlgError on unsupported structure")

    print("ALL frame3d.py SELF-TESTS PASSED")


if __name__ == "__main__":
    _self_test()

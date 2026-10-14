#!/usr/bin/env python3
"""
impec1d_mrst.py -- our volume-balance IMPEC scheme, reconfigured to the
                   MRST eCPA benchmark used by the student's code.

What is kept from impec1d.py
    the IMPEC formulation itself: volume balance V_t(p,N) = phi as the
    pressure equation, numerical V_ti and dVt/dp, tridiagonal implicit
    pressure, explicit upwind transport of total moles N_i.

What is taken from the benchmark (so the comparison is meaningful)
    geometry, rock, BCs, dt                      -> Params
    Spycher-Pruess equilibrium with y_H2O = 0    -> spycher_pruess_eq
    Redlich-Kwong Z-factor for the CO2 phase     -> rk_z_factor
    Garcia (2001) apparent molar volume of CO2   -> flash
    Lohrenz-Bray-Clark liquid viscosity          -> liquid_viscosity_lbc
    quadratic Corey with Sw_irr = 0              -> relperm

Reference data: MRST_eCPA_results.csv at t = 45000 s.

Dependencies: numpy.  matplotlib and the csv module only for the comparison.
"""

import os
import sys
import csv
import numpy as np

# =====================================================================
# 0.  Constants
# =====================================================================

R = 8.314                       # J/(mol.K)
R_BAR = 83.14                   # bar.cm^3/(mol.K)  -- Spycher-Pruess units
MW_H2O = 0.018015268            # kg/mol
MW_CO2 = 0.0440098              # kg/mol
DARCY = 9.869233e-13            # m^2

ICO2, IH2O = 0, 1
NC = 2

PA_PER_PSI = 6894.757293168
CP_TO_PA_S = 1e-3

# LBC component data, order (H2O, CO2)
TC_LBC = np.array([647.096, 304.1282])          # K
PC_LBC = np.array([22.064e6, 7.3773e6])         # Pa
MW_LBC = np.array([MW_H2O, MW_CO2])             # kg/mol
RHOC_LBC = np.array([322.0, 467.600001282])     # kg/m^3
VC_LBC = MW_LBC / RHOC_LBC                      # m^3/mol


# =====================================================================
# 1.  Parameters -- all taken from the benchmark script
# =====================================================================

class Params:

    def __init__(self):
        # ---- grid ---------------------------------------------------
        self.L     = 200.0
        self.ncell = 100
        self.area  = 1.0

        # ---- rock ---------------------------------------------------
        self.phi  = 0.2                 # constant: no rock compressibility
        self.perm = 0.5 * DARCY

        # ---- state --------------------------------------------------
        self.T_K    = 313.15            # 40 degC
        self.p_res  = 10.0e6            # 100 bar, initial and outlet

        # ---- fluids -------------------------------------------------
        self.rho_w_ref   = 997.96624546  # kg/m^3 at p_res
        self.c_w         = 3.85e-10      # 1/Pa
        self.mu_g        = 0.06e-3       # Pa.s, CO2-rich phase (aqueous mu from LBC)
        self.rho_co2_inj = 600.0         # kg/m^3, used ONLY for the inlet flux

        # ---- injection ----------------------------------------------
        self.u_total = 1.0e-5            # m/s, prescribed Darcy velocity in

        # ---- relative permeability ----------------------------------
        self.Sw_irr = 0.0

        # ---- time ---------------------------------------------------
        self.dt    = 900.0
        self.t_end = 45000.0

        # ---- numerics -----------------------------------------------
        self.eps_N = 1.0e-6              # relative perturbation for dVt/dN
        self.eps_p = 1.0e3               # Pa, perturbation for dVt/dp

        # ---- options ------------------------------------------------
        # The benchmark sets a_mix = a_CO2 and b_mix = b_CO2, i.e. assumes
        # y_H2O = 0 in the mixing rule (Spycher's own simplification).
        #
        # a_CO2 slope: the student's script uses 4.13e4, but Table 1 of
        # Spycher et al. (2003) gives 4.02e4 -- verified against their own
        # tabulated a_CO2 (6.34e7 at 25 degC, 6.04e7 at 100 degC).  The
        # published value brings the RK CO2 density at 135.6 bar to 735
        # kg/m3 (MRST eCPA: 738; 4.13e4 gives 728) and slightly improves the
        # Sg match at every dt.  Its effect on pressure is a ~0.25 bar level
        # shift whose sign relative to MRST depends on dt, so the pressure
        # comparison alone does not discriminate between the two values.
        # Set to 4.13e4 to reproduce the student's run.
        self.a_co2_slope = 4.02e4

    @property
    def dx(self):
        return self.L / self.ncell

    @property
    def vcell(self):
        return self.dx * self.area

    @property
    def flux_in(self):
        """Prescribed inlet molar flux of pure CO2, mol/(m^2 s)."""
        return self.rho_co2_inj * self.u_total / MW_CO2


# =====================================================================
# 2.  Redlich-Kwong, in the Z-factor form used by the benchmark
# =====================================================================

def rk_AB(p_pa, T_k, par):
    """Dimensionless RK parameters.  a is converted from the Spycher units
    (bar cm^6 K^0.5 mol^-2) to SI by the factor 1e-7."""
    a_co2 = (7.54e7 - par.a_co2_slope * T_k) * 1e-7
    b_co2 = 27.80e-6
    A = a_co2 * p_pa / (R**2 * T_k**2.5)
    B = b_co2 * p_pa / (R * T_k)
    return a_co2, b_co2, A, B


def rk_z_factor(p_pa, T_k, par):
    """Largest real root of Z^3 - Z^2 + (A - B - B^2) Z - A B = 0."""
    _, _, A, B = rk_AB(p_pa, T_k, par)
    roots = np.roots([1.0, -1.0, A - B - B * B, -A * B])
    real = roots[np.abs(roots.imag) < 1e-8].real
    real = real[real > B]
    if real.size == 0:
        raise RuntimeError("no valid RK root at p = %.4e Pa" % p_pa)
    return real.max()


def fugacity_rk(p_pa, T_k, par):
    """Fugacity coefficients (phi_CO2, phi_H2O) in the CO2-rich phase.
    Mixing rule evaluated at y_H2O = 0, as the benchmark does."""
    a_co2, b_co2, A, B = rk_AB(p_pa, T_k, par)
    a_h2o_co2 = 7.89e7 * 1e-7
    b_h2o = 18.18e-6
    Z = rk_z_factor(p_pa, T_k, par)

    b_i = np.array([b_co2, b_h2o])
    sum_xa = np.array([a_co2, a_h2o_co2])

    lnphi = (b_i / b_co2 * (Z - 1.0)
             - np.log(Z - B)
             - A / B * (2.0 * sum_xa / a_co2 - b_i / b_co2) * np.log(1.0 + B / Z))
    return np.exp(lnphi)


# =====================================================================
# 3.  Spycher-Pruess equilibrium
# =====================================================================

def spycher_pruess_eq(p_pa, T_k, par):
    """Return (x_CO2_eq, y_CO2_eq): the CO2 mole fraction in the aqueous
    phase and in the CO2-rich phase at equilibrium."""
    p_bar = p_pa / 1e5
    T_c = T_k - 273.15
    phi_co2, phi_h2o = fugacity_rk(p_pa, T_k, par)

    K0_H2O = 10.0 ** (-2.209 + 3.097e-2 * T_c - 1.098e-4 * T_c**2 + 2.048e-7 * T_c**3)
    K0_CO2 = 10.0 ** (1.189 + 1.304e-2 * T_c - 5.446e-5 * T_c**2)

    A = (K0_H2O / (phi_h2o * p_bar)) * np.exp((p_bar - 1.0) * 18.1 / (R_BAR * T_k))
    B = (phi_co2 * p_bar / (55.508 * K0_CO2)) * np.exp(-(p_bar - 1.0) * 32.6 / (R_BAR * T_k))

    y_h2o = (1.0 - B) / (1.0 / A - B)
    x_co2 = B * (1.0 - y_h2o)
    return float(np.clip(x_co2, 0.0, 1.0)), float(np.clip(1.0 - y_h2o, 0.0, 1.0))


# =====================================================================
# 4.  Flash and phase densities
# =====================================================================

def flash(z, p_pa, T_k, par, eq=None):
    """Two-component flash at fixed (p, T).

    Because the equilibrium compositions depend only on (p, T), the phase
    split is the lever rule on the overall CO2 mole fraction:

        V = (z - x_eq) / (y_eq - x_eq)

    which is exactly Rachford-Rice for nc = 2.

    Returns (V, Sl, Sg, x_l, x_g, rho_l_molar, rho_g_molar).
    """
    z = float(np.clip(z, 1e-12, 1.0 - 1e-12))
    if eq is None:
        eq = spycher_pruess_eq(p_pa, T_k, par)
    x_eq, y_eq = eq

    if z <= x_eq:                       # single aqueous phase
        V, x_l, x_g = 0.0, z, y_eq
    elif z >= y_eq:                     # single CO2-rich phase
        V, x_l, x_g = 1.0, x_eq, z
    else:                               # two phases
        V = (z - x_eq) / (y_eq - x_eq)
        x_l, x_g = x_eq, y_eq

    # ---- aqueous density: Garcia (2001) apparent molar volume -------
    MW_l = x_l * MW_CO2 + (1.0 - x_l) * MW_H2O
    X_l_CO2 = x_l * MW_CO2 / MW_l                     # mass fraction
    T_c = T_k - 273.15
    Vphi = (37.51 - 9.585e-2 * T_c + 8.740e-4 * T_c**2 - 5.044e-7 * T_c**3) * 1e-6
    rho_co2_diss = MW_CO2 / Vphi
    rho_w_p = par.rho_w_ref * np.exp(par.c_w * (p_pa - par.p_res))
    rho_l_mass = 1.0 / ((1.0 - X_l_CO2) / rho_w_p + X_l_CO2 / rho_co2_diss)
    rho_l_molar = rho_l_mass / MW_l

    # ---- CO2-rich density from the RK Z-factor ----------------------
    Z_g = rk_z_factor(p_pa, T_k, par)
    rho_g_molar = p_pa / (Z_g * R * T_k)

    if V <= 0.0:
        Sg = 0.0
    elif V >= 1.0:
        Sg = 1.0
    else:
        vg = V / rho_g_molar
        Sg = vg / ((1.0 - V) / rho_l_molar + vg)

    return V, 1.0 - Sg, Sg, x_l, x_g, rho_l_molar, rho_g_molar


def liquid_viscosity_lbc(x_l_CO2, rho_l_molar, T_k):
    """Lohrenz-Bray-Clark correlation, as in the benchmark."""
    x_l = np.array([1.0 - x_l_CO2, x_l_CO2])       # order: H2O, CO2
    sqrt_MW = np.sqrt(1000.0 * MW_LBC)
    a = b = 0.0
    for i in range(2):
        Tr = T_k / TC_LBC[i]
        e_i = (5.4402 * (TC_LBC[i] * 1.8) ** (1.0 / 6.0)
               / (sqrt_MW[i] * (PC_LBC[i] / PA_PER_PSI) ** (2.0 / 3.0) * CP_TO_PA_S))
        if Tr > 1.5:
            mu_i = 17.78e-5 * (4.58 * Tr - 1.67) ** 0.625 / e_i
        else:
            mu_i = 34.0e-5 * Tr ** 0.94 / e_i
        a += x_l[i] * mu_i * sqrt_MW[i]
        b += x_l[i] * sqrt_MW[i]
    mu_atm = a / b

    T_pc = np.sum(x_l * TC_LBC)
    P_pc = np.sum(x_l * PC_LBC)
    Vc = np.sum(x_l * VC_LBC)
    MW_pc = np.sum(x_l * MW_LBC)
    e_mix = (5.4402 * (T_pc * 1.8) ** (1.0 / 6.0)
             / (np.sqrt(1000.0 * MW_pc) * (P_pc / PA_PER_PSI) ** (2.0 / 3.0) * CP_TO_PA_S))
    rho_r = Vc * rho_l_molar
    poly = (0.1023 + 0.023364 * rho_r + 0.058533 * rho_r**2
            - 0.040758 * rho_r**3 + 0.0093324 * rho_r**4)
    return mu_atm + (poly**4 - 1.0e-4) / e_mix


def relperm(Sw, Sg, par):
    """Quadratic Corey with Sw_irr = 0 and no residual gas."""
    Sw = min(max(Sw, 0.0), 1.0)
    Sg = min(max(Sg, 0.0), 1.0)
    krw = 0.0 if Sw <= par.Sw_irr else ((Sw - par.Sw_irr) / (1.0 - par.Sw_irr))**2
    krg = 0.0 if Sg <= 0.0 else (Sg / (1.0 - par.Sw_irr))**2
    return krw, krg


# =====================================================================
# 5.  Cell state and the IMPEC coefficients
# =====================================================================

def cell_state(p, N, par, eq=None):
    NT = N.sum()
    z = N[ICO2] / NT
    V, Sl, Sg, x_l, x_g, rho_l, rho_g = flash(z, p, par.T_K, par, eq)
    Vt = V * NT / rho_g + (1.0 - V) * NT / rho_l
    return dict(V=V, Sl=Sl, Sg=Sg, x_l=x_l, x_g=x_g,
                rho_l=rho_l, rho_g=rho_g, Vt=Vt, z=z)


def Vt_of(p, N, par, eq=None):
    return cell_state(p, N, par, eq)["Vt"]


def coefficients(p, N, par, eq=None):
    """V_ti = dVt/dN_i and dVt/dp, by finite differences of the flash."""
    if eq is None:
        eq = spycher_pruess_eq(p, par.T_K, par)
    Vt0 = Vt_of(p, N, par, eq)
    dVt_dp = (Vt_of(p + par.eps_p, N, par) - Vt0) / par.eps_p
    dN = max(par.eps_N * N.sum(), 1.0e-10)
    Vti = np.empty(NC)
    for i in range(NC):
        Np = N.copy()
        Np[i] += dN
        Vti[i] = (Vt_of(p, Np, par, eq) - Vt0) / dN
    return Vt0, dVt_dp, Vti


# =====================================================================
# 6.  Tridiagonal solve
# =====================================================================

def thomas(lower, diag, upper, rhs):
    n = len(diag)
    c = np.zeros(n)
    d = np.zeros(n)
    beta = diag[0]
    c[0] = upper[0] / beta
    d[0] = rhs[0] / beta
    for i in range(1, n):
        beta = diag[i] - lower[i] * c[i - 1]
        c[i] = upper[i] / beta if i < n - 1 else 0.0
        d[i] = (rhs[i] - lower[i] * d[i - 1]) / beta
    x = np.empty(n)
    x[-1] = d[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d[i] - c[i] * x[i + 1]
    return x


# =====================================================================
# 7.  Simulator
# =====================================================================

class Simulator:

    def __init__(self, par):
        self.par = par
        n = par.ncell

        self.p = np.full(n, par.p_res)
        self.N = np.zeros((n, NC))

        st0 = flash(1e-12, par.p_res, par.T_K, par)
        self.N[:, IH2O] = par.phi * st0[5]          # rho_l_molar at z ~ 0
        self.N[:, ICO2] = 0.0

        self.T_int = par.perm * par.area / par.dx
        self.T_out = 2.0 * par.perm * par.area / par.dx

        # inlet: prescribed molar flux of pure CO2 into cell 0
        self.q = np.zeros((n, NC))
        self.q[0, ICO2] = par.flux_in / par.dx

        self.t = 0.0
        self._update_state()

    # -----------------------------------------------------------------
    def _update_state(self):
        par = self.par
        n = par.ncell
        self.rho_l = np.empty(n)
        self.rho_g = np.empty(n)
        self.x_l = np.empty(n)
        self.x_g = np.empty(n)
        self.Sg = np.empty(n)
        self.mu_l = np.empty(n)
        self.lam_w = np.empty(n)
        self.lam_g = np.empty(n)
        self.Vt = np.empty(n)
        self.dVt_dp = np.empty(n)
        self.Vti = np.empty((n, NC))
        self.z = np.empty(n)

        for j in range(n):
            eq = spycher_pruess_eq(self.p[j], par.T_K, par)
            st = cell_state(self.p[j], self.N[j], par, eq)
            Vt, dVtdp, Vti = coefficients(self.p[j], self.N[j], par, eq)

            self.rho_l[j] = st["rho_l"]
            self.rho_g[j] = st["rho_g"]
            self.x_l[j] = st["x_l"]
            self.x_g[j] = st["x_g"]
            self.Sg[j] = st["Sg"]
            self.z[j] = st["z"]
            self.Vt[j] = Vt
            self.dVt_dp[j] = dVtdp
            self.Vti[j] = Vti

            mu = liquid_viscosity_lbc(st["x_l"], st["rho_l"], par.T_K)
            self.mu_l[j] = mu
            krw, krg = relperm(st["Sl"], st["Sg"], par)
            self.lam_w[j] = krw / mu
            self.lam_g[j] = krg / par.mu_g

        self.vol_err = np.max(np.abs(self.Vt - par.phi) / par.phi)

    # -----------------------------------------------------------------
    def _pressure_system(self):
        par = self.par
        n = par.ncell
        Vc = par.vcell

        lower = np.zeros(n)
        diag = np.zeros(n)
        upper = np.zeros(n)
        rhs = np.zeros(n)

        Cp = -self.dVt_dp                       # phi is constant here
        diag += Vc * Cp / par.dt
        rhs += Vc * Cp / par.dt * self.p
        rhs += Vc * np.sum(self.Vti * self.q, axis=1)
        rhs += Vc * (self.Vt - par.phi) / par.dt        # volume-balance correction

        for f in range(1, n):
            jl, jr = f - 1, f
            up = jl if self.p[jl] >= self.p[jr] else jr
            cw = self.rho_l[up] * np.array([self.x_l[up], 1.0 - self.x_l[up]])
            cg = self.rho_g[up] * np.array([self.x_g[up], 1.0 - self.x_g[up]])
            lw, lg = self.lam_w[up], self.lam_g[up]

            th_l = self.T_int * (lw * np.dot(self.Vti[jl], cw)
                                 + lg * np.dot(self.Vti[jl], cg))
            th_r = self.T_int * (lw * np.dot(self.Vti[jr], cw)
                                 + lg * np.dot(self.Vti[jr], cg))
            diag[jl] += th_l
            upper[jl] -= th_l
            diag[jr] += th_r
            lower[jr] -= th_r

        j = n - 1
        cw = self.rho_l[j] * np.array([self.x_l[j], 1.0 - self.x_l[j]])
        cg = self.rho_g[j] * np.array([self.x_g[j], 1.0 - self.x_g[j]])
        th = self.T_out * (self.lam_w[j] * np.dot(self.Vti[j], cw)
                           + self.lam_g[j] * np.dot(self.Vti[j], cg))
        diag[j] += th
        rhs[j] += th * par.p_res
        return lower, diag, upper, rhs

    # -----------------------------------------------------------------
    def _face_flows(self, p_new):
        par = self.par
        n = par.ncell
        Qw = np.zeros(n + 1)
        Qg = np.zeros(n + 1)
        up_idx = np.zeros(n + 1, dtype=int)
        for f in range(1, n):
            jl, jr = f - 1, f
            up = jl if p_new[jl] >= p_new[jr] else jr
            dp = p_new[jl] - p_new[jr]
            Qw[f] = self.T_int * self.lam_w[up] * dp
            Qg[f] = self.T_int * self.lam_g[up] * dp
            up_idx[f] = up
        j = n - 1
        dp = p_new[j] - par.p_res
        Qw[n] = self.T_out * self.lam_w[j] * dp
        Qg[n] = self.T_out * self.lam_g[j] * dp
        up_idx[n] = j
        return Qw, Qg, up_idx

    # -----------------------------------------------------------------
    def step(self):
        par = self.par
        lower, diag, upper, rhs = self._pressure_system()
        p_new = thomas(lower, diag, upper, rhs)

        Qw, Qg, up_idx = self._face_flows(p_new)
        n = par.ncell
        ndot = np.zeros((n + 1, NC))
        for f in range(1, n + 1):
            up = up_idx[f]
            cw = self.rho_l[up] * np.array([self.x_l[up], 1.0 - self.x_l[up]])
            cg = self.rho_g[up] * np.array([self.x_g[up], 1.0 - self.x_g[up]])
            ndot[f] = cw * Qw[f] + cg * Qg[f]

        N_new = self.N.copy()
        for j in range(n):
            N_new[j] += par.dt / par.vcell * (ndot[j] - ndot[j + 1])
        N_new += par.dt * self.q
        N_new = np.maximum(N_new, 0.0)

        self.p, self.N = p_new, N_new
        self._update_state()
        self.t += par.dt
        self.u_face = (Qw + Qg) / par.area
        self.produced = ndot[-1] * par.dt

    # -----------------------------------------------------------------
    def run(self, verbose=True):
        par = self.par
        injected = 0.0
        produced = np.zeros(NC)
        while self.t < par.t_end - 1e-9:
            self.step()
            injected += par.flux_in * par.area * par.dt
            produced += self.produced
            if verbose and (round(self.t) % 9000 == 0):
                print("  t = %6.0f s   p_in = %7.3f bar   Sg_max = %.4f"
                      "   z_max = %.4f   volerr = %.2e"
                      % (self.t, self.p[0] / 1e5, self.Sg.max(),
                         self.z.max(), self.vol_err))
        in_place = np.sum(self.N[:, ICO2]) * par.vcell
        self.balance = injected - produced[ICO2] - in_place
        self.injected = injected
        return self


# =====================================================================
# 8.  Comparison against the MRST reference
# =====================================================================

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, "MRST_eCPA_results.csv")   # default: next to this script


def find_reference():
    """Reference CSV from --ref PATH, else MRST_eCPA_results.csv beside the
    script or in the current directory. Returns None if not found."""
    if "--ref" in sys.argv:
        i = sys.argv.index("--ref")
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1] if os.path.exists(sys.argv[i + 1]) else None
    for path in (REF, os.path.join(os.getcwd(), "MRST_eCPA_results.csv")):
        if os.path.exists(path):
            return path
    return None


def load_reference(path=REF):
    cols = {}
    with open(path, newline="") as fh:
        rd = csv.DictReader(fh)
        for name in rd.fieldnames:
            cols[name] = []
        for row in rd:
            for name in rd.fieldnames:
                cols[name].append(float(row[name]))
    return {k: np.array(v) for k, v in cols.items()}


def compare(sim, ref):
    par = sim.par
    x = np.linspace(0.5 * par.dx, par.L - 0.5 * par.dx, par.ncell)

    def report(name, ours, theirs, scale=None):
        s = scale if scale is not None else max(np.max(np.abs(theirs)), 1e-30)
        rmse = np.sqrt(np.mean((ours - theirs) ** 2))
        linf = np.max(np.abs(ours - theirs))
        print("  %-14s RMSE = %10.4e  (%6.2f %% of range)   Linf = %10.4e"
              % (name, rmse, 100 * rmse / s, linf))

    print("\n--- ours vs MRST eCPA at t = %.0f s ---" % sim.t)
    report("z_CO2", sim.z, ref["z_CO2"])
    report("Sg", sim.Sg, ref["Sg"])
    report("pressure [bar]", sim.p / 1e5, ref["pressure_Pa"] / 1e5,
           scale=(ref["pressure_Pa"].max() - ref["pressure_Pa"].min()) / 1e5)
    rho_l_mass = sim.rho_l * (sim.x_l * MW_CO2 + (1 - sim.x_l) * MW_H2O)
    report("rhoL [kg/m3]", rho_l_mass, ref["rhoL_kg_m3"])
    report("muL [Pa.s]", sim.mu_l, ref["muL_Pa_s"])

    print("\n  cell    x     z_ours    z_MRST     Sg_ours   Sg_MRST"
          "    p_ours    p_MRST  [bar]")
    for j in list(range(8)) + [15, 30, 60, 99]:
        print("  %4d %5.0f  %9.5f %9.5f  %9.5f %9.5f  %9.3f %9.3f"
              % (j, x[j], sim.z[j], ref["z_CO2"][j], sim.Sg[j], ref["Sg"][j],
                 sim.p[j] / 1e5, ref["pressure_Pa"][j] / 1e5))
    return x


def main(plot=True):
    par = Params()
    print("Benchmark configuration")
    print("  L = %.0f m, Nx = %d, dx = %.1f m, k = %.3g m2, phi = %.2f"
          % (par.L, par.ncell, par.dx, par.perm, par.phi))
    print("  T = %.2f K, p_out = %.1f bar, u_inj = %.1e m/s"
          % (par.T_K, par.p_res / 1e5, par.u_total))
    print("  inlet CO2 molar flux = %.6f mol/(m2 s)" % par.flux_in)
    x_eq, y_eq = spycher_pruess_eq(par.p_res, par.T_K, par)
    print("  Spycher-Pruess at 100 bar: x_CO2_eq = %.6f, y_CO2_eq = %.6f"
          % (x_eq, y_eq))
    print("  RK CO2 density at 100 bar = %.1f kg/m3"
          % (par.p_res / (rk_z_factor(par.p_res, par.T_K, par) * R * par.T_K) * MW_CO2))

    sim = Simulator(par).run()
    print("\n  CO2 injected  = %.4e mol" % sim.injected)
    print("  balance error = %.3e  (%.2e relative)"
          % (sim.balance, abs(sim.balance) / sim.injected))

    ref_path = find_reference()
    if ref_path is None:
        print("\nMRST reference not found -- skipping the comparison.")
        print("Put MRST_eCPA_results.csv next to this script, or run with")
        print("  python impec1d_mrst.py --ref /path/to/MRST_eCPA_results.csv")
        return sim
    print("\nreference: %s" % ref_path)
    ref = load_reference(ref_path)
    x = compare(sim, ref)

    if plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return sim
        fig, ax = plt.subplots(3, 1, figsize=(7, 9), sharex=True)
        ax[0].plot(x, sim.z, "-", label="ours (IMPEC + Spycher-Pruess)")
        ax[0].plot(ref["x_m"], ref["z_CO2"], "o", ms=3, label="MRST eCPA")
        ax[0].set_ylabel("$z_{CO_2}$")
        ax[1].plot(x, sim.Sg, "-")
        ax[1].plot(ref["x_m"], ref["Sg"], "o", ms=3)
        ax[1].set_ylabel("$S_g$")
        ax[2].plot(x, sim.p / 1e5, "-")
        ax[2].plot(ref["x_m"], ref["pressure_Pa"] / 1e5, "o", ms=3)
        ax[2].set_ylabel("pressure [bar]")
        ax[2].set_xlabel("distance [m]")
        for a in ax:
            a.grid(alpha=0.3)
        ax[0].legend()
        ax[0].set_xlim(0, 40)
        fig.suptitle("1D IMPEC vs MRST eCPA, t = 45000 s")
        fig.tight_layout()
        fig.savefig("impec1d_mrst.png", dpi=140)
        print("\nwrote impec1d_mrst.png")
    return sim


if __name__ == "__main__":
    main(plot="--no-plot" not in sys.argv)
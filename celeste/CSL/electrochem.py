'''
Electrochemical reactors
'''

from . import interfaces, ModelBase
from . import equilibrium as eq
from .exergy import T0
from ..jx import nasa as _nasa
from .. import solvers
import numpy as np
import scipy.optimize as spo
from celeste import functions as funs

F_CONST = 96485  #: Faraday constant used by the electrochemical models, C/mol


class SOC(ModelBase):
    r'''
    Solid oxide cell stack (electrolysis for I < 0, fuel cell for I > 0) with the fuel side at
    water-gas-shift and methane-reforming equilibrium.

    The fuel outlet follows from three reaction extents,

        n_out = n_in + xi_EC*nu_EC + xi_WGS*nu_WGS + xi_MSR*nu_MSR

    with the electrochemical extent fixed by the current, xi_EC = Ncells*I/(2F) (H2 + O -> H2O on the
    fuel side), so the element balances hold by construction. The unknowns are xi_WGS and xi_MSR, found
    from the mass-action laws in log form, and the outlet temperature when a heat duty is specified.

    Energy specification (one of them):
        T:  isothermal stack, outlet temperature T [K] (default 1073.15); Q is computed
        Q:  heat added to the stack [kJ/s]; Q=0 is adiabatic and the outlet temperature is solved

    Ideal voltage (``V_model``):
        'mean':    the reversible voltage averaged over the charge passed, Vid = int_0^1 E(lambda) dlambda,
                   where E = -dG_r(T, x, P)/(2F) is the local Nernst potential of H2 + 1/2 O2 -> H2O at the
                   local equilibrium composition after a fraction lambda of the charge, with T linear in
                   lambda from the fuel inlet to the outlet temperature. For an isothermal stack fed with an
                   equilibrium mixture this equals the Gibbs energy approach, Vid = (G_out - G_in)/(-I*Ncells).
                   (Gauss-Legendre quadrature with ``n_nodes`` points.)
        'outlet':  Nernst potential at the outlet, the highest (electrolysis) / lowest (fuel cell)
                   local value, i.e. the limit for a stack held at one voltage.
        'streams': Gibbs energy difference of the streams, (G_out - G_in)/(-I*Ncells), the former model.
                   With a temperature change it also contains -int S dT and is no longer a voltage.
    The three values are available as ``V_mean``, ``V_nernst_out`` and ``V_streams`` whatever the choice.

    Parameters:
        name:      objects name
        MediumF:   fuel side medium with H2, CH4, CO2, CO, H2O, N2 (media.Fuel_CH4())
        MediumA:   air side medium, media.Air() (O2, N2)
        I:         current per cell in [A]; negative in electrolysis mode
        cellArea:  active cell area in [cm2]
        Ncells:    number of cells
        f_ASR:     multiplier on the area specific resistance correlation
        Qloss:     heat exchanged with the surroundings in [kJ/s]; like Q and W positive into the stack, so a
                   loss is negative (Qloss = -5 means 5 kW lost)
        T, Q:      energy specification, see above
        V_model:   'mean' (default), 'outlet' or 'streams'
        n_nodes:   quadrature points of the 'mean' voltage

    Deprecated: ``Tsoc0``/``T_soc0`` (set ``F0.T``/``A0.T``), ``Tsoc1``/``T_soc1`` (use ``T``),
    ``P_soc`` (set ``F0.P``/``A0.P``), ``extent`` (no effect).

    Ports: F0 -> F1 (fuel side), A0 -> A1 (air side). The outlets leave at the stack (outlet) temperature
    and the fuel inlet pressure.

    Properties:
        Vid, Vop:  reversible and operating cell voltage [V]
        j:         current density [A/cm2]
        W:         electrical power into the stack [kJ/s]
        Q:         heat into the stack [kJ/s]
        xi:        extents (EC, WGS, MSR) [mol/s]
        RC:        reactant conversion [-]
    '''
    reactions = {
        'EC':  {'H2': -1, 'H2O': 1},                         # H2 + O(2-) -> H2O + 2e-
        'WGS': eq.REACTIONS['WGS'],                          # CO + H2O <-> CO2 + H2
        'MSR': eq.REACTIONS['MSR'],                          # CH4 + H2O <-> CO + 3 H2
    }
    V_models = ('mean', 'outlet', 'streams')

    def __init__(self, name, MediumF, MediumA, Tsoc0:float|None=None, Tsoc1:float|None=None, I:float=0.0,
                 cellArea:float=100., Ncells:int|float=1, f_ASR:float=1., extent:bool|None=None, Qloss:float=0.0, *,
                 T:float|None=None, Q:float|None=None, V_model:str='mean', n_nodes:int=12,
                 T_soc0:float|None=None, T_soc1:float|None=None, P_soc:float|None=None):
        if V_model not in self.V_models:
            raise ValueError(f"{name}: V_model must be one of {self.V_models}, not {V_model!r}")
        T_in = T_soc0 if T_soc0 is not None else Tsoc0
        T_out = T_soc1 if T_soc1 is not None else Tsoc1
        deprecated_kw = [k for k, v in (('Tsoc0', Tsoc0), ('T_soc0', T_soc0), ('Tsoc1', Tsoc1), ('T_soc1', T_soc1),
                                        ('P_soc', P_soc), ('extent', extent)) if v is not None]
        if T_out is not None:
            if T is not None or Q is not None:
                raise ValueError(f"{name}: give the outlet temperature as T only")
            T = T_out
        if T is not None and Q is not None:
            raise ValueError(f"{name}: give either T (isothermal) or Q (heat duty), not both")
        if T is None and Q is None:
            T = 1073.15
        super().__init__(dict(name=name, MediumF=MediumF, MediumA=MediumA, I=I, cellArea=cellArea, Ncells=Ncells,
                              f_ASR=f_ASR, Qloss=Qloss, V_model=V_model, n_nodes=n_nodes,
                              Tsoc0=T_in, Tsoc1=T_out, extent=bool(extent)))
        self.T_spec, self.Q_spec = T, Q
        T_init = T_in if T_in is not None else (T if T is not None else 1073.15)
        P0 = 1. if P_soc is None else P_soc
        self.F0 = interfaces.Flange(name + '.F0', Medium=MediumF, T=T_init, P=P0)
        self.F1 = interfaces.Flange(name + '.F1', Medium=MediumF, T=T if T is not None else T_init, P=P0)
        self.A0 = interfaces.Flange(name + '.A0', Medium=MediumA, T=T_init, P=P0)
        self.A1 = interfaces.Flange(name + '.A1', Medium=MediumA, T=T if T is not None else T_init, P=P0)

        species = [str(sp) for sp in MediumF.species]
        self._nu = np.zeros((3, len(species)))
        for r, stoich in enumerate(self.reactions.values()):
            for sp, v in stoich.items():
                self._nu[r, species.index(sp)] = v
        self._i = {sp: species.index(sp) for sp in ('H2', 'CH4', 'CO2', 'CO', 'H2O')}
        self._iO2 = [str(sp) for sp in MediumA.species].index('O2')
        self._gdata = _nasa.nasa_data(('H2', 'H2O', 'O2'))
        self._N_H = np.array([2,4,0,0,2,0])
        self._N_C = np.array([0,1,1,1,0,0])
        self._N_O = np.array([0,0,2,1,1,0])
        self._N_N = np.array([0,0,0,0,0,2])
        self._xi = np.zeros(3)
        self._V_mean = np.nan
        self._ASR_mean = np.nan

        if deprecated_kw:
            from .._deprecation import warn_deprecated
            warn_deprecated(f"SOC({', '.join(deprecated_kw)}=...) is deprecated: give the outlet temperature as T= "
                            f"(or a heat duty Q=), set the inlet state on the flanges (unit.F0.T = ..., unit.F0.P = ...); "
                            f"extent has no effect, the ideal voltage excludes the reforming reactions",
                            stacklevel=2)

    # ---- energy specification ---------------------------------------------------------------------
    @property
    def isothermal(self) -> bool:
        return self.Q_spec is None

    @property
    def T(self):
        '''Stack (outlet) temperature: the specified T, or the solved outlet temperature.'''
        return self.T_spec if self.T_spec is not None else self.F1.T

    @T.setter
    def T(self, value):
        '''Setting T makes the stack isothermal at T.'''
        self.T_spec, self.Q_spec = value, None

    @property
    def Q(self):
        '''Heat added to the stack [kJ/s] (net of Qloss).'''
        return self.F1.n*self.F1.h + self.A1.n*self.A1.h - self.F0.n*self.F0.h - self.A0.n*self.A0.h - self.W - self.Qloss

    @Q.setter
    def Q(self, value):
        '''Setting Q (0 for adiabatic) makes the outlet temperature follow from the energy balance.'''
        self.Q_spec, self.T_spec = value, None

    @property
    def P_soc(self):
        return self.F1.P

    @P_soc.setter
    def P_soc(self, P):
        self.F1.P = P
        self.A0.P = P
        self.A1.P = P

    # ---- stoichiometry ------------------------------------------------------------------------------
    @property
    def n_O(self):
        '''Oxygen ions transferred to the fuel side [mol/s] (negative in electrolysis); the EC extent.'''
        return self.Ncells*self.I/2/F_CONST

    @property
    def xi(self):
        '''Extents of the electrochemical, water-gas-shift and reforming reactions [mol/s].'''
        return self._xi.copy()

    @property
    def MSR_extent(self):
        return float(self._xi[2])

    def _n_fuel(self, xi_EC, xi):
        return self.F0.n*np.asarray(self.F0.Xi, dtype=float) + xi_EC*self._nu[0] + xi @ self._nu[1:]

    def _air(self, lam):
        '''Air-side O2 and total flow after a fraction lam of the charge.'''
        n_O2 = self.A0.n*self.A0.Xi[self._iO2] - 0.5*lam*self.n_O
        return n_O2, self.A0.n - 0.5*lam*self.n_O

    def _extent_bounds(self, xi_EC):
        '''WGS / MSR extent range that keeps every mole number non-negative.'''
        b = self.F0.n*np.asarray(self.F0.Xi, dtype=float) + xi_EC*self._nu[0]
        key = tuple(b)
        cache = self.__dict__.setdefault('_bounds_cache', {})
        if key in cache:
            return cache[key]
        if len(cache) > 256:
            cache.clear()
        lo, hi = np.empty(2), np.empty(2)
        for r in range(2):
            for sign, out in ((1.0, hi), (-1.0, lo)):
                c = np.zeros(2); c[r] = -sign
                res = spo.linprog(c, A_ub=-self._nu[1:].T, b_ub=b, bounds=[(None, None)]*2, method='highs')
                out[r] = sign*(-res.fun) if res.status == 0 else sign*np.inf
        cache[key] = (lo, hi)
        return lo, hi

    def _log_residuals(self, xi, xi_EC, T, P, lnK=None):
        """ln(Q/K) of the water-gas shift and reforming reactions for the extents xi."""
        lnK = self._lnK(T) if lnK is None else lnK
        n = np.maximum(self._n_fuel(xi_EC, xi), 1e-300)
        nu = self._nu[1:]
        return nu @ np.log(n) + nu.sum(axis=1)*np.log(P/n.sum()) - lnK

    def _lnK(self, T):
        return np.log([eq.K_WGS(T), eq.K_MSR(T)])

    def _equilibrium(self, lam, T, xi0=None, warn=True):
        """WGS / MSR extents at equilibrium after a fraction lam of the charge, at T (damped Newton)."""
        xi_EC = lam*self.n_O
        P = self.F0.P
        lnK = self._lnK(T)
        lo, hi = self._extent_bounds(xi_EC)
        span = np.where(np.isfinite(hi - lo), hi - lo, self.F0.n)
        lo_b, hi_b = lo + 1e-12*span, hi - 1e-12*span
        x = np.clip(self._xi[1:] if xi0 is None else np.asarray(xi0, dtype=float), lo_b, hi_b)
        if np.any(self._n_fuel(xi_EC, x) <= 0):
            x = np.where(np.isfinite(lo) & np.isfinite(hi), 0.5*(lo + hi), 0.0)
        fun = lambda v: self._log_residuals(v, xi_EC, T, P, lnK)
        nu = self._nu[1:]
        dnu = nu.sum(axis=1)
        r = fun(x)
        for _ in range(100):
            if np.max(np.abs(r)) < 1e-12:
                break
            n = self._n_fuel(xi_EC, x)
            J = (nu/n) @ nu.T - np.outer(dnu, dnu)/n.sum()
            try:
                dx = np.linalg.solve(J, -r)
            except np.linalg.LinAlgError:
                break
            # stay inside the feasible region (all mole numbers positive)
            dn = dx @ nu
            neg = dn < 0
            t = min(1.0, 0.95*np.min(-n[neg]/dn[neg])) if np.any(neg) else 1.0
            x_new = x + t*dx
            r_new = fun(x_new)
            while np.linalg.norm(r_new) > np.linalg.norm(r) and t > 1e-8:
                t *= 0.5
                x_new = x + t*dx
                r_new = fun(x_new)
            if np.allclose(x_new, x, rtol=0, atol=1e-15*self.F0.n):
                break
            x, r = x_new, r_new
        newton = spo.OptimizeResult(x=x, fun=r, success=bool(np.max(np.abs(r)) <= 1e-10),
                                    residual_norm=float(np.max(np.abs(r))), method='newton')
        if newton.success or not warn:
            # (without warn: a bracketing evaluation, e.g. at a temperature limit, where trace species can drop
            # below round-off and the mass-action law cannot be met to the tolerance; no fallback needed)
            return newton
        return solvers.solve(fun, x, bounds=(lo_b, hi_b), tol=1e-10, name=self.name, warn=warn)

    def _set_outlets(self, xi, T):
        n = self._n_fuel(self.n_O, xi)
        n_safe = np.maximum(n, 1e-14*max(self.F0.n, 1e-300))
        self._xi = np.concatenate(([self.n_O], xi))
        P = self.F0.P
        self.F1.T, self.F1.P = T, P
        self.F1.n = float(n.sum())
        self.F1.Xi = n_safe/n_safe.sum()
        n_O2, n_air = self._air(1.0)
        self.A1.T, self.A1.P = T, P
        self.A0.P = P
        self.A1.n = n_air
        self.A1.Xi = np.array([n_O2/n_air, 1 - n_O2/n_air], dtype=float) if self._iO2 == 0 else \
            np.array([1 - n_O2/n_air, n_O2/n_air], dtype=float)

    # ---- voltages -----------------------------------------------------------------------------------
    def nernst(self, T, P, x_H2, x_H2O, x_O2):
        '''Nernst potential of H2 + 1/2 O2 -> H2O [V] at T [K], P [bar] and the given mole fractions.'''
        _, h, s = _nasa.pure_props(np, T, self._gdata.a, self._gdata.T_break, self._gdata.T_lo, self._gdata.T_hi)
        g0 = h - T*s                                               # kJ/mol, H2, H2O, O2
        dG = g0[1] - g0[0] - 0.5*g0[2] + _nasa.R*T*(np.log(x_H2O) - np.log(x_H2) - 0.5*np.log(x_O2*P))
        return -dG*1e3/(2*F_CONST)

    def _profile(self, T1, warn=True):
        '''Charge-averaged Nernst potential and ASR, T linear in the charge from F0.T to T1.'''
        if self.n_O == 0:
            return np.nan, float(self.ASR(T1))
        nodes, weights = np.polynomial.legendre.leggauss(int(self.n_nodes))
        lam, w = 0.5*(nodes + 1), 0.5*weights
        E, A, xi = 0.0, 0.0, self._xi[1:]
        for l, wk in zip(lam, w):
            Tl = self.F0.T + l*(T1 - self.F0.T)
            res = self._equilibrium(l, Tl, xi, warn=warn)
            xi = res.x
            n = np.maximum(self._n_fuel(l*self.n_O, xi), 1e-300)
            n_O2, n_air = self._air(l)
            E += wk*self.nernst(Tl, self.F0.P, n[self._i['H2']]/n.sum(), n[self._i['H2O']]/n.sum(), n_O2/n_air)
            A += wk*self.ASR(Tl)
        return float(E), float(A)

    @property
    def V_mean(self):
        '''Charge-averaged Nernst potential [V] (computed by run_sol).'''
        return self._V_mean

    @property
    def V_nernst_out(self):
        '''Nernst potential at the outlet [V].'''
        x = np.asarray(self.F1.Xi)
        return float(self.nernst(self.F1.T, self.F1.P, x[self._i['H2']], x[self._i['H2O']], self.A1.Xi[self._iO2]))

    @property
    def V_nernst_in(self):
        '''Nernst potential of the inlet streams as fed [V].'''
        x = np.asarray(self.F0.Xi)
        return float(self.nernst(self.F0.T, self.F0.P, x[self._i['H2']], x[self._i['H2O']], self.A0.Xi[self._iO2]))

    @property
    def V_streams(self):
        '''(G_out - G_in)/(-I*Ncells) of the streams [V]; the former ideal voltage.'''
        dG = self.F1.n*self.F1.g + self.A1.n*self.A1.g - self.F0.n*self.F0.g - self.A0.n*self.A0.g
        return dG*1e3/(-self.I*self.Ncells)

    @property
    def Vid(self):
        '''Ideal (reversible) cell voltage [V] of the chosen ``V_model``.'''
        return {'mean': self.V_mean, 'outlet': self.V_nernst_out, 'streams': self.V_streams}[self.V_model]

    @property
    def Wid(self):
        '''Reversible electrical power [kJ/s].'''
        return self.Vid*(-self.I)*self.Ncells*1e-3

    @property
    def j(self):
        return self.I/self.cellArea

    @property
    def ASR_eff(self):
        '''Area specific resistance used for the ohmic loss [Ohm cm2]: averaged like the voltage ('mean') or at the outlet.'''
        return self._ASR_mean if self.V_model == 'mean' else float(self.ASR())

    @property
    def Vop(self):
        return self.Vid - self.f_ASR*self.ASR_eff*self.j

    @property
    def W(self):
        return self.Vop*(-self.I)*self.Ncells*1e-3

    def heat_and_work(self) -> float:
        """Q + W + Qloss: ``Q`` is defined net of ``Qloss``, so the energy balance needs all three."""
        return float(self.Q + self.W + self.Qloss)

    # ---- second law ---------------------------------------------------------------------------------
    @property
    def Exth(self):
        # Q crosses the boundary at the stack temperature; Qloss at T0 and carries no exergy
        return self.Q*(1 - (T0/self.F1.T))

    @property
    def Exloss(self):
        return self.F0.Ex*self.F0.n + self.A0.Ex*self.A0.n - self.F1.Ex*self.F1.n - self.A1.Ex*self.A1.n + self.W + self.Exth

    @property
    def Sgen(self):
        return self.F1.s*self.F1.n + self.A1.s*self.A1.n - self.F0.s*self.F0.n - self.A0.s*self.A0.n - self.Q/self.F1.T - self.Qloss/T0

    # ---- performance --------------------------------------------------------------------------------
    @property
    def RC(self):
        if self.I<0:
            val=-self.Ncells*self.I/(F_CONST*(self.F0.n*(2*self.F0.Xi[2] + 2*self.F0.Xi[4])))
        else:
            val=self.Ncells*self.I/(2*F_CONST*self.F0.n*(self.F0.Xi[0] + 4*self.F0.Xi[1] + self.F0.Xi[3]))
        return val

    @property
    def RC_EC(self):
        return (self.F0.n*(self.F0.Xi[2] + self.F0.Xi[4]) - self.F1.n*(self.F1.Xi[2] + self.F1.Xi[4]))/(
                self.F0.n*(self.F0.Xi[2] + self.F0.Xi[4]))

    def K_WGS(self):
        return eq.K_WGS(self.F1.T)

    def K_MSR(self):
        return eq.K_MSR(self.F1.T)

    def ASR(self, T=None):
        # M. Riedel 2020 DOI 10.1149/1945-7111/ab6820, Ohm cm2
        T = self.F1.T if T is None else T
        return 7.28e2*np.exp(-8.29e-3*(T - 273.15))

    def cform(self, side='out'):
        f = self.F1 if side == 'out' else self.F0
        return funs.carbonFraction(f.T, f.P, dict(zip(f.Medium.species, f.Xi)))

    @property
    def EnthalpyEfficiency(self):
        LHV_H2 = self.F1.Medium.h_i[4] - self.F1.Medium.h_i[0] - 0.5*self.A1.Medium.h_i[0]
        LHV_CO = self.F1.Medium.h_i[2] - self.F1.Medium.h_i[3] - 0.5*self.A1.Medium.h_i[0]
        LHV_CH4 = self.F1.Medium.h_i[2] + 2*self.F1.Medium.h_i[4] - self.F1.Medium.h_i[3] - 2*self.A1.Medium.h_i[0]
        LHV_F0 = np.dot(-np.array([LHV_H2, LHV_CH4, 0, LHV_CO, 0, 0]), self.F0.Xi)*self.F0.n
        LHV_F1 = np.dot(-np.array([LHV_H2, LHV_CH4, 0, LHV_CO, 0, 0]), self.F1.Xi)*self.F1.n
        return (LHV_F1 - LHV_F0)/self.W

    # ---- solution -----------------------------------------------------------------------------------
    def _solve_at(self, T1, warn=False):
        '''Outlet equilibrium (and the charge-averaged voltage) at the outlet temperature T1; returns the equilibrium solve.'''
        res = self._equilibrium(1.0, T1, warn=warn)
        self._set_outlets(res.x, T1)
        if self.V_model == 'mean':
            self._V_mean, self._ASR_mean = self._profile(T1, warn=warn)
        else:
            self._V_mean, self._ASR_mean = np.nan, np.nan
        return res

    def f_sol(self, x):
        '''Residuals for the unknowns x = (xi_WGS, xi_MSR[, T_out]); T_out only with a heat duty.'''
        x = np.atleast_1d(np.asarray(x, dtype=float))
        T1 = self.T_spec if self.isothermal else float(x[2])
        self._set_outlets(x[:2], T1)
        if self.V_model == 'mean':
            self._V_mean, self._ASR_mean = self._profile(T1, warn=False)
        return self.Eqs()

    def Eqs(self):
        '''Residuals at the current state: the two equilibria in log form (and the heat duty).'''
        r = list(self._log_residuals(self._xi[1:], self.n_O, self.F1.T, self.F0.P))
        if not self.isothermal:
            r.append((self.Q - self.Q_spec)/max(self.F0.n, 1e-300))
        self.Eq = np.array(r)
        return self.Eq

    def run_sol(self, x0=None, n_scan=None):
        '''Solve for the extents (and the outlet temperature for a heat duty). x0 and n_scan are not used.'''
        if self.isothermal:
            res = self._solve_at(self.T_spec, warn=True)
        else:
            def balance(T1):
                self._solve_at(T1)
                return (self.Q - self.Q_spec)/max(self.F0.n, 1e-300)
            x0 = float(self.F1.T) if 400.0 < self.F1.T < 2000.0 else float(self.F0.T)
            Tres = solvers.solve_scalar(balance, 400.0, 2000.0, x0=x0, tol=1e-10, name=self.name)
            T1 = float(Tres.x[0])
            res = self._solve_at(T1, warn=True)
            res.T = T1
            if not Tres.success:
                res.success = False
        res.residual_norm = float(np.max(np.abs(self.Eqs())))
        if not res.success or res.residual_norm > 1e-6:
            res.success = False
            import warnings
            warnings.warn(f'{self.name}: not converged (max |residual| = {res.residual_norm:.3e})',
                          solvers.ConvergenceWarning, stacklevel=2)
        return res


class AEL(ModelBase):
    '''
    Alkaline electrolyser model
    It determines the current density [A/cm2] (j) and outlet mole flow and composition for a given mole flow of water

    Parameters:
        name:   objects name
        Medium: media.Media_AEL() (H2O(L), H2, O2, H2O)
        T, P:   operating temperature [K] and pressure [bar]
        j:      initial guess of the current density [A/cm2], negative for electrolysis
        Ncells: number of cells
        A:      active cell area [cm2]

    Ports: F0 (water in) -> F1 (hydrogen side out), F2 (oxygen out)
    '''
    def __init__(self, name, Medium, T=298.15, P=1, j=-0.32, Ncells=None, A=None,
                 F0_n=1, F0_Xi=np.array([1, 0, 0, 0]), 
                 F1_n=1, F1_Xi=None, #['H2O(L)', 'H2', 'O2', 'H2O']
                 F2_n=1, F2_Xi=np.array([0, 0, 1, 0]) 
                ): 
                
        '''
        media.Media_AEL()
        '''
        super().__init__(locals())
        
        self.F0 = interfaces.Flange(name + '.F0', Medium=Medium, n=F0_n, Xi=F0_Xi, T=T, P=P)
        self.F1 = interfaces.Flange(name + '.F1', Medium=Medium, n=F1_n, Xi=F1_Xi, T=T, P=P) #H2
        self.F2 = interfaces.Flange(name + '.F2', Medium=Medium, n=F2_n, Xi=F2_Xi, T=T, P=P) #O2
        self._j = j
        self.A = A
        
    def Eqs(self):
        y = np.array((
            [self.F0.Xi[0]+self.F0.Xi[3], self.F2.Xi[0]+self.F2.Xi[3], self.F1.Xi[0]+self.F1.Xi[3], 1], # H2O
            [self.F0.Xi[1], self.F2.Xi[1], self.F1.Xi[1], -1], # H2
            [self.F0.Xi[2], self.F2.Xi[2], self.F1.Xi[2], -0.5])) # O2
        
        n = np.array([self.F0.n, - self.F2.n, -self.F1.n, self.Ncells*self.I/2/F_CONST])

        self.Eq = np.concatenate((
            [
            np.sum(self.F1.Xi) - 1,
            ],
            np.dot(y, n),
        ))
            
        self.F1.P = self.F0.P
        self.F2.P = self.F0.P  
        
        return self.Eq

    @property
    def j(self):
        return self._j
    
    @j.setter
    def j(self, val):
        self._j = val
    
    @property
    def I(self):
        return self.j*self.A
    
    @property
    def Vop(self):
        return 1.481 + 2.9*1e-3*self.F0.P - self.j*1.05

    @property
    def W(self):
        return -self.I*self.Vop*self.Ncells*1e-3 # to kJ/s

    @property
    def Q(self):
        return self.F2.n*self.F2.h + self.F1.n*self.F1.h - self.F0.n*self.F0.h - self.W
    
    @property
    def Exth(self):
        return self.Q*(1 - 298.15/self.F1.T)
    
    @property
    def Exloss(self):
        return self.F0.n*self.F0.Ex - self.F1.n*self.F1.Ex - self.F2.n*self.F2.Ex + self.Exth + self.W
    
    @property
    def Exeff(self):
        return 1 - self.Exloss/(self.F0.n*self.F0.Ex + np.max([0, self.Exth]) +  self.W)
        
    def f_sol(self, x):
        self.j = x[0]
        self.F1.n = x[1]
        self.F1.Xi = np.array([0, x[2], 0.001, 0.002])
        self.F2.n = x[3]
        return self.Eqs()
        
    def run_sol(self):
        x0 = np.array([self.j,self.F0.n,0.997,self.F0.n])  
        return solvers.solve(self.f_sol, x0, scale=self.F0.n, name=self.name)

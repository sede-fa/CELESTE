"""
Heat exchanger network synthesis (HENS).

:class:`HENSynthesiser` takes the same stream table as
:class:`celeste.HEN.pinch.PinchAnalyser` (``SysBase.pinch_stream_table()``) and
returns a complete network: which streams exchange heat, in which order, with
which duty, temperatures and area, and where utilities are needed.

Method
------
The stage-wise superstructure of Yee and Grossmann (1990) is used: every hot
stream may meet every cold stream once in each of ``n_stages`` stages, stream
splits are mixed isothermally, utilities sit at the stream ends. The full
problem (topology + areas) is a non-convex MINLP. It is solved here by a
decomposition that needs nothing but SciPy and JAX:

1. **Topology (MILP).** With isothermal mixing all constraints of the
   superstructure are linear. Minimising utility cost plus a fixed charge per
   exchanger is therefore a mixed-integer *linear* program, solved to proven
   optimality with HiGHS (``scipy.optimize.milp``). The minimum approach
   temperature ``dt_min`` is enforced at both ends of every exchanger.
2. **Loads and areas (NLP).** For the chosen matches the stream temperatures
   are linear in the exchanger duties, so the total annual cost including the
   area cost is minimised over the duties under linear constraints (SLSQP with
   exact gradients from JAX).

Units follow the stream table: temperatures in degC, duties in kW, heat
capacity flow rates in kW/K; film coefficients in kW/m2/K, areas in m2, costs
per year.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import optimize as spo
from scipy import sparse

from .pinch import PinchAnalyser

__all__ = ["HENSynthesiser", "HENCosts", "Utility"]


@dataclass
class HENCosts:
    """
    Cost model (annualised). The defaults are placeholders of a realistic order of
    magnitude - replace them with project figures.

    exchanger cost per year = ``unit_fixed + area_coeff * area**area_exp``
    """
    hot_utility: float = 100.0      #: per kW and year
    cold_utility: float = 15.0      #: per kW and year
    unit_fixed: float = 8000.0      #: per exchanger and year
    area_coeff: float = 500.0       #: per m2**area_exp and year
    area_exp: float = 0.8


@dataclass
class Utility:
    """A utility stream: inlet / outlet temperature in degC and film coefficient in kW/m2/K."""
    T_in: float
    T_out: float
    h: float = 1.0


def _lmtd(dt1, dt2, xp=np):
    """Chen's approximation of the logarithmic mean temperature difference (smooth, no singularity at dt1 = dt2)."""
    return (dt1*dt2*0.5*(dt1 + dt2))**(1.0/3.0)


class HENSynthesiser:
    """
    Synthesis of a heat exchanger network from a stream table.

    Parameters
    ----------
    segments : DataFrame in the format of ``PinchAnalyser`` / ``SysBase.pinch_stream_table()``:
               columns ``segment_type`` ('sensible' | 'latent'), ``side`` ('hot' | 'cold'), ``T_in``, ``T_out``
               [degC], ``CP`` [kW/K] for sensible and ``T_phase`` [degC], ``Q`` [kW] for latent segments;
               the stream name is the index or a column ``segment_id``. An optional column ``h`` holds
               film heat transfer coefficients [kW/m2/K] (default ``h_default``).
    dt_min   : minimum approach temperature of every exchanger [K]
    n_stages : number of stages of the superstructure (default: max(number of hot, cold streams), at most 4)
    hot_utility, cold_utility : :class:`Utility`; by default placed just outside the temperature range of the streams
    costs    : :class:`HENCosts`
    latent_dT : temperature glide given to isothermal (latent) segments so that they have a finite CP [K]
    h_default : film coefficient for streams without an ``h`` entry [kW/m2/K]

    Example
    -------
    >>> table = system.pinch_stream_table()
    >>> hen = HENSynthesiser(table, dt_min=10).solve()
    >>> hen.summary()
    >>> hen.network          # process-process exchangers
    >>> hen.utilities        # heaters and coolers
    >>> hen.plot_grid_diagram()
    """

    def __init__(self, segments: pd.DataFrame, dt_min: float, n_stages: int | None = None,
                 hot_utility: Utility | None = None, cold_utility: Utility | None = None,
                 costs: HENCosts | None = None, latent_dT: float = 1.0, h_default: float = 1.0):
        self.dt_min = float(dt_min)
        if not np.isfinite(self.dt_min) or self.dt_min <= 0:
            raise ValueError("dt_min must be a positive finite number.")
        self.costs = costs or HENCosts()
        self.latent_dT = float(latent_dT)
        self.segments = segments.copy()
        self.streams = self._streams(segments, h_default)

        hot = self.streams[self.streams.side == "hot"]
        cold = self.streams[self.streams.side == "cold"]
        self._hot, self._cold = hot, cold
        self.n_stages = int(n_stages) if n_stages else max(1, min(max(len(hot), len(cold)), 4))

        margin = max(self.dt_min, 10.0)
        if hot_utility is None:
            top = cold.T_out.max() if len(cold) else hot.T_in.max()
            hot_utility = Utility(top + margin + 1.0, top + margin, h_default)
        if cold_utility is None:
            bottom = hot.T_out.min() if len(hot) else cold.T_in.min()
            cold_utility = Utility(bottom - margin - 5.0, bottom - margin, h_default)
        self.hot_utility, self.cold_utility = hot_utility, cold_utility
        if len(cold) and min(hot_utility.T_in, hot_utility.T_out) < cold.T_out.max() + self.dt_min - 1e-9:
            raise ValueError("The hot utility is too cold to heat every cold stream to its target with dt_min.")
        if len(hot) and max(cold_utility.T_in, cold_utility.T_out) > hot.T_out.min() - self.dt_min + 1e-9:
            raise ValueError("The cold utility is too warm to cool every hot stream to its target with dt_min.")

        self._solution = None
        self.network = None
        self.utilities = None
        self.milp_status = None

    # ------------------------------------------------------------------
    # input
    # ------------------------------------------------------------------
    def _streams(self, segments: pd.DataFrame, h_default: float) -> pd.DataFrame:
        seg = segments.copy()
        seg.columns = seg.columns.str.strip()
        if "segment_id" not in seg.columns:
            seg = seg.rename_axis("segment_id").reset_index()
        PinchAnalyser(seg, self.dt_min)        # same validation rules as the pinch analysis
        rows = []
        for _, r in seg.iterrows():
            side = str(r["side"]).strip().lower()
            if str(r["segment_type"]).strip().lower() == "latent":
                glide = self.latent_dT if side == "cold" else -self.latent_dT
                T_in, T_out, CP = float(r["T_phase"]), float(r["T_phase"]) + glide, float(r["Q"])/self.latent_dT
            else:
                T_in, T_out, CP = float(r["T_in"]), float(r["T_out"]), float(r["CP"])
            h = r["h"] if "h" in seg.columns and pd.notna(r["h"]) else h_default
            rows.append(dict(stream=str(r["segment_id"]), side=side, T_in=T_in, T_out=T_out, CP=CP,
                             Q=CP*abs(T_in - T_out), h=float(h)))
        streams = pd.DataFrame(rows).set_index("stream")
        if streams.index.has_duplicates:
            raise ValueError("Stream names (segment_id) must be unique.")
        return streams

    # ------------------------------------------------------------------
    # step 1: topology (MILP)
    # ------------------------------------------------------------------
    def _solve_milp(self, time_limit, mip_gap):
        hot, cold, K, dt = self._hot, self._cold, self.n_stages, self.dt_min
        nh, nc = len(hot), len(cold)
        Fh, Fc = hot.CP.values, cold.CP.values
        Thi, Tho, Tci, Tco = hot.T_in.values, hot.T_out.values, cold.T_in.values, cold.T_out.values
        Qh, Qc = hot.Q.values, cold.Q.values

        # variable layout
        n_q = nh*nc*K
        o_q, o_z = 0, n_q
        o_th = 2*n_q
        o_tc = o_th + nh*(K + 1)
        o_qcu = o_tc + nc*(K + 1)
        o_qhu = o_qcu + nh
        o_zcu = o_qhu + nc
        o_zhu = o_zcu + nh
        n = o_zhu + nc
        q = lambda i, j, k: o_q + (i*nc + j)*K + k
        z = lambda i, j, k: o_z + (i*nc + j)*K + k
        th = lambda i, k: o_th + i*(K + 1) + k
        tc = lambda j, k: o_tc + j*(K + 1) + k

        rows, cols, vals, lb, ub = [], [], [], [], []

        def add(coeffs, lo, hi):
            r = len(lb)
            for c, v in coeffs:
                rows.append(r); cols.append(c); vals.append(v)
            lb.append(lo); ub.append(hi)

        for i in range(nh):
            # overall and stage heat balances, cooler duty
            add([(q(i, j, k), 1.0) for j in range(nc) for k in range(K)] + [(o_qcu + i, 1.0)], Qh[i], Qh[i])
            for k in range(K):
                add([(th(i, k), Fh[i]), (th(i, k + 1), -Fh[i])] + [(q(i, j, k), -1.0) for j in range(nc)], 0.0, 0.0)
            add([(th(i, K), Fh[i]), (o_qcu + i, -1.0)], Fh[i]*Tho[i], Fh[i]*Tho[i])
            add([(o_qcu + i, 1.0), (o_zcu + i, -Qh[i])], -np.inf, 0.0)
        for j in range(nc):
            add([(q(i, j, k), 1.0) for i in range(nh) for k in range(K)] + [(o_qhu + j, 1.0)], Qc[j], Qc[j])
            for k in range(K):
                add([(tc(j, k), Fc[j]), (tc(j, k + 1), -Fc[j])] + [(q(i, j, k), -1.0) for i in range(nh)], 0.0, 0.0)
            add([(tc(j, 0), -Fc[j]), (o_qhu + j, -1.0)], -Fc[j]*Tco[j], -Fc[j]*Tco[j])
            add([(o_qhu + j, 1.0), (o_zhu + j, -Qc[j])], -np.inf, 0.0)
        for i in range(nh):
            for j in range(nc):
                omega = min(Qh[i], Qc[j])
                gamma = max(0.0, Tci[j] - Thi[i], Tci[j] - Tho[i], Tco[j] - Thi[i], Tco[j] - Tho[i]) + dt
                for k in range(K):
                    add([(q(i, j, k), 1.0), (z(i, j, k), -omega)], -np.inf, 0.0)
                    # approach temperature at the hot and the cold end of the exchanger
                    add([(th(i, k), 1.0), (tc(j, k), -1.0), (z(i, j, k), -gamma)], dt - gamma, np.inf)
                    add([(th(i, k + 1), 1.0), (tc(j, k + 1), -1.0), (z(i, j, k), -gamma)], dt - gamma, np.inf)

        lo, hi = np.zeros(n), np.full(n, np.inf)
        integrality = np.zeros(n)
        for i in range(nh):
            for j in range(nc):
                for k in range(K):
                    hi[q(i, j, k)] = min(Qh[i], Qc[j])
                    hi[z(i, j, k)] = 1.0
                    integrality[z(i, j, k)] = 1
            lo[th(i, 0)] = hi[th(i, 0)] = Thi[i]                 # inlet temperature
            for k in range(1, K + 1):
                lo[th(i, k)], hi[th(i, k)] = Tho[i], Thi[i]      # monotonic by the stage balances (q >= 0)
            hi[o_qcu + i] = Qh[i]
            hi[o_zcu + i] = 1.0
            integrality[o_zcu + i] = 1
        for j in range(nc):
            lo[tc(j, K)] = hi[tc(j, K)] = Tci[j]
            for k in range(K):
                lo[tc(j, k)], hi[tc(j, k)] = Tci[j], Tco[j]
            hi[o_qhu + j] = Qc[j]
            hi[o_zhu + j] = 1.0
            integrality[o_zhu + j] = 1

        c = np.zeros(n)
        cost = self.costs
        c[o_qcu:o_qcu + nh] = cost.cold_utility
        c[o_qhu:o_qhu + nc] = cost.hot_utility
        c[o_z:o_z + n_q] = cost.unit_fixed
        c[o_zcu:o_zcu + nh] = cost.unit_fixed
        c[o_zhu:o_zhu + nc] = cost.unit_fixed
        # tie-break between topologies of equal cost: prefer large driving forces (smaller areas)
        scale = 1e-6*max(cost.unit_fixed, 1.0)/max(Qh.sum() + Qc.sum(), 1.0)
        for i in range(nh):
            for j in range(nc):
                for k in range(K):
                    c[q(i, j, k)] = scale/max(Thi[i] - Tci[j], self.dt_min)

        A = sparse.csr_matrix((vals, (rows, cols)), shape=(len(lb), n))
        res = spo.milp(c, constraints=spo.LinearConstraint(A, np.array(lb), np.array(ub)), integrality=integrality,
                       bounds=spo.Bounds(lo, hi), options=dict(time_limit=time_limit, mip_rel_gap=mip_gap))
        if res.x is None:
            raise RuntimeError(f"HEN topology problem could not be solved: {res.message}")
        self.milp_status = dict(status=int(res.status), message=str(res.message), optimal=bool(res.status == 0),
                                gap=float(getattr(res, "mip_gap", np.nan)))
        x = res.x
        qm = np.array([[[x[q(i, j, k)] for k in range(K)] for j in range(nc)] for i in range(nh)]).reshape(nh, nc, K)
        zm = np.array([[[x[z(i, j, k)] for k in range(K)] for j in range(nc)] for i in range(nh)]).reshape(nh, nc, K) > 0.5
        qm = np.where(zm & (qm > 1e-9*max(Qh.sum(), 1.0)), qm, 0.0)
        return qm

    # ------------------------------------------------------------------
    # network evaluation (shared by the NLP and the reports)
    # ------------------------------------------------------------------
    def _temperatures(self, q, xp=np):
        """Stage temperatures th (nh, K+1) and tc (nc, K+1) for the duties q (nh, nc, K)."""
        Fh, Fc = xp.asarray(self._hot.CP.values), xp.asarray(self._cold.CP.values)
        Thi, Tci = xp.asarray(self._hot.T_in.values), xp.asarray(self._cold.T_in.values)
        qh = xp.sum(q, axis=1)                      # (nh, K) heat released by hot stream i in stage k
        qc = xp.sum(q, axis=0)                      # (nc, K)
        zero_h, zero_c = xp.zeros((qh.shape[0], 1)), xp.zeros((qc.shape[0], 1))
        th = Thi[:, None] - xp.concatenate([zero_h, xp.cumsum(qh, axis=1)], axis=1)/Fh[:, None]
        tc = Tci[:, None] + xp.concatenate([xp.cumsum(qc[:, ::-1], axis=1)[:, ::-1], zero_c], axis=1)/Fc[:, None]
        return th, tc

    def _u(self):
        hh, hc = self._hot.h.values, self._cold.h.values
        U = 1.0/(1.0/hh[:, None] + 1.0/hc[None, :])
        U_cu = 1.0/(1.0/hh + 1.0/self.cold_utility.h)
        U_hu = 1.0/(1.0/hc + 1.0/self.hot_utility.h)
        return U, U_cu, U_hu

    def _annual_cost(self, q, active, cu_on, hu_on, xp=np):
        """Total annual cost for duties q; ``active`` etc. are the (fixed) sets of existing exchangers."""
        cost, eps = self.costs, 1e-9
        U, U_cu, U_hu = self._u()
        th, tc = self._temperatures(q, xp)
        Qh, Qc = xp.asarray(self._hot.Q.values), xp.asarray(self._cold.Q.values)
        qcu = Qh - xp.sum(q, axis=(1, 2))
        qhu = Qc - xp.sum(q, axis=(0, 2))
        dt_hot = th[:, None, :-1] - tc[None, :, :-1]
        dt_cold = th[:, None, 1:] - tc[None, :, 1:]
        lm = _lmtd(xp.maximum(dt_hot, eps), xp.maximum(dt_cold, eps), xp)
        area = q/(xp.asarray(U)[:, :, None]*lm)
        # coolers: hot stream th[:, K] -> T_out against the cold utility; heaters: tc[:, 0] -> T_out against the hot utility
        cu, hu = self.cold_utility, self.hot_utility
        Tho, Tco = xp.asarray(self._hot.T_out.values), xp.asarray(self._cold.T_out.values)
        lm_cu = _lmtd(xp.maximum(th[:, -1] - cu.T_out, eps), xp.maximum(Tho - cu.T_in, eps), xp)
        lm_hu = _lmtd(xp.maximum(hu.T_in - Tco, eps), xp.maximum(hu.T_out - tc[:, 0], eps), xp)
        a_cu = xp.maximum(qcu, 0.0)/(xp.asarray(U_cu)*lm_cu)
        a_hu = xp.maximum(qhu, 0.0)/(xp.asarray(U_hu)*lm_hu)
        smooth = lambda a: (a + 1e-8)**cost.area_exp        # finite slope at zero area
        capital = (cost.area_coeff*(xp.sum(smooth(area)*active) + xp.sum(smooth(a_cu)*cu_on) + xp.sum(smooth(a_hu)*hu_on))
                   + cost.unit_fixed*(active.sum() + cu_on.sum() + hu_on.sum()))
        utilities = cost.hot_utility*xp.sum(qhu) + cost.cold_utility*xp.sum(qcu)
        return capital + utilities, (area, a_cu, a_hu, qcu, qhu, th, tc, lm, lm_cu, lm_hu)

    # ------------------------------------------------------------------
    # step 2: loads and areas (NLP)
    # ------------------------------------------------------------------
    def _refine(self, q0):
        import jax
        import jax.numpy as jnp
        from ..jx import enable_x64
        enable_x64()

        nh, nc, K = q0.shape
        active = q0 > 0.0
        idx = np.argwhere(active)
        if len(idx) == 0:
            return q0
        Qh, Qc = self._hot.Q.values, self._cold.Q.values
        tol = 1e-9*max(Qh.sum() + Qc.sum(), 1.0)
        cu_on = (Qh - q0.sum(axis=(1, 2))) > tol
        hu_on = (Qc - q0.sum(axis=(0, 2))) > tol

        def expand(v, xp=np):
            if xp is np:
                q = np.zeros((nh, nc, K)); q[tuple(idx.T)] = v
                return q
            return jnp.zeros((nh, nc, K)).at[tuple(idx.T)].set(v)

        # all constraints are linear in the duties: build them from the (linear) temperature map
        base = self._temperatures(np.zeros((nh, nc, K)))
        m = len(idx)
        Jh = np.zeros((nh, K + 1, m)); Jc = np.zeros((nc, K + 1, m))
        for col in range(m):
            e = np.zeros(m); e[col] = 1.0
            th, tc = self._temperatures(expand(e))
            Jh[:, :, col], Jc[:, :, col] = th - base[0], tc - base[1]
        A, lo, hi = [], [], []
        for (i, j, k) in idx:
            for loc in (k, k + 1):
                A.append(Jh[i, loc] - Jc[j, loc]); lo.append(self.dt_min - (base[0][i, loc] - base[1][j, loc])); hi.append(np.inf)
        for i in range(nh):
            row = np.array([1.0 if a == i else 0.0 for a, _, _ in idx])
            A.append(row); hi.append(Qh[i]); lo.append(-np.inf if cu_on[i] else Qh[i])   # no new cooler
        for j in range(nc):
            row = np.array([1.0 if b == j else 0.0 for _, b, _ in idx])
            A.append(row); hi.append(Qc[j]); lo.append(-np.inf if hu_on[j] else Qc[j])   # no new heater
        A, lo, hi = np.array(A), np.array(lo), np.array(hi)

        act, cu_j, hu_j = jnp.asarray(active, dtype=float), jnp.asarray(cu_on, dtype=float), jnp.asarray(hu_on, dtype=float)
        scale = max(float(self._annual_cost(q0, active, cu_on, hu_on)[0]), 1.0)
        fun = jax.jit(jax.value_and_grad(lambda v: self._annual_cost(expand(v, jnp), act, cu_j, hu_j, jnp)[0]/scale))

        def objective(v):
            f, g = fun(v)
            return float(f), np.asarray(g)

        v0 = q0[tuple(idx.T)]
        eq = lo == hi
        constraints = []
        if eq.any():
            constraints.append(dict(type="eq", fun=lambda v: A[eq] @ v - lo[eq], jac=lambda v: A[eq]))
        ineq_lo, ineq_hi = (~eq) & np.isfinite(lo), (~eq) & np.isfinite(hi)
        if ineq_lo.any():
            constraints.append(dict(type="ineq", fun=lambda v: A[ineq_lo] @ v - lo[ineq_lo], jac=lambda v: A[ineq_lo]))
        if ineq_hi.any():
            constraints.append(dict(type="ineq", fun=lambda v: hi[ineq_hi] - A[ineq_hi] @ v, jac=lambda v: -A[ineq_hi]))
        res = spo.minimize(objective, v0, jac=True, method="SLSQP", bounds=[(0.0, None)]*m, constraints=constraints,
                           options=dict(maxiter=300, ftol=1e-10))
        v = np.maximum(res.x, 0.0)
        feasible = (np.all(A @ v >= lo - 1e-6*np.maximum(1.0, np.abs(lo))) and np.all(A @ v <= hi + 1e-6*np.maximum(1.0, np.abs(hi))))
        if not feasible or not np.isfinite(res.fun) or res.fun > objective(v0)[0]:
            return q0                    # keep the feasible MILP point if the refinement did not improve on it
        return expand(v)

    # ------------------------------------------------------------------
    # driver and reports
    # ------------------------------------------------------------------
    def solve(self, refine: bool = True, time_limit: float = 60.0, mip_gap: float = 1e-4):
        """
        Synthesise the network.

        Parameters
        ----------
        refine     : optimise duties and areas for the total annual cost after the topology search
        time_limit : time limit of the MILP in seconds (the best network found so far is used afterwards)
        mip_gap    : relative optimality gap at which the MILP stops
        """
        if len(self._hot) == 0 or len(self._cold) == 0:
            q = np.zeros((len(self._hot), len(self._cold), self.n_stages))
            self.milp_status = dict(status=0, message="no process-process matches possible", optimal=True, gap=0.0)
        else:
            q = self._solve_milp(time_limit, mip_gap)
            if refine:
                q = self._refine(q)
            q = np.where(q > 1e-7*max(self._hot.Q.sum(), 1.0), q, 0.0)
        self._build_reports(q)
        return self

    def _build_reports(self, q):
        hot, cold, cost = self._hot, self._cold, self.costs
        active = q > 0.0
        Qh, Qc = hot.Q.values, cold.Q.values
        tol = 1e-7*max(Qh.sum() + Qc.sum(), 1.0)
        cu_on = (Qh - q.sum(axis=(1, 2))) > tol
        hu_on = (Qc - q.sum(axis=(0, 2))) > tol
        total, (area, a_cu, a_hu, qcu, qhu, th, tc, lm, lm_cu, lm_hu) = self._annual_cost(q, active, cu_on, hu_on)
        U, U_cu, U_hu = self._u()
        unit_cost = lambda a: cost.unit_fixed + cost.area_coeff*a**cost.area_exp

        rows = []
        for i, j, k in np.argwhere(active):
            rows.append(dict(hot=hot.index[i], cold=cold.index[j], stage=k + 1, Q=q[i, j, k],
                             T_hot_in=th[i, k], T_hot_out=th[i, k + 1], T_cold_in=tc[j, k + 1], T_cold_out=tc[j, k],
                             dT_hot_end=th[i, k] - tc[j, k], dT_cold_end=th[i, k + 1] - tc[j, k + 1],
                             LMTD=lm[i, j, k], U=U[i, j], area=area[i, j, k], cost=unit_cost(area[i, j, k])))
        cols = ["hot", "cold", "stage", "Q", "T_hot_in", "T_hot_out", "T_cold_in", "T_cold_out", "dT_hot_end",
                "dT_cold_end", "LMTD", "U", "area", "cost"]
        self.network = pd.DataFrame(rows, columns=cols).sort_values(["stage", "hot", "cold"]).reset_index(drop=True)

        rows = []
        for i in np.flatnonzero(cu_on):
            rows.append(dict(stream=hot.index[i], type="cooler", Q=qcu[i], T_in=th[i, -1], T_out=hot.T_out.iloc[i],
                             LMTD=lm_cu[i], U=U_cu[i], area=a_cu[i], cost=unit_cost(a_cu[i])))
        for j in np.flatnonzero(hu_on):
            rows.append(dict(stream=cold.index[j], type="heater", Q=qhu[j], T_in=tc[j, 0], T_out=cold.T_out.iloc[j],
                             LMTD=lm_hu[j], U=U_hu[j], area=a_hu[j], cost=unit_cost(a_hu[j])))
        self.utilities = pd.DataFrame(rows, columns=["stream", "type", "Q", "T_in", "T_out", "LMTD", "U", "area", "cost"])

        self._solution = dict(q=q, th=th, tc=tc, qcu=np.where(cu_on, qcu, 0.0), qhu=np.where(hu_on, qhu, 0.0))

    def _require_solved(self):
        if self._solution is None:
            raise RuntimeError("HENSynthesiser has not been solved yet.")

    @property
    def hot_utility_load(self) -> float:
        self._require_solved()
        return float(self._solution["qhu"].sum())

    @property
    def cold_utility_load(self) -> float:
        self._require_solved()
        return float(self._solution["qcu"].sum())

    @property
    def heat_recovery(self) -> float:
        self._require_solved()
        return float(self._solution["q"].sum())

    def targets(self) -> pd.Series:
        """Energy targets of the pinch analysis for the same streams and dt_min."""
        return PinchAnalyser(self.segments, self.dt_min).solve().summary()

    def summary(self) -> pd.Series:
        """Key figures of the network next to the pinch targets."""
        self._require_solved()
        cost = self.costs
        target = self.targets()
        capital = float(self.network.cost.sum() + self.utilities.cost.sum())
        utility = cost.hot_utility*self.hot_utility_load + cost.cold_utility*self.cold_utility_load
        return pd.Series({
            "dt_min": self.dt_min,
            "n_stages": self.n_stages,
            "hot_utility": self.hot_utility_load,
            "cold_utility": self.cold_utility_load,
            "heat_recovery": self.heat_recovery,
            "hot_utility_target": target["hot_utility"],
            "cold_utility_target": target["cold_utility"],
            "n_exchangers": len(self.network),
            "n_heaters": int((self.utilities.type == "heater").sum()),
            "n_coolers": int((self.utilities.type == "cooler").sum()),
            "total_area": float(self.network.area.sum() + self.utilities.area.sum()),
            "capital_cost": capital,
            "utility_cost": utility,
            "total_annual_cost": capital + utility,
            "milp_optimal": self.milp_status["optimal"],
            "milp_gap": self.milp_status["gap"],
        })

    def check(self, tol: float = 1e-6) -> pd.Series:
        """Largest violations of the stream energy balances and of dt_min (all should be ~0)."""
        self._require_solved()
        s = self._solution
        scale = max(self._hot.Q.sum() + self._cold.Q.sum(), 1.0)
        hot_err = np.abs(s["q"].sum(axis=(1, 2)) + s["qcu"] - self._hot.Q.values).max(initial=0.0)
        cold_err = np.abs(s["q"].sum(axis=(0, 2)) + s["qhu"] - self._cold.Q.values).max(initial=0.0)
        approach = min(self.network[["dT_hot_end", "dT_cold_end"]].min().min(), np.inf) if len(self.network) else np.inf
        return pd.Series({"hot_balance_error": hot_err, "cold_balance_error": cold_err,
                          "min_approach": approach,
                          "feasible": bool(hot_err < tol*scale and cold_err < tol*scale and approach >= self.dt_min - 1e-6)})

    def plot_grid_diagram(self, ax=None):
        """Grid diagram: hot streams run left to right, cold streams right to left, exchangers as vertical links."""
        import matplotlib.pyplot as plt
        self._require_solved()
        hot, cold, K = self._hot, self._cold, self.n_stages
        names = list(hot.index) + list(cold.index)
        y = {name: len(names) - 1 - r for r, name in enumerate(names)}
        if ax is None:
            fig, ax = plt.subplots(figsize=(3.0 + 2.2*K, 1.2 + 0.8*len(names)))
        else:
            fig = ax.figure
        x0, x1 = 0.0, K + 1.0
        for name in hot.index:
            ax.annotate("", xy=(x1, y[name]), xytext=(x0, y[name]), arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.8))
            ax.text(x0 - 0.05, y[name], f"{name}  {hot.T_in[name]:.0f}°C", ha="right", va="center", fontsize=8)
            ax.text(x1 + 0.05, y[name], f"{hot.T_out[name]:.0f}°C", ha="left", va="center", fontsize=8)
        for name in cold.index:
            ax.annotate("", xy=(x0, y[name]), xytext=(x1, y[name]), arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1.8))
            ax.text(x0 - 0.05, y[name], f"{name}  {cold.T_out[name]:.0f}°C", ha="right", va="center", fontsize=8)
            ax.text(x1 + 0.05, y[name], f"{cold.T_in[name]:.0f}°C", ha="left", va="center", fontsize=8)
        slots = {}
        for _, r in self.network.iterrows():
            n_in_stage = slots.get(r.stage, 0)
            slots[r.stage] = n_in_stage + 1
            count = (self.network.stage == r.stage).sum()
            x = r.stage - 0.5 + (n_in_stage + 1)/(count + 1)
            ax.plot([x, x], [y[r.hot], y[r.cold]], color="0.25", lw=1.2, zorder=2)
            for name in (r.hot, r.cold):
                ax.plot(x, y[name], "o", ms=11, mfc="white", mec="0.25", zorder=3)
            ax.text(x + 0.05, y[r.hot] - 0.3, f"{r.Q:.1f}", fontsize=7, va="center")
        for _, r in self.utilities.iterrows():
            x = x1 - 0.25 if r.type == "cooler" else x0 + 0.25
            colour = "tab:blue" if r.type == "cooler" else "tab:red"
            ax.plot(x, y[r.stream], "o", ms=13, mfc=colour, mec="0.25", alpha=0.85, zorder=3)
            ax.text(x, y[r.stream], "C" if r.type == "cooler" else "H", color="white", ha="center", va="center", fontsize=7, zorder=4)
            ax.text(x, y[r.stream] + 0.28, f"{r.Q:.1f}", ha="center", fontsize=7)
        for k in range(1, K):
            ax.axvline(k + 0.5, color="0.85", lw=0.8, ls=":")
        ax.set_xlim(x0 - 1.4, x1 + 0.7)
        ax.set_ylim(-0.7, len(names) - 0.3)
        ax.set_axis_off()
        ax.set_title(f"Heat exchanger network  (ΔT_min = {self.dt_min:g} K, duties in kW)", fontsize=9)
        fig.tight_layout()
        return fig, ax

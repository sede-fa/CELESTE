"""
Compressor and turbine performance maps with design-point scaling,
smooth interpolation, plotting and Dymola export.
"""
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator, RegularGridInterpolator, interp1d

from .dymola_io import export_1d_table, export_2d_map

# 'pchip' : C1, shape-preserving (no overshoot) -> default, safe for least-squares
# 'cubic' : C2 spline, much faster per call but can overshoot between sparse grid points
# 'linear': original behaviour (kinks at grid lines -> noisy finite-difference Jacobians)
_METHODS = ('linear', 'pchip', 'cubic')


# ---------------------------------------------------------------------- #
#  Result dataclasses                                                    #
# ---------------------------------------------------------------------- #
@dataclass
class CompressorState:
    """Compressor operating point returned by CompressorMap.query()."""
    ncorr: float
    beta: float
    wcorr: float
    PR: float
    eta: float
    feasible: bool            # False if the query was clipped to the map domain or is in surge
    in_surge: bool = False
    message: str = ""

    def __str__(self):
        lines = [
            f"  feasible   : {self.feasible}",
            f"  in_surge   : {self.in_surge}",
            f"  Ncorr      : {self.ncorr:.4f}",
            f"  beta       : {self.beta:.4f}",
            f"  Wcorr      : {self.wcorr:.4f}",
            f"  PR         : {self.PR:.4f}",
            f"  efficiency : {self.eta:.4f}",
        ]
        if self.message:
            lines.append(f"  message    : {self.message}")
        return "\n".join(lines)


@dataclass
class TurbineState:
    """Turbine operating point returned by TurbineMap.query()."""
    ncorr: float
    beta: float
    wcorr: float
    PR: float
    eta: float
    feasible: bool            # False if the query was clipped to the map domain
    message: str = ""

    def __str__(self):
        lines = [
            f"  feasible   : {self.feasible}",
            f"  Ncorr      : {self.ncorr:.4f}",
            f"  beta       : {self.beta:.4f}",
            f"  Wcorr      : {self.wcorr:.4f}",
            f"  PR         : {self.PR:.4f}",
            f"  efficiency : {self.eta:.4f}",
        ]
        if self.message:
            lines.append(f"  message    : {self.message}")
        return "\n".join(lines)


# ---------------------------------------------------------------------- #
#  Shared base                                                           #
# ---------------------------------------------------------------------- #
class _MapBase:
    """
    Everything compressor and turbine maps have in common: grid storage,
    interpolation, design-point calibration, plotting helpers and Dymola export.

    Subclasses provide ``_validate_shapes``, ``_build_interpolators`` (which
    must define ``pr_map`` and then call ``_build_rgi``), ``scale_map``,
    ``query``, ``plot_map``, ``get_scaled_maps``, ``_xy`` and ``_export_arrays``.
    """
    _marker_size = 9

    def _init_common(self, namemap, ncorr_map, beta_map, wcorr_map, eff_map,
                     map_extrap, interp_method):
        self.namemap = namemap
        self.ncorr_map = np.asarray(ncorr_map, dtype=float)
        self.beta_map = np.asarray(beta_map, dtype=float)
        self.wcorr_map = np.asarray(wcorr_map, dtype=float)
        self.eff_map = np.asarray(eff_map, dtype=float)
        self._extrap = bool(map_extrap)
        if interp_method not in _METHODS:
            raise ValueError(f"interp_method must be one of {_METHODS}, got {interp_method!r}")
        self.interp_method = interp_method

        # Scaling factors - identity until calibrate_design()
        self._sWc: float = 1.0
        self._sPR: float = 1.0
        self._sE: float = 1.0
        self._calibrated: bool = False
        self.design_point: Optional[dict] = None

        self.log: list = []
        self.log_queries: bool = False

    @property
    def ncorr_bounds(self):
        return float(self.ncorr_map.min()), float(self.ncorr_map.max())

    @property
    def beta_bounds(self):
        return float(self.beta_map.min()), float(self.beta_map.max())

    # -- interpolation ---------------------------------------------------- #
    def _build_rgi(self):
        rgi_kw = dict(method=self.interp_method, bounds_error=not self._extrap,
                      fill_value=None if self._extrap else np.nan)
        grid = (self.ncorr_map, self.beta_map)

        # One interpolator for all three fields: values has shape (nN, nBeta, 3)
        stacked = np.stack([self.wcorr_map, self.pr_map, self.eff_map], axis=-1)
        self._rgi_all = RegularGridInterpolator(grid, stacked, **rgi_kw)

        self._rgi_W = RegularGridInterpolator(grid, self.wcorr_map, **rgi_kw)
        self._rgi_E = RegularGridInterpolator(grid, self.eff_map, **rgi_kw)
        self._rgi_PR = RegularGridInterpolator(grid, self.pr_map, **rgi_kw)

    def _raw(self, ncorr: float, beta: float, which: str) -> float:
        """Interpolate in raw (unscaled) map space. Single access point."""
        pt = (ncorr, beta)
        if which == 'W':
            return float(self._rgi_W(pt))
        if which == 'E':
            return float(self._rgi_E(pt))
        if which == 'P':
            return float(self._rgi_PR(pt))
        raise ValueError("which must be W, E, or P")

    def _raw_all(self, ncorr: float, beta: float):
        """Raw (unscaled) Wcorr, PR, eta in a single interpolation call."""
        out = self._rgi_all(np.array([[ncorr, beta]]))[0]     # shape (3,)
        return out[0], out[1], out[2]

    def _dense(self, rgi, arr, n_hr, b_hr, NN, BB):
        """Evaluate on a dense grid. For 'pchip' the 1-D PCHIP passes are vectorised (identical
        result to the RegularGridInterpolator, which would take ~40 s for the plot grid)."""
        if self.interp_method != 'pchip':
            return rgi((NN, BB))
        out = PchipInterpolator(self.beta_map, arr, axis=1)(b_hr)
        out = PchipInterpolator(self.ncorr_map, out, axis=0)(n_hr)
        return out

    def _scale(self, Wc_raw, PR_raw, eff_raw):
        """Apply scaling factors (identity if not calibrated)."""
        return (Wc_raw * self._sWc,
                1.0 + (PR_raw - 1.0) * self._sPR,
                eff_raw * self._sE)

    # -- calibration ------------------------------------------------------ #
    def calibrate_design(self, ncorr_des, beta_des, PR_des, eff_des, Wcorr_des,
                         RCeta_design: float = 1.0, RCm_design: float = 1.0):
        """
        Scale the map so that (ncorr_des, beta_des) matches your physical
        design point (Wcorr_des, PR_des, eff_des).
        After this call all public methods automatically work in scaled space.
        """
        Wc_raw = self._raw(ncorr_des, beta_des, 'W')
        PR_raw = self._raw(ncorr_des, beta_des, 'P')
        eta_raw = self._raw(ncorr_des, beta_des, 'E')

        if np.any(np.isnan([Wc_raw, PR_raw, eta_raw])):
            raise ValueError("Design point outside map range.")
        if abs(PR_raw - 1.0) < 1e-12:
            raise ValueError("Cannot scale PR: PR_raw too close to 1.0.")
        if eta_raw < 1e-12:
            raise ValueError("Cannot scale efficiency: eta_raw too close to 0.")

        self._sWc = float((Wcorr_des / RCm_design) / Wc_raw)
        self._sPR = float((PR_des - 1.0) / (PR_raw - 1.0))
        self._sE = float((eff_des / RCeta_design) / eta_raw)
        self._calibrated = True

        self.design_point = dict(Ncorr=ncorr_des, Beta=beta_des,
                                 PR=PR_des, Efficiency=eff_des, Wcorr=Wcorr_des)
        self.log.append({"action": "calibrate_design",
                         "sWc": self._sWc, "sPR": self._sPR, "sE": self._sE})
        print(f"[{type(self).__name__}] Calibrated — sWc={self._sWc:.4f}, "
              f"sPR={self._sPR:.4f}, sE={self._sE:.4f}")

    def _log_scale_map(self, scaled, sWc, sPR, sE):
        self.log.append({"action": "scale_map", "sWc": sWc, "sPR": sPR, "sE": sE,
                         "new_name": scaled.namemap})
        print(f"[{type(self).__name__}] scale_map → '{scaled.namemap}' "
              f"(sWc={sWc:.4f}, sPR={sPR:.4f}, sE={sE:.4f})")

    # -- queries ---------------------------------------------------------- #
    def _xy(self, state):
        """Plot coordinates of a state (x differs between compressor and turbine)."""
        raise NotImplementedError

    def query_many(self, points: Sequence[Tuple[float, float]],
                   labels: Optional[Sequence[str]] = None, plot: bool = False) -> list:
        """
        Evaluate the map at several (ncorr, beta) pairs and, optionally,
        plot all of them on a single map call.

        Parameters
        ----------
        points : sequence of (ncorr, beta) pairs
        labels : optional per-point labels; defaults to "β=…, PR=…"
        plot   : if True, draws every point on one shared map
        """
        # plot=False here, we plot once at the end
        states = [self.query(ncorr, beta, plot=False) for ncorr, beta in points]

        if plot:
            if labels is None:
                labels = [f"β={s.beta:.3f}, PR={s.PR:.3f}" for s in states]
            cmap = plt.get_cmap("tab10")
            self.plot_map(
                highlight_point=[self._xy(s) for s in states],
                highlight_label=labels,
                highlight_color=[cmap(i % cmap.N) for i in range(len(states))],
                scaled=self._calibrated,
            )
        return states

    def get_log(self):
        return self.log

    # -- plotting helper -------------------------------------------------- #
    def _draw_highlights(self, ax, highlight_point, highlight_label, highlight_color):
        is_single = isinstance(highlight_point[0], (int, float))
        points = [highlight_point] if is_single else list(highlight_point)
        n = len(points)

        labels = highlight_label if isinstance(highlight_label, (list, tuple)) else [highlight_label] * n
        colors = highlight_color if isinstance(highlight_color, (list, tuple)) else [highlight_color] * n

        if len(labels) != n or len(colors) != n:
            raise ValueError(
                f"highlight_point has {n} points; highlight_label has {len(labels)}, "
                f"highlight_color has {len(colors)}. Pass one value to apply to all "
                f"points, or a list matching highlight_point's length."
            )

        seen_labels = set()
        for (x, y), label, color in zip(points, labels, colors):
            # avoid duplicate legend entries when several points share a label
            legend_label = label if label not in seen_labels else None
            seen_labels.add(label)
            ax.plot(x, y, 'o', color=color, ms=self._marker_size, zorder=5, label=legend_label)

    # -- Dymola export ---------------------------------------------------- #
    def _export_arrays(self):
        """Return ``(Wc_map, PR_map, Eff_map, [(table_name, x, y), ...])`` for export.

        Arrays are scaled when the map was calibrated with ``calibrate_design``;
        a map made with ``scale_map`` already holds scaled arrays.
        """
        raise NotImplementedError

    def export_dymola(self, out_dir: str, suffix: str = "SOFC_LPTcycle"):
        """
        Write the map as Dymola CombiTable files into ``out_dir``.

        Files: ``massflow_map``, ``efficiency_map`` and ``pressureratio_map``
        (2D, over ncorr x beta) plus the map-specific line tables, each named
        ``<table>_scaled_<suffix>.csv``.
        """
        os.makedirs(out_dir, exist_ok=True)
        Wc, PR, Eff, lines = self._export_arrays()

        for table, values in (("massflow_map", Wc), ("efficiency_map", Eff), ("pressureratio_map", PR)):
            export_2d_map(os.path.join(out_dir, f"{table}_scaled_{suffix}.csv"), table,
                          values, self.ncorr_map, self.beta_map)
        for table, x, y in lines:
            export_1d_table(os.path.join(out_dir, f"{table}_scaled_{suffix}.csv"), table, x, y)


# ---------------------------------------------------------------------- #
#  Compressor                                                            #
# ---------------------------------------------------------------------- #
class CompressorMap(_MapBase):
    """
    Compressor performance map with optional design-point scaling.

    Raw arrays:
      - (ncorr, beta) -> Wcorr, PR, Eff
      - Surge line defined in raw space as (swc_map, spr_map).

    Basic usage (no calibration needed):
        cmap = CompressorMap(...)
        state = cmap.query(ncorr=0.95, beta=0.5)

    With design-point scaling:
        cmap.calibrate_design(...)
        state = cmap.query(ncorr=0.95, beta=0.5)   # same call, scaled automatically

    Queries outside the map are clipped to it and flagged (state.feasible = False).
    `interp_method` ('pchip' default, 'cubic', 'linear') sets the smoothness of the interpolation.
    `map_extrap` only affects the raw interpolators used by _raw (e.g. in calibrate_design).
    Set `log_queries = True` to record every query() in `log`; it is off by default because
    the log grows without bound inside solver loops.
    """
    _marker_size = 9

    def __init__(self, namemap, ncorr_map, beta_map, wcorr_map, eff_map, pr_map, swc_map, spr_map,
                 *, map_extrap: bool = False, interp_method: str = 'pchip'):
        self._init_common(namemap, ncorr_map, beta_map, wcorr_map, eff_map, map_extrap, interp_method)
        self.pr_map = np.asarray(pr_map, dtype=float)
        self.swc_map = np.asarray(swc_map, dtype=float)  # surge Wc (RAW)
        self.spr_map = np.asarray(spr_map, dtype=float)  # surge PR (RAW)

        self._validate_shapes()
        self._build_interpolators()

    # -- setup ------------------------------------------------------------ #
    def _validate_shapes(self):
        if (self.wcorr_map.shape != self.pr_map.shape or
                self.wcorr_map.shape != self.eff_map.shape):
            raise ValueError("wcorr_map, pr_map, eff_map must have identical shapes (nN x nBeta).")
        if self.wcorr_map.shape != (self.ncorr_map.size, self.beta_map.size):
            raise ValueError("Map shape must be (len(ncorr_map), len(beta_map)).")
        if self.swc_map.size != self.spr_map.size:
            raise ValueError("swc_map and spr_map must have the same length.")

    def _build_interpolators(self):
        # Sort surge curve by Wcorr ascending for robust 1D interpolation
        idx = np.argsort(self.swc_map)
        self.swc_map = self.swc_map[idx]
        self.spr_map = self.spr_map[idx]
        self._surge_interp = interp1d(
            self.swc_map, self.spr_map, kind='linear', bounds_error=True)
        self._build_rgi()

    def scale_map(self, sWc: float = 1.0, sPR: float = 1.0, sE: float = 1.0,
                  new_name: str = None) -> "CompressorMap":
        """
        Create a new CompressorMap instance with the map arrays pre-scaled
        by the given factors. Independent of calibrate_design().

        Parameters
        ----------
        sWc      : Wcorr scaling factor
        sPR      : PR scaling factor  (applied as 1 + (PR-1)*sPR)
        sE       : Efficiency scaling factor
        new_name : name for the new instance (defaults to '{name}_scaled')
        """
        scaled = CompressorMap(
            namemap=new_name or f"{self.namemap}_scaled",
            ncorr_map=np.copy(self.ncorr_map),
            beta_map=np.copy(self.beta_map),
            wcorr_map=self.wcorr_map * sWc,
            eff_map=self.eff_map * sE,
            pr_map=1.0 + (self.pr_map - 1.0) * sPR,
            swc_map=self.swc_map * sWc,
            spr_map=1.0 + (self.spr_map - 1.0) * sPR,
            map_extrap=self._extrap,
            interp_method=self.interp_method,
        )
        self._log_scale_map(scaled, sWc, sPR, sE)
        return scaled

    # -- surge ------------------------------------------------------------ #
    def _surge_pr_at(self, wcorr_scaled: float) -> float:
        """Surge PR in scaled space at a given scaled Wcorr."""
        wcorr_raw = wcorr_scaled / self._sWc
        pr_surge_raw = float(self._surge_interp(wcorr_raw))
        return 1.0 + (pr_surge_raw - 1.0) * self._sPR

    def _is_in_surge(self, wcorr: float, PR: float) -> bool:
        """Check surge in the current space (scaled if calibrated, raw otherwise)."""
        try:
            return PR > self._surge_pr_at(wcorr)
        except ValueError:
            return False  # outside surge curve range — assume not in surge

    # -- public API ------------------------------------------------------- #
    def query(self, ncorr: float, beta: float, plot: bool = False) -> CompressorState:
        """
        Evaluate map at (ncorr, beta). Works with or without calibration.
        Single entry point for a known (ncorr, beta) pair.
        """
        ncorr_c = np.clip(ncorr, self.ncorr_map.min(), self.ncorr_map.max())
        beta_c = np.clip(beta, self.beta_map.min(), self.beta_map.max())
        out_of_domain = (ncorr_c != ncorr) or (beta_c != beta)

        Wc, PR, eta = self._scale(*self._raw_all(ncorr_c, beta_c))

        # Surge check must also be bounds-safe
        in_surge = self._is_in_surge(Wc, PR)
        feasible = (not in_surge) and (not out_of_domain)
        msg = []
        if out_of_domain:
            msg.append("Clipped to map domain.")
        if in_surge:
            msg.append("In surge region.")

        state = CompressorState(ncorr=ncorr, beta=beta, wcorr=Wc, PR=PR, eta=eta,
                                feasible=feasible, in_surge=in_surge,
                                message=" ".join(msg))

        if self.log_queries:
            self.log.append({"action": "query", "ncorr": ncorr, "beta": beta,
                             "Wcorr": Wc, "PR": PR, "eta": eta, "in_surge": in_surge})
        if plot:
            self.plot_map(highlight_point=(Wc, PR),
                          highlight_label=f"β={beta:.3f}, PR={PR:.3f}",
                          highlight_color="red" if in_surge else "limegreen",
                          scaled=self._calibrated)
        return state

    def _xy(self, state):
        return state.wcorr, state.PR

    def get_scaled_maps(self, sWc: float = 1.0, sPR: float = 1.0, sE: float = 1.0):
        """
        Return scaled map arrays using explicit factors without modifying
        internal state. Useful for exporting or inspecting scaled surfaces.
        """
        Wc_map = self.wcorr_map * sWc
        PR_map = 1.0 + (self.pr_map - 1.0) * sPR
        Eff_map = self.eff_map * sE
        swc = self.swc_map * sWc
        spr = 1.0 + (self.spr_map - 1.0) * sPR
        return Wc_map, PR_map, Eff_map, swc, spr

    def _export_arrays(self):
        if self._calibrated:
            Wc, PR, Eff, swc, spr = self.get_scaled_maps(self._sWc, self._sPR, self._sE)
        else:
            Wc, PR, Eff, swc, spr = self.wcorr_map, self.pr_map, self.eff_map, self.swc_map, self.spr_map
        return Wc, PR, Eff, [("surgeline", swc, spr)]

    # -- plotting --------------------------------------------------------- #
    def plot_map(self, highlight_point=None, highlight_label='Evaluated Point',
                 highlight_color='limegreen', scaled: bool = None):
        """
        Plot compressor map. scaled=None → auto (True if calibrated, raw otherwise).
        x-axis = Wcorr, y-axis = PR (compressor convention, not Wcorr x Ncorr).
        """
        if scaled is None:
            scaled = self._calibrated

        sWc, sPR, sE = (self._sWc, self._sPR, self._sE) if scaled else (1.0, 1.0, 1.0)
        Wc_map, PR_map, Eff_map, swc, spr = self.get_scaled_maps(sWc, sPR, sE)

        fig, ax = plt.subplots(figsize=(14, 9))

        # Speed lines
        for i, nc in enumerate(self.ncorr_map):
            x, y = Wc_map[i], PR_map[i]
            m = np.isfinite(x) & np.isfinite(y)
            if np.any(m):
                ax.plot(x[m], y[m], color='black', lw=1.6, alpha=0.7, zorder=3)
                if i % max(1, len(self.ncorr_map) // 8) == 0:
                    k = np.flatnonzero(m)[len(np.flatnonzero(m)) // 2]
                    ax.text(x[k], y[k], f'N={nc:.2f}', fontsize=12,
                            color='black', fontweight='bold',
                            ha='center', va='bottom')

        # Beta lines
        for j in range(len(self.beta_map)):
            x, y = Wc_map[:, j], PR_map[:, j]
            m = np.isfinite(x) & np.isfinite(y)
            if np.any(m):
                ax.plot(x[m], y[m], '--', color='0.6', lw=0.8, alpha=0.3, zorder=2)

        # Surge line
        ms = np.isfinite(swc) & np.isfinite(spr)
        if np.any(ms):
            ax.plot(swc[ms], spr[ms], '--', color='tab:red',
                    lw=2.0, zorder=4, label='Surge line')

        # High-res efficiency contours
        n_hr = np.linspace(float(self.ncorr_map[0]), float(self.ncorr_map[-1]), 200)
        b_hr = np.linspace(float(self.beta_map[0]), float(self.beta_map[-1]), 320)
        NN, BB = np.meshgrid(n_hr, b_hr, indexing='ij')
        Wc_hr = self._dense(self._rgi_W, self.wcorr_map, n_hr, b_hr, NN, BB) * sWc
        PR_hr = 1.0 + (self._dense(self._rgi_PR, self.pr_map, n_hr, b_hr, NN, BB) - 1.0) * sPR
        Eff_hr = self._dense(self._rgi_E, self.eff_map, n_hr, b_hr, NN, BB) * sE

        mask = np.isfinite(Eff_hr) & np.isfinite(Wc_hr) & np.isfinite(PR_hr)
        vmin = float(np.nanmin(Eff_hr[mask]))
        vmax = float(np.nanmax(Eff_hr[mask]))
        if vmax - vmin < 5e-3:
            vmin, vmax = max(0.0, vmin - 0.005), min(0.999, vmax + 0.005)

        rd = lambda x, s: s * np.floor(x / s)
        ru = lambda x, s: s * np.ceil(x / s)
        minor = np.arange(rd(vmin, 0.01), ru(vmax, 0.01) + 1e-9, 0.01)
        major = np.arange(rd(vmin, 0.02), ru(vmax, 0.02) + 1e-9, 0.02)

        cf = ax.contourf(Wc_hr, PR_hr, Eff_hr,
                         levels=minor, cmap='viridis', alpha=0.5, zorder=0)
        ax.contour(Wc_hr, PR_hr, Eff_hr,
                   levels=minor, colors='lightcoral',
                   linewidths=1.1, linestyles='--', alpha=0.8, zorder=1)
        cb = fig.colorbar(cf, ax=ax,
                          ticks=major[::max(1, len(major) // 8)],
                          shrink=0.94, pad=0.015)
        cb.set_label('Isentropic efficiency η', fontsize=11)

        if highlight_point is not None:
            self._draw_highlights(ax, highlight_point, highlight_label, highlight_color)

        suffix = ' (scaled)' if scaled else ' (raw)'
        ax.set_title(f"{self.namemap}{suffix}", fontsize=16, fontweight='bold')
        ax.set_xlabel('Corrected Mass Flow (Wcorr)', fontsize=13)
        ax.set_ylabel('Pressure Ratio', fontsize=13)
        ax.minorticks_on()
        ax.grid(which='major', color='lightgray', linestyle='--', lw=0.8)
        ax.grid(which='minor', color='gainsboro', linestyle=':', lw=0.5)
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        ax.set_xlim(x0 - 0.05 * (x1 - x0), x1 + 0.05 * (x1 - x0))
        ax.set_ylim(y0 - 0.05 * (y1 - y0), y1 + 0.05 * (y1 - y0))
        if highlight_point is not None:
            ax.legend(loc='best', frameon=True, fontsize=10)
        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------------------- #
#  Turbine                                                               #
# ---------------------------------------------------------------------- #
class TurbineMap(_MapBase):
    """
    Turbine performance map with optional design-point scaling (same conventions as
    CompressorMap).

    Raw arrays:
      - (ncorr, beta) -> Wcorr, Eff
      - minPR(ncorr), maxPR(ncorr) define the PR band; beta in [0, 1] runs linearly from
        minPR to maxPR, which gives the raw PR map.

    After calibrate_design(...), query() works in scaled space automatically.
    Queries outside the map are clipped to it and flagged (state.feasible = False).
    `interp_method` ('pchip' default, 'cubic', 'linear') sets the smoothness of the interpolation.
    `map_extrap` only affects the raw interpolators used by _raw (e.g. in calibrate_design).
    Set `log_queries = True` to record every query() in `log`; it is off by default because
    the log grows without bound inside solver loops.
    """
    _marker_size = 10

    def __init__(self, namemap, ncorr_map, beta_map, wcorr_map, eff_map,
                 minpr_curve, maxpr_curve, *, map_extrap: bool = False, interp_method: str = 'pchip'):
        self._init_common(namemap, ncorr_map, beta_map, wcorr_map, eff_map, map_extrap, interp_method)
        self.minpr_curve = np.asarray(minpr_curve, dtype=float)
        self.maxpr_curve = np.asarray(maxpr_curve, dtype=float)

        self._validate_shapes()
        self._build_interpolators()

    # -- setup ------------------------------------------------------------ #
    def _validate_shapes(self):
        if self.wcorr_map.shape != self.eff_map.shape:
            raise ValueError("wcorr_map and eff_map must have identical shapes (nN x nBeta).")
        if self.wcorr_map.shape != (self.ncorr_map.size, self.beta_map.size):
            raise ValueError("Map shapes must be (len(ncorr_map) x len(beta_map)).")
        if self.minpr_curve.size != self.ncorr_map.size or self.maxpr_curve.size != self.ncorr_map.size:
            raise ValueError("minpr_curve/maxpr_curve must have length len(ncorr_map).")

    def _build_interpolators(self):
        # raw PR map (beta parameterizes linearly between min & max)
        self.pr_map = (self.minpr_curve[:, np.newaxis]
                       + (self.maxpr_curve[:, np.newaxis] - self.minpr_curve[:, np.newaxis])
                       * self.beta_map[np.newaxis, :])
        self._build_rgi()

    def scale_map(self, sWc: float = 1.0, sPR: float = 1.0, sE: float = 1.0,
                  new_name: str = None) -> "TurbineMap":
        """
        Create a new TurbineMap instance with the map arrays pre-scaled
        by the given factors. The new instance is independent — modifying
        it does not affect the original.

        This is different from calibrate_design(), which stores scalars
        internally and applies them at query time. Here the arrays themselves
        are scaled, so the new instance works in scaled space natively
        without needing calibrate_design().

        Parameters
        ----------
        sWc      : Wcorr scaling factor
        sPR      : PR scaling factor  (applied as 1 + (PR-1)*sPR)
        sE       : Efficiency scaling factor
        new_name : name for the new instance (defaults to '{name}_scaled')
        """
        scaled = TurbineMap(
            namemap=new_name or f"{self.namemap}_scaled",
            ncorr_map=np.copy(self.ncorr_map),
            beta_map=np.copy(self.beta_map),
            wcorr_map=self.wcorr_map * sWc,
            eff_map=self.eff_map * sE,
            minpr_curve=1.0 + (self.minpr_curve - 1.0) * sPR,
            maxpr_curve=1.0 + (self.maxpr_curve - 1.0) * sPR,
            map_extrap=self._extrap,
            interp_method=self.interp_method,
        )
        self._log_scale_map(scaled, sWc, sPR, sE)
        return scaled

    # -- public API ------------------------------------------------------- #
    def query(self, ncorr: float, beta: float, plot: bool = False) -> TurbineState:
        """
        Evaluate map at (ncorr, beta). Works with or without calibration.
        This is the single entry point for a known (ncorr, beta) pair.
        """
        ncorr_c = np.clip(ncorr, self.ncorr_map.min(), self.ncorr_map.max())
        beta_c = np.clip(beta, self.beta_map.min(), self.beta_map.max())
        out_of_domain = (ncorr_c != ncorr) or (beta_c != beta)

        Wc, PR, eta = self._scale(*self._raw_all(ncorr_c, beta_c))

        state = TurbineState(ncorr=ncorr, beta=beta, wcorr=Wc, PR=PR, eta=eta,
                             feasible=not out_of_domain,
                             message="Clipped to map domain." if out_of_domain else "")

        if self.log_queries:
            self.log.append({"action": "query", "ncorr": ncorr, "beta": beta,
                             "Wcorr": Wc, "PR": PR, "eta": eta, "out_of_domain": out_of_domain})

        if plot:
            self.plot_map(highlight_point=(Wc * ncorr, PR),
                          highlight_label=f"β={beta:.3f}, PR={PR:.3f}",
                          scaled=self._calibrated)
        return state

    def _xy(self, state):
        return state.wcorr * state.ncorr, state.PR

    def get_scaled_maps(self, sWc: float = 1.0, sPR: float = 1.0, sE: float = 1.0):
        """
        Return scaled map arrays using explicit factors without modifying
        internal state. Useful for exporting or inspecting scaled surfaces.
        """
        Wc_map = self.wcorr_map * sWc
        PR_map = 1.0 + (self.pr_map - 1.0) * sPR
        Eff_map = self.eff_map * sE
        minPR = 1.0 + (self.minpr_curve - 1.0) * sPR
        maxPR = 1.0 + (self.maxpr_curve - 1.0) * sPR
        return Wc_map, PR_map, Eff_map, minPR, maxPR

    def _export_arrays(self):
        if self._calibrated:
            Wc, PR, Eff, mn, mx = self.get_scaled_maps(self._sWc, self._sPR, self._sE)
        else:
            Wc, PR, Eff, mn, mx = self.wcorr_map, self.pr_map, self.eff_map, self.minpr_curve, self.maxpr_curve
        return Wc, PR, Eff, [("minpr_line", self.ncorr_map, mn), ("maxpr_line", self.ncorr_map, mx)]

    # -- plotting --------------------------------------------------------- #
    def plot_map(self, highlight_point=None, highlight_label='Evaluated Point',
                 highlight_color='red', scaled: bool = None):
        """
        Plot turbine map. scaled=None → auto (True if calibrated, raw otherwise).
        """
        if scaled is None:
            scaled = self._calibrated

        sWc, sPR, sE = (self._sWc, self._sPR, self._sE) if scaled else (1.0, 1.0, 1.0)
        Wc_map, PR_map, Eff_map, minPR, maxPR = self.get_scaled_maps(sWc, sPR, sE)

        WN_map = Wc_map * self.ncorr_map[:, np.newaxis]

        fig, ax = plt.subplots(figsize=(12, 8))

        for i, ncorr in enumerate(self.ncorr_map):
            ax.plot(WN_map[i], PR_map[i], lw=2, color='black', alpha=0.6)
            if i >= 5:
                idx = int(0.75 * len(WN_map[i]))
                ax.text(WN_map[i][idx], PR_map[i][idx],
                        f'N={ncorr:.2f}', fontsize=10, fontweight='bold',
                        color='black', ha='center', va='bottom')

        ax.plot(WN_map[:, 0], minPR, 'b--', lw=2, label='Min PR')
        ax.plot(WN_map[:, -1], maxPR, 'r--', lw=2, label='Max PR')

        for j in range(len(self.beta_map)):
            ax.plot(WN_map[:, j], PR_map[:, j],
                    '--', color='gray', lw=1, alpha=0.2)

        eff_levels = [0.6, 0.8, 0.81, 0.87, 0.88, 0.89, 0.9, 0.91]
        Eff_clipped = np.clip(Eff_map, 0.01, 1.0)
        if Eff_clipped.shape != WN_map.shape:
            Eff_clipped = Eff_clipped.T
        CS = ax.contour(WN_map, PR_map, Eff_clipped,
                        levels=eff_levels, colors='darkred', alpha=0.8, linewidths=1)
        ax.clabel(CS, inline=True, fontsize=10, fmt='η=%.2f')
        CS.set_linestyle('--')

        if highlight_point is not None:
            self._draw_highlights(ax, highlight_point, highlight_label, highlight_color)

        suffix = " (scaled)" if scaled else " (raw)"
        ax.set_title(f"{self.namemap} — PR vs Wcorr x Ncorr{suffix}",
                     fontsize=16, fontweight='bold')
        ax.set_xlabel("Wcorr x Ncorr", fontsize=14)
        ax.set_ylabel("Pressure Ratio", fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.legend(fontsize=10)
        ax.minorticks_on()
        ax.grid(which='major', color='lightgray', linestyle='--', lw=0.8)
        ax.grid(which='minor', color='gainsboro', linestyle=':', lw=0.6)

        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        ax.set_xlim(x_min - 0.05 * (x_max - x_min), x_max + 0.05 * (x_max - x_min))
        ax.set_ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))

        plt.tight_layout()
        plt.show()

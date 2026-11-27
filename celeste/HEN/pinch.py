"""
Pinch analysis: energy targets, composite curves and grand composite curve of a set of hot and cold streams.

Every stream segment is a heat load ``Q`` between two temperatures ``T_hi >= T_lo`` (equal for a latent segment).
The whole analysis rests on one function, the heat of the segments that lies above (or below) a temperature:

* problem table / heat cascade: net heat surplus above each shifted temperature level
  (hot temperatures lowered and cold temperatures raised by dt_min/2);
* minimum hot utility: the largest deficit of the cascade; cold utility: what is left at the bottom;
* composite curves: heat of each side below each temperature; the cold curve starts at the cold utility;
* grand composite curve: the cascade with the hot utility added, a straight line between the levels.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class PinchAnalyser:
    """
    Pinch analysis (problem table algorithm) of a set of hot and cold streams.

    Parameters
    ----------
    segments : DataFrame with one row per stream segment and the columns
               ``segment_type`` ('sensible' | 'latent'), ``side`` ('hot' | 'cold'),
               ``T_in``, ``T_out`` [degC] and ``CP`` [kW/K] for sensible segments,
               ``T_phase`` [degC] and ``Q`` [kW] for latent (isothermal) segments.
               The segment name is the index or a column ``segment_id``.
               ``SysBase.pinch_stream_table()`` builds this table from a solved system.
    dt_min   : minimum temperature difference between hot and cold streams [K]

    Usage
    -----
    >>> pa = PinchAnalyser(system.pinch_stream_table(), dt_min=10).solve()
    >>> pa.summary()                 # minimum utilities, heat recovery, pinch temperatures
    >>> pa.plot_summary()            # composite curves and grand composite curve
    >>> PinchAnalyser.sweep_dt_min(table)   # targets as a function of dt_min
    """

    def __init__(self, segments: pd.DataFrame, dt_min: float, tol: float = 1e-9):
        self.dt_min, self.tol = float(dt_min), float(tol)
        if not np.isfinite(self.dt_min) or self.dt_min <= 0:
            raise ValueError("dt_min must be a positive finite number.")
        if not np.isfinite(self.tol) or self.tol < 0:
            raise ValueError("tol must be a non-negative finite number.")
        self._segments = self._validate(segments)
        self._solved = False

    # ---- input --------------------------------------------------------------------------------------------
    @staticmethod
    def _validate(segments: pd.DataFrame) -> pd.DataFrame:
        """Check the table; add ``T_hi``, ``T_lo`` (degC), the heat load ``Q`` of every segment and ``sign`` (+1 hot)."""
        s = segments.copy()
        s.columns = s.columns.str.strip()
        if "segment_id" not in s.columns:              # names as the index, as from SysBase.pinch_stream_table()
            s = s.rename_axis("segment_id").reset_index()
        for col in ("T_in", "T_out", "CP", "T_phase", "Q"):
            s[col] = pd.to_numeric(s[col], errors="coerce") if col in s.columns else np.nan
        s["segment_type"] = s["segment_type"].str.strip().str.lower()
        s["side"] = s["side"].str.strip().str.lower()

        def check(mask, message):
            bad = s.loc[mask, "segment_id"].tolist()
            if bad:
                raise ValueError(f"{message}: {bad}")

        if set(s.segment_type) - {"sensible", "latent"}:
            raise ValueError(f"Unknown segment_type(s): {set(s.segment_type) - {'sensible', 'latent'}}. "
                             "Must be 'sensible' or 'latent'.")
        if set(s.side) - {"hot", "cold"}:
            raise ValueError(f"Unknown side(s): {set(s.side) - {'hot', 'cold'}}. Must be 'hot' or 'cold'.")
        sens, hot = (s.segment_type == "sensible"), (s.side == "hot")
        check(sens & s[["T_in", "T_out", "CP"]].isna().any(axis=1), "Sensible segments missing T_in / T_out / CP")
        check(~sens & s[["T_phase", "Q"]].isna().any(axis=1), "Latent segments missing T_phase / Q")
        check(sens & hot & (s.T_in <= s.T_out), "Hot sensible streams must have T_in > T_out")
        check(sens & ~hot & (s.T_in >= s.T_out), "Cold sensible streams must have T_in < T_out")
        check(sens & (s.CP <= 0), "CP must be > 0")
        check(~sens & (s.Q <= 0), "Latent Q must be > 0")

        s["T_hi"] = np.where(sens, s[["T_in", "T_out"]].max(axis=1), s.T_phase)
        s["T_lo"] = np.where(sens, s[["T_in", "T_out"]].min(axis=1), s.T_phase)
        s["Q"] = np.where(sens, s.CP*(s.T_hi - s.T_lo), s.Q)
        s["sign"] = np.where(hot, 1.0, -1.0)
        return s

    # ---- the one building block ---------------------------------------------------------------------------
    @staticmethod
    def _heat(seg: pd.DataFrame, T: float, above: bool, inclusive: bool, shift: float = 0.0) -> np.ndarray:
        """
        Heat of every segment that lies above (``above``) or below temperature ``T``. ``shift`` moves the hot
        segments down and the cold ones up by that amount (dt_min/2 for the problem table). ``inclusive``
        counts a latent segment at exactly ``T``.
        """
        hi, lo = seg.T_hi.values - seg.sign.values*shift, seg.T_lo.values - seg.sign.values*shift
        span = hi - lo
        frac = np.clip(((hi - T) if above else (T - lo))/np.where(span > 0, span, 1.0), 0.0, 1.0)
        at = (hi >= T) if above else (lo <= T)
        strict = (hi > T) if above else (lo < T)
        latent = (at if inclusive else strict).astype(float)
        return seg.Q.values*np.where(span > 0, frac, latent)

    # ---- solve --------------------------------------------------------------------------------------------
    def solve(self):
        seg, dth = self._segments, 0.5*self.dt_min
        shifted = np.r_[seg.T_hi - seg.sign*dth, seg.T_lo - seg.sign*dth]
        levels = np.unique(shifted)[::-1]                                   # descending
        if levels.size < 2:
            raise ValueError("At least two distinct shifted temperature levels are required to form intervals.")

        # net surplus above each level, just above it (latent segments at the level excluded) and at it
        surplus = lambda T, inc: float(seg.sign.values @ self._heat(seg, T, True, inc, dth))
        above = np.array([surplus(T, False) for T in levels])
        at = np.array([surplus(T, True) for T in levels])
        self.hot_utility = max(0.0, -min(above.min(), at.min()))
        self.cold_utility = self.hot_utility + at[-1]
        self.heat_recovery = float(seg.Q[seg.sign > 0].sum()) - self.cold_utility

        # grand composite curve: the cascade with the hot utility, two points where a latent segment sits
        T_gcc = np.repeat(levels, 2)
        H_gcc = np.ravel(np.c_[above, at]) + self.hot_utility
        keep = np.r_[True, (np.diff(H_gcc) != 0) | (np.diff(T_gcc) != 0)]
        self.grand_composite_curve = pd.DataFrame({"H": H_gcc[keep], "T_shift": T_gcc[keep],
                                                   "T_hot": T_gcc[keep] + dth, "T_cold": T_gcc[keep] - dth})
        pinch = levels[np.minimum(above, at) + self.hot_utility <= self.tol + 1e-12*max(1.0, self.hot_utility)]
        if pinch.size == 0:                                                 # threshold problem: closest approach
            pinch = levels[[np.argmin(np.minimum(above, at))]]
        self.pinch_temps = {"shifted": pinch.tolist(), "hot": (pinch + dth).tolist(), "cold": (pinch - dth).tolist()}

        # problem table: one row per interval between two levels
        self.problem_table = pd.DataFrame({"T_upper": levels[:-1], "T_lower": levels[1:], "dT": -np.diff(levels),
                                           "deltaH": np.diff(at), "cascade": at[1:] + self.hot_utility})
        self.heat_cascade = pd.DataFrame({"T_shift": levels, "cascade": at + self.hot_utility,
                                          "is_pinch": np.isin(levels, pinch)})

        # composite curves, in true and in shifted temperatures; the cold curves start at the cold utility
        self.composite_curves = {}
        for side, sign in (("hot", 1.0), ("cold", -1.0)):
            part = seg[seg.sign == sign]
            for name, shift in ((side, 0.0), (side + "_shifted", dth)):
                T = np.unique(np.r_[part.T_hi, part.T_lo] - sign*shift)
                H = [[self._heat(part, t, False, inc, shift).sum() for inc in (False, True)] for t in T]
                T, H = np.repeat(T, 2), np.ravel(H) + (self.cold_utility if side == "cold" else 0.0)
                keep = np.r_[True, (np.diff(H) != 0) | (np.diff(T) != 0)]
                self.composite_curves[name] = pd.DataFrame({"H": H[keep], "T": T[keep]})
        self._solved = True
        return self

    # ---- results ------------------------------------------------------------------------------------------
    @property
    def segments(self) -> pd.DataFrame:
        return self._segments.copy()

    def _require_solved(self):
        if not self._solved:
            raise RuntimeError("PinchAnalyser has not been solved yet.")

    def summary(self) -> pd.Series:
        self._require_solved()
        return pd.Series({
            "dt_min": self.dt_min,
            "hot_utility": self.hot_utility,
            "cold_utility": self.cold_utility,
            "heat_recovery": self.heat_recovery,
            "pinch_shifted": self.pinch_temps["shifted"],
            "pinch_hot": self.pinch_temps["hot"],
            "pinch_cold": self.pinch_temps["cold"],
            "n_intervals": len(self.problem_table),
            "n_segments": len(self._segments),
        })

    @classmethod
    def sweep_dt_min(cls, segments: pd.DataFrame, dt_min_range=None) -> pd.DataFrame:
        """Utility targets, heat recovery and pinch temperature against dt_min (NaN where the analysis fails)."""
        rows = []
        for dt in (np.arange(2, 52, 2) if dt_min_range is None else dt_min_range):
            try:
                s = cls(segments, dt_min=float(dt)).solve().summary()
                rows.append({"dt_min": dt, "hot_utility": s.hot_utility, "cold_utility": s.cold_utility,
                             "heat_recovery": s.heat_recovery, "pinch_hot": s.pinch_hot[0] if s.pinch_hot else np.nan})
            except Exception:
                rows.append({"dt_min": dt, "hot_utility": np.nan, "cold_utility": np.nan, "heat_recovery": np.nan,
                             "pinch_hot": np.nan})
        return pd.DataFrame(rows)

    # ---- plots --------------------------------------------------------------------------------------------
    @staticmethod
    def _axes(ax, figsize):
        if ax is None:
            return plt.subplots(figsize=figsize)
        return ax.figure, ax

    def plot_composite_curves(self, ax=None, shifted=False, show_pinch=True, annotate_utilities=True):
        """Hot and cold composite curves; the overlap is the heat recovery, the ends the utilities."""
        self._require_solved()
        fig, ax = self._axes(ax, (7, 5))
        sfx = "_shifted" if shifted else ""
        hot, cold = self.composite_curves["hot" + sfx], self.composite_curves["cold" + sfx]

        H = np.linspace(cold.H.min(), hot.H.max(), 400)
        if H[-1] > H[0]:
            ax.fill_between(H, np.interp(H, hot.H, hot["T"]), np.interp(H, cold.H, cold["T"]), color="tab:green",
                            alpha=0.15, label=f"Heat recovery = {self.heat_recovery:.1f}")
        ax.plot(hot.H, hot["T"], color="tab:red", lw=2.2, label="Hot composite")
        ax.plot(cold.H, cold["T"], color="tab:blue", lw=2.2, label="Cold composite")
        if annotate_utilities:                      # shaded strip and double arrow for each utility
            for Q, H0, H1, T, color, name in ((self.cold_utility, 0.0, cold.H.min(), hot["T"].iloc[0], "tab:blue", "Q_C"),
                                             (self.hot_utility, hot.H.max(), cold.H.max(), cold["T"].iloc[-1], "tab:red", "Q_H")):
                if Q > self.tol:
                    ax.axvspan(H0, H1, color=color, alpha=0.08, label=f"${name}$ = {Q:.1f}")
                    ax.annotate("", xy=(H0, T), xytext=(H1, T), arrowprops=dict(arrowstyle="<->", color=color, lw=1.4))
                    ax.text(0.5*(H0 + H1), T, f"${name}$ = {Q:.1f}", color=color, fontsize=8, ha="center", va="bottom")
        if show_pinch:                              # hot and cold pinch temperatures (one line when shifted)
            lines = ([("shifted", "0.35", "Pinch")] if shifted else
                     [("hot", "tab:red", "Hot pinch"), ("cold", "tab:blue", "Cold pinch")])
            for key, color, label in lines:
                for i, T in enumerate(self.pinch_temps[key]):
                    ax.axhline(T, color=color, lw=1.0, ls="--", label=label if i == 0 else None)
        ax.set_xlim(left=0)
        ax.set(xlabel="Cumulative heat load", ylabel="Shifted temperature [°C]" if shifted else "Temperature [°C]",
               title="Composite curves" + (" (shifted)" if shifted else ""))
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        return fig, ax

    def plot_grand_composite_curve(self, ax=None, show_pinch=True):
        """Net heat flow against shifted temperature: straight lines between the levels, flat for latent heat."""
        self._require_solved()
        fig, ax = self._axes(ax, (6.5, 5))
        gcc = self.grand_composite_curve
        ax.plot(gcc.H, gcc.T_shift, color="tab:green", lw=2.2, label="Grand composite curve")
        if show_pinch:
            for i, T in enumerate(self.pinch_temps["shifted"]):
                ax.axhline(T, color="0.35", lw=1.0, ls="--", label="Pinch" if i == 0 else None)
        ax.axvline(0.0, color="0.4", lw=1.0)
        ax.set(xlabel="Net heat flow", ylabel="Shifted temperature [°C]", title="Grand composite curve")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        return fig, ax

    def plot_summary(self, shifted=False, figsize=(12, 5)):
        """Composite curves and grand composite curve side by side."""
        self._require_solved()
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        self.plot_composite_curves(ax=axes[0], shifted=shifted)
        self.plot_grand_composite_curve(ax=axes[1])
        fig.tight_layout()
        return fig, axes

    @staticmethod
    def plot_utility_sweep(sweep_df: pd.DataFrame, current_dt_min: float = None, ax=None):
        """Q_H and Q_C against dt_min (from ``sweep_dt_min``)."""
        fig, ax = PinchAnalyser._axes(ax, (7, 4.5))
        df = sweep_df.dropna(subset=["hot_utility", "cold_utility"])
        ax.plot(df.dt_min, df.hot_utility, color="tab:red", lw=2, marker="o", ms=3, label="$Q_H$ (hot utility)")
        ax.plot(df.dt_min, df.cold_utility, color="tab:blue", lw=2, marker="o", ms=3, label="$Q_C$ (cold utility)")
        if current_dt_min is not None:
            ax.axvline(current_dt_min, color="gray", ls="--", lw=1.2, label=f"Current ΔT_min = {current_dt_min}")
        ax.set(xlabel=r"$\Delta T_{min}$ [°C]", ylabel="Utility load [kW]", title=r"Utility targets vs $\Delta T_{min}$")
        ax.legend(framealpha=0.9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig, ax

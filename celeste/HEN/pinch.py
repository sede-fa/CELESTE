import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt


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

    REQUIRED_COLUMNS = {
        "segment_id",
        "segment_type",   # "sensible" | "latent"
        "side",           # "hot" | "cold"
        "T_in",
        "T_out",
        "CP",
        "T_phase",
        "Q",
    }

    def __init__(self, segments: pd.DataFrame, dt_min: float, tol: float = 1e-9):
        self.dt_min = float(dt_min)
        self.tol = float(tol)

        if not np.isfinite(self.dt_min) or self.dt_min <= 0:
            raise ValueError("dt_min must be a positive finite number.")
        if not np.isfinite(self.tol) or self.tol < 0:
            raise ValueError("tol must be a non-negative finite number.")

        self._shifted_segments = None
        self._intervals = None
        self._problem_table = None
        self._heat_cascade = None
        self._hot_utility = None
        self._cold_utility = None
        self._heat_recovery = None
        self._pinch_temps = None
        self._composite_curves = None
        self._grand_composite_curve = None
        self._is_sensible = None
        self._is_latent   = None
        self._is_hot      = None
        self._is_cold     = None

        self._terminal_latent = 0.0
        self._segments = self._validate_segments(segments)

    def _require_solved(self):
        if self._problem_table is None:
            raise RuntimeError("PinchAnalyser has not been solved yet.")

    @property
    def segments(self):
        return self._segments.copy()

    @property
    def shifted_segments(self):
        self._require_solved()
        return self._shifted_segments.copy()

    @property
    def intervals(self):
        self._require_solved()
        return self._intervals.copy()

    @property
    def problem_table(self):
        self._require_solved()
        return self._problem_table.copy()

    @property
    def heat_cascade(self):
        self._require_solved()
        return self._heat_cascade.copy()

    @property
    def hot_utility(self):
        self._require_solved()
        return self._hot_utility

    @property
    def cold_utility(self):
        self._require_solved()
        return self._cold_utility

    @property
    def pinch_temps(self):
        self._require_solved()
        return dict(self._pinch_temps)

    @property
    def composite_curves(self):
        self._require_solved()
        return {k: v.copy() for k, v in self._composite_curves.items()}

    @property
    def grand_composite_curve(self):
        self._require_solved()
        return self._grand_composite_curve.copy()

    def _validate_segments(self, segments: pd.DataFrame) -> pd.DataFrame:
        # ── 1. Copy & normalise column names ────────────────────────────────
        out = segments.copy()
        out.columns = out.columns.str.strip()
        if "segment_id" not in out.columns:
            # the stream names may also be the index, as returned by SysBase.pinch_stream_table()
            out = out.rename_axis("segment_id").reset_index()

        # ── 2. Required columns present ─────────────────────────────────────
        for col in ("T_in", "T_out", "CP", "T_phase", "Q"):
            if col not in out.columns:
                out[col] = np.nan

        # ── 4. Normalise categoricals ────────────────────────────────────────
        out["segment_type"] = out["segment_type"].str.strip().str.lower()
        out["side"]         = out["side"].str.strip().str.lower()

        valid_types = {"sensible", "latent"}
        bad_types = set(out["segment_type"]) - valid_types
        if bad_types:
            raise ValueError(f"Unknown segment_type(s): {bad_types}. Must be 'sensible' or 'latent'.")

        valid_sides = {"hot", "cold"}
        bad_sides = set(out["side"]) - valid_sides
        if bad_sides:
            raise ValueError(f"Unknown side(s): {bad_sides}. Must be 'hot' or 'cold'.")

        # ── 5. Build masks as numpy arrays aligned to out's index ───────────
        is_sensible = out["segment_type"].eq("sensible").to_numpy()
        is_latent   = out["segment_type"].eq("latent").to_numpy()
        is_hot      = out["side"].eq("hot").to_numpy()
        is_cold     = out["side"].eq("cold").to_numpy()

        # ── 6. Per-type validation — always use FULL-DataFrame Series ────────
        # Compute missing flags on the FULL frame so shapes match the masks
        sens_fields_missing = out[["T_in", "T_out", "CP"]].isna().any(axis=1).to_numpy()
        bad = out.loc[is_sensible & sens_fields_missing, "segment_id"].tolist()
        if bad:
            raise ValueError(f"Sensible segments missing T_in / T_out / CP: {bad}")

        latent_fields_missing = out[["T_phase", "Q"]].isna().any(axis=1).to_numpy()
        bad = out.loc[is_latent & latent_fields_missing, "segment_id"].tolist()
        if bad:
            raise ValueError(f"Latent segments missing T_phase / Q: {bad}")

        # ── 7. Numeric coercion ──────────────────────────────────────────────
        for col in ("T_in", "T_out", "CP", "T_phase", "Q"):
            out[col] = pd.to_numeric(out[col], errors="coerce")

        # ── 8. Physical sanity checks ────────────────────────────────────────
        # Sensible: hot streams must have T_in > T_out
        bad = out.loc[is_sensible & is_hot  & (out["T_in"] <= out["T_out"]), "segment_id"].tolist()
        if bad: raise ValueError(f"Hot sensible streams must have T_in > T_out: {bad}")

        bad = out.loc[is_sensible & is_cold & (out["T_in"] >= out["T_out"]), "segment_id"].tolist()
        if bad: raise ValueError(f"Cold sensible streams must have T_in < T_out: {bad}")

        bad = out.loc[is_sensible & (out["CP"] <= 0), "segment_id"].tolist()
        if bad: raise ValueError(f"CP must be > 0: {bad}")

        bad = out.loc[is_latent & (out["Q"] <= 0), "segment_id"].tolist()
        if bad: raise ValueError(f"Latent Q must be > 0: {bad}")

        # ── 9. Store masks on self AFTER all validation passes ───────────────
        self._is_sensible = is_sensible
        self._is_latent   = is_latent
        self._is_hot      = is_hot
        self._is_cold     = is_cold

        return out

    def _shift_temperatures(self) -> None:
        out = self._segments.copy()
        dth = 0.5 * self.dt_min

        for col in ["T_in_shift", "T_out_shift", "T_phase_shift", "T_high_shift", "T_low_shift"]:
            out[col] = np.nan

        sens_hot  = self._is_sensible & self._is_hot
        sens_cold = self._is_sensible & self._is_cold
        lat_hot   = self._is_latent  & self._is_hot
        lat_cold  = self._is_latent  & self._is_cold

        out.loc[sens_hot,  "T_in_shift"]  = out.loc[sens_hot,  "T_in"]  - dth
        out.loc[sens_hot,  "T_out_shift"] = out.loc[sens_hot,  "T_out"] - dth
        out.loc[sens_cold, "T_in_shift"]  = out.loc[sens_cold, "T_in"]  + dth
        out.loc[sens_cold, "T_out_shift"] = out.loc[sens_cold, "T_out"] + dth

        out.loc[self._is_sensible, "T_high_shift"] = out.loc[self._is_sensible, ["T_in_shift", "T_out_shift"]].max(axis=1)
        out.loc[self._is_sensible, "T_low_shift"] = out.loc[self._is_sensible, ["T_in_shift", "T_out_shift"]].min(axis=1)

        out.loc[lat_hot,  "T_phase_shift"] = out.loc[lat_hot,  "T_phase"] - dth
        out.loc[lat_cold, "T_phase_shift"] = out.loc[lat_cold, "T_phase"] + dth

        self._shifted_segments = out

    def _build_intervals(self) -> None:
        seg = self._shifted_segments

        levels = pd.concat([
            seg.loc[self._is_sensible, "T_high_shift"],
            seg.loc[self._is_sensible, "T_low_shift"],
            seg.loc[self._is_latent,   "T_phase_shift"],
        ]).unique()[::-1]  # unique() returns a numpy array, sorted ascending by default

        # numpy unique sorts ascending — flip to descending
        levels = np.sort(levels)[::-1]

        if levels.size < 2:
            raise ValueError(
                "At least two distinct shifted temperature levels are required "
                "to form intervals."
            )

        self._intervals = pd.DataFrame({
            "interval_id": np.arange(levels.size - 1),
            "T_upper":     levels[:-1],
            "T_lower":     levels[1:],
            "dT":          levels[:-1] - levels[1:],
        })

    def _build_problem_table(self) -> None:
        seg       = self._shifted_segments
        pt = self._intervals.copy()

        I_upper = pt["T_upper"].values
        I_lower = pt["T_lower"].values

        # ── Sensible contribution ────────────────────────────────────────────────
        T_high    = seg.loc[self._is_sensible, "T_high_shift"].values
        T_low     = seg.loc[self._is_sensible, "T_low_shift"].values
        CP        = seg.loc[self._is_sensible, "CP"].values
        CP_signed = np.where(seg.loc[self._is_sensible, "side"].values == "hot", CP, -CP)

        active = (
            (T_high[np.newaxis, :] >= I_upper[:, np.newaxis] - self.tol) &
            (T_low[np.newaxis, :]  <= I_lower[:, np.newaxis] + self.tol)
        )

        pt["net_CP"]       = (active * CP_signed[np.newaxis, :]).sum(axis=1)
        pt["sensible_dH"] = pt["net_CP"]  * pt["dT"]

        # ── Latent contribution ──────────────────────────────────────────────────
        latent = seg.loc[self._is_latent, ["T_phase_shift", "Q", "side"]].copy()
        latent["Q_signed"] = np.where(latent["side"] == "hot", latent["Q"], -latent["Q"])

        # Match each latent stream to the interval whose T_upper == T_phase_shift
        pt["latent_dH"] = (
            latent
            .assign(T_lower=latent["T_phase_shift"])
            .groupby("T_lower")["Q_signed"]
            .sum()
            .reindex(pt["T_lower"], fill_value=0.0)
            .values
        )

        # ── Assemble ─────────────────────────────────────────────────────────────
        pt["deltaH"] = pt["sensible_dH"] + pt["latent_dH"]

        self._problem_table = pt

    def _build_heat_cascade(self):
        pt   = self._problem_table.copy()
        dH   = pt["deltaH"].values

        # ── Raw cascade (assuming zero hot utility input) ────────────────────────
        # Cumulative sum top-down; negative values mean infeasibility
        raw_cascade = np.cumsum(dH)

        # ── Minimum hot utility ──────────────────────────────────────────────────
        # Add enough hot_utility to make the cascade non-negative everywhere
        hot_utility = max(0.0, -raw_cascade.min())

        # ── Feasible cascade ─────────────────────────────────────────────────────
        feasible_cascade = raw_cascade + hot_utility

        # ── Cold utility ─────────────────────────────────────────────────────────
        # Heat leaving the bottom of the cascade
        cold_utility = float(feasible_cascade[-1])

        self._hot_utility = float(hot_utility)
        self._cold_utility = float(cold_utility)

        # ── Full cascade including the top boundary ──────────────────────────────
        # Prepend hot_utility at the top temperature level (heat entering the cascade)
        full_cascade = np.r_[hot_utility, feasible_cascade]
        full_levels  = np.r_[pt["T_upper"].iloc[0], pt["T_lower"].values]

        # ── Pinch: level(s) where feasible cascade = 0 ──────────────────────────
        pinch_mask = np.isclose(full_cascade, 0.0, atol=self.tol, rtol=0.0)

        # Fallback for threshold problems: no true zero exists, pick minimum
        if not pinch_mask.any():
            pinch_mask = np.zeros(len(full_cascade), dtype=bool)
            pinch_mask[np.argmin(full_cascade)] = True

        pinch_shifted = full_levels[pinch_mask].tolist()

        self._pinch_temps = {
            "shifted": pinch_shifted,
            "hot":     [T + 0.5 * self.dt_min for T in pinch_shifted],
            "cold":    [T - 0.5 * self.dt_min for T in pinch_shifted],
        }

        # ── Store annotated problem table ────────────────────────────────────────
        pt["raw_cascade"]      = raw_cascade
        pt["feasible_cascade"] = feasible_cascade
        pt["is_pinch"]         = np.isclose(feasible_cascade, 0.0, atol=self.tol, rtol=0.0)

        self._heat_cascade = pt


    def _build_side_composite_curve(self, side: str) -> pd.DataFrame:
        seg       = self._shifted_segments
        intervals = self._intervals

        is_side = getattr(self, f"_is_{side}")

        # ── CP per interval for this side ────────────────────────────────────────
        T_high = seg.loc[self._is_sensible & is_side, "T_high_shift"].values
        T_low  = seg.loc[self._is_sensible & is_side, "T_low_shift"].values
        CP     = seg.loc[self._is_sensible & is_side, "CP"].values

        I_upper = intervals["T_upper"].values
        I_lower = intervals["T_lower"].values
        dT      = intervals["dT"].values

        # active[i, j] = True if stream j spans interval i
        active = ( (T_high[np.newaxis, :] >= I_upper[:, np.newaxis] - self.tol) & (T_low[np.newaxis, :]  <= I_lower[:, np.newaxis] + self.tol) )
        cp_per_interval = (active * CP[np.newaxis, :]).sum(axis=1)  # shape (n_intervals,)

        # ── Latent Q per boundary level for this side ────────────────────────────
        latent = seg.loc[self._is_latent & is_side, ["T_phase_shift", "Q"]]
        all_levels = np.r_[I_lower[-1], I_upper[::-1]]  # ascending: bottom → top

        latent_at_level = (
            latent
            .assign(T_upper=latent["T_phase_shift"])
            .groupby("T_upper")["Q"]
            .sum()
            .reindex(all_levels, fill_value=0.0)
            .values
        )  # shape (n_levels,) = (n_intervals + 1,)

        # ── Build (H, T_shift) points walking upward ─────────────────────────────
        # Reverse intervals to walk ascending (index n-1 → 0)
        cp_asc      = cp_per_interval[::-1]   # cp for interval i, ascending order
        dT_asc      = dT[::-1]

        H_vals = [0.0]
        T_vals = [float(all_levels[0])]  # start at bottom temperature

        has_stream_above = np.zeros(len(dT_asc), dtype=bool)
        for i in range(len(dT_asc) - 1):
            has_stream_above[i] = (cp_asc[i+1:] > self.tol).any()

        for i in range(len(dT_asc)):
            T_lo = float(all_levels[i])
            T_hi = float(all_levels[i + 1])

            q_lat_lo = float(latent_at_level[i])
            if q_lat_lo > self.tol:
                H_vals.append(H_vals[-1] + q_lat_lo)
                T_vals.append(T_lo)

            q_sens = float(cp_asc[i] * dT_asc[i])
            if q_sens > self.tol:
                H_vals.append(H_vals[-1] + q_sens)
                T_vals.append(T_hi)
            elif len(H_vals) > 1 and has_stream_above[i]:
                # True mid-range gap: active before, will be active again above
                # Add flat step to preserve the temperature boundary
                H_vals.append(H_vals[-1])
                T_vals.append(T_hi)

        # Latent at the topmost boundary
        q_lat_top = float(latent_at_level[-1])
        if q_lat_top > self.tol:
            H_vals.append(H_vals[-1] + q_lat_top)
            T_vals.append(float(all_levels[-1]))

        return pd.DataFrame({"H": H_vals, "T_shift": T_vals})

    def _build_composite_curves(self):
        hot_shifted  = self._build_side_composite_curve("hot")
        cold_shifted = self._build_side_composite_curve("cold")

        # Align cold curve: shift right by QC_min so that:
        #   left  end (low T)  starts at H = _cold_utility
        #   right end (high T) ends   at H = _cold_utility + Q_cold_total = Q_hot_total + QH_min
        cold_shifted_aligned = cold_shifted.copy()
        cold_shifted_aligned["H"] = cold_shifted_aligned["H"] + self._cold_utility

        # Convert shifted temperatures back to true temperatures
        dth = 0.5 * self.dt_min

        hot_true  = hot_shifted.copy().rename(columns={"T_shift": "T"})
        cold_true = cold_shifted_aligned.copy().rename(columns={"T_shift": "T"})

        hot_true["T"]  = hot_true["T"]  + dth
        cold_true["T"] = cold_true["T"] - dth

        self._heat_recovery = self._heat_recovery = float(min(hot_true["H"].max(), cold_true["H"].max()) - max(hot_true["H"].min(), cold_true["H"].min()) )

        self._composite_curves = {
            "hot_shifted":          hot_shifted,
            "cold_shifted":         cold_shifted,
            "cold_shifted_aligned": cold_shifted_aligned,
            "hot_true":             hot_true,
            "cold_true":            cold_true,
        }
        

    def _build_grand_composite_curve(self) -> None:
        cascade = self._heat_cascade

        # One temperature level per cascade boundary (n_intervals + 1 levels)
        # Top level gets QH_min, each subsequent level gets feasible_cascade[i]
        T_shift = np.r_[
            float(cascade["T_upper"].iloc[0]),
            cascade["T_lower"].values,
        ]
        H_vals = np.r_[
            self._hot_utility,
            cascade["feasible_cascade"].values,
        ]

        gcc = pd.DataFrame({"H": H_vals, "T_shift": T_shift})
        gcc["T_hot"]  = gcc["T_shift"] + 0.5 * self.dt_min
        gcc["T_cold"] = gcc["T_shift"] - 0.5 * self.dt_min

        self._grand_composite_curve = gcc

    def solve(self):
        self._shift_temperatures()
        self._build_intervals()
        self._build_problem_table()
        self._build_heat_cascade()
        self._build_composite_curves()
        self._build_grand_composite_curve()
        return self

    def summary(self) -> pd.Series:
        self._require_solved()
        return pd.Series({
            "dt_min": self.dt_min,
            "hot_utility": self.hot_utility,
            "cold_utility": self.cold_utility,
            "heat_recovery":   self._heat_recovery,
            "pinch_shifted": self._pinch_temps["shifted"],
            "pinch_hot": self._pinch_temps["hot"],
            "pinch_cold": self._pinch_temps["cold"],
            "n_intervals": len(self._intervals),
            "n_segments": len(self._segments),
        })

    @classmethod
    def sweep_dt_min(cls, segments: pd.DataFrame, dt_min_range=None) -> pd.DataFrame:
        """Sweep ΔT_min and return a DataFrame of utility targets vs ΔT_min."""
        if dt_min_range is None:
            dt_min_range = np.arange(2, 52, 2)

        records = []
        for dt in dt_min_range:
            try:
                pa = cls(segments, dt_min=float(dt)).solve()
                s = pa.summary()
                records.append({
                    "dt_min":        dt,
                    "hot_utility":   s["hot_utility"],
                    "cold_utility":  s["cold_utility"],
                    "heat_recovery": s["heat_recovery"],
                    "pinch_hot":     s["pinch_hot"][0] if len(s["pinch_hot"]) else np.nan,
                })
            except Exception:
                records.append({
                    "dt_min": dt,
                    "hot_utility": np.nan,
                    "cold_utility": np.nan,
                    "heat_recovery": np.nan,
                    "pinch_hot": np.nan,
                })

        return pd.DataFrame(records)

    def plot_composite_curves(self, ax=None, shifted=False, show_pinch=True, annotate_utilities=True):
        self._require_solved()

        curves = self._composite_curves
        if shifted:
            hot, cold  = curves["hot_shifted"], curves["cold_shifted_aligned"]
            ycol, ylabel = "T_shift", "Shifted temperature [°C]"
            pinch_vals = self._pinch_temps["shifted"]
        else:
            hot, cold  = curves["hot_true"], curves["cold_true"]
            ycol, ylabel = "T", "Temperature [°C]"
            pinch_vals = self._pinch_temps["hot"]

        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 5))
        else:
            fig = ax.figure

        # ── Heat recovery shading ────────────────────────────────────────────────
        # Overlap in H-space: from cold_start to hot_end
        H_overlap_min = float(cold["H"].min())   # left edge = start of cold curve
        H_overlap_max = float(hot["H"].max())    # right edge = end of hot curve

        H_common      = np.linspace(H_overlap_min, H_overlap_max, 1000)

        # Interpolate T of each curve as a function of H
        T_hot_interp  = np.interp(H_common, hot["H"].values,  hot[ycol].values)
        T_cold_interp = np.interp(H_common, cold["H"].values, cold[ycol].values)

        ax.fill_between(
            H_common, T_hot_interp, T_cold_interp,
            alpha=0.15, color="tab:green",
            label=f"Heat recovery = {self._heat_recovery:.1f}",
        )

        # ── Composite curves ─────────────────────────────────────────────────────
        ax.plot(hot["H"],  hot[ycol],  color="tab:red",  lw=2.2, label="Hot composite")
        ax.plot(cold["H"], cold[ycol], color="tab:blue", lw=2.2, label="Cold composite")

        # ── Pinch line ───────────────────────────────────────────────────────────
        if show_pinch and pinch_vals:
            for i, Tp in enumerate(pinch_vals):
                ax.axhline(Tp, color="0.35", lw=1.0, ls="--",
                        label="Pinch" if i == 0 else None)

        # ── Utility arrows ───────────────────────────────────────────────────────
        if annotate_utilities:
            # ── Hot utility shading (top-right gap between curves) ───────────────
            if self._hot_utility > self.tol:
                H_hot_end  = float(hot["H"].max())
                H_cold_end = float(cold["H"].max())
                T_top      = float(cold[ycol].iloc[-1])

                # Shade the vertical strip between cold end and hot end
                ax.axvspan(H_cold_end, H_hot_end,
                        color="tab:red", alpha=0.10, label=f"$Q_H$ = {self._hot_utility:.1f}")

                # Arrow
                ax.annotate("", xy=(H_cold_end, T_top), xytext=(H_hot_end, T_top),
                            arrowprops=dict(arrowstyle="<->", color="tab:red", lw=1.4))
                ax.text(0.5*(H_cold_end + H_hot_end), T_top,
                        f"$Q_H$ = {self._hot_utility:.1f}",
                        fontsize=8, ha="center", va="bottom", color="tab:red")

            # ── Cold utility shading (bottom-left gap between curves) ────────────
            if self._cold_utility > self.tol:
                H_cold_start = float(cold["H"].min())
                H_hot_start  = float(hot["H"].min())   # always 0
                T_bot        = float(hot[ycol].iloc[-1])

                # Shade the vertical strip between hot start and cold start
                ax.axvspan(H_hot_start, H_cold_start,
                        color="tab:blue", alpha=0.10, label=f"$Q_C$ = {self._cold_utility:.1f}")

                # Arrow
                ax.annotate("", xy=(H_cold_start, T_bot), xytext=(H_hot_start, T_bot),
                            arrowprops=dict(arrowstyle="<->", color="tab:blue", lw=1.4))
                ax.text(0.5*(H_cold_start + H_hot_start), T_bot,
                        f"$Q_C$ = {self._cold_utility:.1f}",
                        fontsize=8, ha="center", va="bottom", color="tab:blue")

        ax.set_xlim(left=0)
        ax.set_xlabel("Cumulative heat load")
        ax.set_ylabel(ylabel)
        ax.set_title("Composite curves" + (" (shifted)" if shifted else ""))
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        return fig, ax

    def plot_grand_composite_curve(self, ax=None, show_pinch=True):
        self._require_solved()
        gcc = self._grand_composite_curve

        if ax is None:
            fig, ax = plt.subplots(figsize=(6.5, 5))
        else:
            fig = ax.figure

        ax.step(
            gcc["H"], gcc["T_shift"],
            where="post",
            color="tab:green",
            lw=2.2,
            label="Grand composite curve",
        )

        if show_pinch:
            for i, Tpinch in enumerate(self._pinch_temps["shifted"]):
                ax.axhline(Tpinch, color="0.35", lw=1.0, ls="--", label="Pinch" if i == 0 else None)

        ax.axvline(0.0, color="0.4", lw=1.0)
        ax.set_xlabel("Net heat flow")
        ax.set_ylabel("Shifted temperature")
        ax.set_title("Grand composite curve")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        return fig, ax

    def plot_summary(self, shifted=False, figsize=(12, 5)):
        self._require_solved()
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        self.plot_composite_curves(ax=axes[0], shifted=shifted, show_pinch=True)
        self.plot_grand_composite_curve(ax=axes[1], show_pinch=True)
        fig.tight_layout()
        return fig, axes

    @staticmethod
    def plot_utility_sweep(sweep_df: pd.DataFrame, current_dt_min: float = None, ax=None):
        """Plot Q_H and Q_C as functions of ΔT_min."""
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker

        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 4.5))
        else:
            fig = ax.figure

        df = sweep_df.dropna(subset=["hot_utility", "cold_utility"])

        ax.plot(df["dt_min"], df["hot_utility"],  color="#c0392b", lw=2, marker="o",
                markersize=3, label=r"$Q_H$ (hot utility)")
        ax.plot(df["dt_min"], df["cold_utility"], color="#2980b9", lw=2, marker="o",
                markersize=3, label=r"$Q_C$ (cold utility)")

        if current_dt_min is not None:
            ax.axvline(current_dt_min, color="gray", ls="--", lw=1.2, label=f"Current ΔT_min = {current_dt_min}")

        ax.set_xlabel(r"$\Delta T_{min}$ [°C]")
        ax.set_ylabel("Utility load [kW]")
        ax.set_title(r"Utility targets vs $\Delta T_{min}$")
        ax.legend(framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

        fig.tight_layout()
        return fig, ax
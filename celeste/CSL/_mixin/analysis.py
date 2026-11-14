import numpy as np
import pandas as pd

from celeste import CSL


class ReportingMixin:
    """DataFrame reports, energy/exergy/mass balances, and totals."""

    def heater_df(self):
        heat_tab = {}
        for i, v in self.get_Heaters().items():
            dT = np.where((v.F1.T - v.F0.T) != 0, (v.F1.T - v.F0.T), 1)
            n = np.where(v.F0.n >= 1e-5, v.F0.n, 1e-8)
            heat_tab[i] = {
                'CP': v.Q / dT,
                'T0': v.F0.T - 273.15,
                'T1': v.F1.T - 273.15,
                'deltaT': v.F1.T - v.F0.T,
                'Q': np.where(np.abs(v.Q) > 1e-7, v.Q, 0).round(5),
                'm': v.F0.m,
                'cp': v.Q / (n * dT),
                'MW': v.F0.mw,
                'Exth': v.Exth
            }
        return self._table(heat_tab, ['CP', 'T0', 'T1', 'deltaT', 'Q', 'm', 'cp', 'MW', 'Exth']).sort_values('Q')

    @staticmethod
    def _table(records: dict, columns: list) -> pd.DataFrame:
        """DataFrame with one row per record and float columns, also when there are no records."""
        if not records:
            return pd.DataFrame(columns=columns, dtype=float)
        return pd.DataFrame(records).transpose().astype(float)

    def pinch_stream_table(self, exclude: list[str] | None = None, include: list[str] | None = None,
                            min_duty: float = 1.0, min_delta_t: float = 0.5) -> pd.DataFrame:
        """
        Build a PinchAnalyser-compatible stream table directly from the system's
        Heater objects.

        The index is the dotted attribute path relative to this system object
        (e.g. 'socbox.HT_F'), usable as self.rgetattr(df.index[i]).

        Parameters
        ----------
        exclude     : substrings — drop heaters whose name contains any of these
        include     : substrings — keep only heaters whose name contains any of these
                    (takes precedence over exclude)
        min_duty    : minimum |Q| [kW] to include a stream
        min_delta_t : |ΔT| [°C] below which a stream is classified as latent
        """
        prefix = self.name + "."

        def _attr_path(full_name: str) -> str:
            return full_name[len(prefix):] if full_name.startswith(prefix) else full_name

        heaters = self.get_Heaters()

        if include is not None:
            heaters = {k: v for k, v in heaters.items() if any(s in k for s in include)}
        elif exclude is not None:
            heaters = {k: v for k, v in heaters.items() if not any(s in k for s in exclude)}

        if not heaters:
            raise ValueError("No heaters remaining after name filtering.")

        rows = []
        for raw_id, h in heaters.items():
            Q = float(h.Q)
            abs_Q = abs(Q)

            if abs_Q < min_duty:
                continue

            T0 = h.F0.T - 273.15
            T1 = h.F1.T - 273.15
            abs_dT = abs(T1 - T0)
            sid = _attr_path(raw_id)

            if abs_dT < min_delta_t:
                rows.append({
                    "segment_id": sid,
                    "segment_type": "latent",
                    "side": "hot" if Q < 0 else "cold",
                    "T_in": np.nan,
                    "T_out": np.nan,
                    "CP": np.nan,
                    "T_phase": T0,
                    "Q": abs_Q,
                })
            else:
                rows.append({
                    "segment_id": sid,
                    "segment_type": "sensible",
                    "side": "hot" if Q < 0 else "cold",
                    "T_in": T0,
                    "T_out": T1,
                    "CP": abs_Q / abs_dT,
                    "T_phase": np.nan,
                    "Q": np.nan,
                })

        if not rows:
            raise ValueError(
                "No streams remaining after duty filter. "
                "Adjust min_duty or your include/exclude lists."
            )

        df_out = pd.DataFrame(rows).set_index("segment_id")

        dupes = df_out.index[df_out.index.duplicated(keep=False)].unique()
        if len(dupes):
            counts = {}
            new_idx = []
            for sid in df_out.index:
                if sid in dupes:
                    counts[sid] = counts.get(sid, 0) + 1
                    new_idx.append(f"{sid}_{counts[sid]}")
                else:
                    new_idx.append(sid)
            df_out.index = new_idx

        return df_out[["segment_type", "side", "T_in", "T_out", "CP", "T_phase", "Q"]]

    def heat_df(self):
        heat_tab = {}
        heat = self.get_Heat()
        for i, v in heat.items():
            heat_tab[i] = {
                'Q': np.where(np.abs(v.Q) > 1e-7, v.Q, 0).round(5),
                'Exth': np.where(np.abs(v.Exth) > 1e-7, v.Exth, 0).round(5)
            }
        return self._table(heat_tab, ['Q', 'Exth']).sort_values('Q')

    def work_df(self):
        """Work of every component exposing ``W`` [kJ/s], as a Series named 'W'."""
        Ws = {f'{self.name}.{i}': float(v.W) for i, v in self.get_Electric().items()}
        return pd.Series(Ws, name='W', dtype=float)

    def flow_df(self):
        tab = {}
        for i, v in self.get_Sources().items():
            tab[i] = {
                'n': v.F1.n,
                'm': v.F1.m,
                'ex': v.F1.Ex,
                'EX': v.F1.n * v.F1.Ex,
                'lhv': v.F1.LHV,
                'LHV': v.F1.n * v.F1.LHV
            }

        for i, v in self.get_Sinks().items():
            tab[i] = {
                'n': v.F0.n,
                'm': v.F0.m,
                'ex': v.F0.Ex,
                'EX': v.F0.n * v.F0.Ex,
                'lhv': v.F0.LHV,
                'LHV': v.F0.n * v.F0.LHV
            }
        return self._table(tab, ['n', 'm', 'ex', 'EX', 'lhv', 'LHV'])

    def exloss_df(self):
        loss = {}
        for i, v in self.get_Exloss().items():
            loss[i] = {
                'Ex loss': np.where(np.abs(v.Exloss) > 1e-7, v.Exloss, 0).round(5)
            }
        return self._table(loss, ['Ex loss'])

    def tot_LHV_Flow_out(self):
        return np.sum([v.F0.LHV * v.F0.n for i, v in self.get_Sinks().items()])

    def tot_LHV_Flow_in(self):
        return np.sum([v.F1.LHV * v.F1.n for i, v in self.get_Sources().items()])

    def tot_Ex_Flow_out(self):
        return np.sum([v.F0.Ex * v.F0.n for i, v in self.get_Sinks().items()])

    def tot_Ex_Flow_in(self):
        return np.sum([v.F1.Ex * v.F1.n for i, v in self.get_Sources().items()])

    def tot_Exloss(self):
        return np.sum([v.Exloss for i, v in self.get_Exloss().items()])

    def tot_heaters_in(self):
        df = self.heater_df()
        return df[(df.Q > 0)].Q.sum()

    def tot_heaters_out(self):
        df = self.heater_df()
        return df[(df.Q < 0)].Q.sum()

    def tot_heat_in(self):
        df = self.heat_df()
        return df[(df.Q > 0)].Q.sum()

    def tot_heat_out(self):
        df = self.heat_df()
        return df[(df.Q < 0)].Q.sum()

    def tot_work_in(self):
        W = self.work_df()
        return W[W > 0].sum()

    def tot_work_out(self):
        W = self.work_df()
        return W[W < 0].sum()

    def tot_work_comp(self):
        """Total work of pumps and compressors [kJ/s]."""
        W = 0
        for i, v in self.get_Electric().items():
            if isinstance(v, (CSL.components.Pump, CSL.turbomachinery.Compressor)) and 'valve' not in i and 'Valve' not in i:
                W += np.where(np.abs(v.W) > 1e-7, v.W, 0)
        return W

    def tot_exth_in(self):
        df1 = self.heat_df()
        df2 = self.heater_df()
        return df1[(df1.Exth > 0)].Exth.sum() + df2[(df2.Exth > 0)].Exth.sum()

    def tot_exth_out(self):
        df1 = self.heat_df()
        df2 = self.heater_df()
        return df1[(df1.Exth < 0)].Exth.sum() + df2[(df2.Exth < 0)].Exth.sum()

    def tot_valveLoss(self):
        W = 0
        for i in self.r_get():
            if 'valve' in i or 'Valve' in i:
                W += self.rgetattr(i).W
        return W

    def tot_Qloss(self):
        """Sum of the heat losses of all SOC stacks [kJ/s]."""
        return sum(unit.Qloss for _, unit in self._units() if isinstance(unit, CSL.electrochem.SOC))

    
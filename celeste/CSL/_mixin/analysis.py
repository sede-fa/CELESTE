import warnings

import numpy as np
import pandas as pd

from celeste import CSL
from celeste._deprecation import deprecated


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
        if not heat_tab:
            warnings.warn('heater_df: the system has no heaters, the DataFrame is empty')
            return pd.DataFrame(columns=['CP', 'T0', 'T1', 'deltaT', 'Q', 'm', 'cp', 'MW', 'Exth'], dtype=float)
        return pd.DataFrame(heat_tab).transpose().sort_values('Q')

    def pinch_stream_table(self, exclude: list[str] | None = None, include: list[str] | None = None,
                            min_duty: float = 1.0, min_delta_t: float = 0.5) -> pd.DataFrame:
        """
        Build a PinchAnalyser-compatible stream table directly from the system's
        Heater objects.

        The index is the dotted attribute path relative to this system object
        (e.g. 'socbox.HT_F'), usable as self.rgetattr(df.index[i]).

        Parameters
        ----------
        exclude     : substrings - drop heaters whose name contains any of these
        include     : substrings - keep only heaters whose name contains any of these
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
                'Exth': np.where(np.abs(getattr(v, 'Exth', 0.)) > 1e-7, getattr(v, 'Exth', 0.), 0).round(5)   # a mixer has Q but no Exth
            }
        if not heat_tab:
            warnings.warn('heat_df: the system has no heat flows, the DataFrame is empty')
            return pd.DataFrame(columns=['Q', 'Exth'], dtype=float)
        return pd.DataFrame(heat_tab).transpose().sort_values('Q')

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
        if not tab:
            warnings.warn('flow_df: the system has no sources or sinks, the DataFrame is empty')
            return pd.DataFrame(columns=['n', 'm', 'ex', 'EX', 'lhv', 'LHV'], dtype=float)
        return pd.DataFrame(tab).transpose()

    def exergy_df(self) -> pd.DataFrame:
        """
        Exergy destroyed and exergy left (rejected as heat) by every unit [kJ/s].

        ``Exdest`` is the irreversibility of the unit (``T0*Sgen``) and can be reduced by a better design;
        ``Exloss`` is the exergy that leaves the unit to the environment as heat at T > T0 and is not used by the process.
        ``Exbalance_error`` is the exergy balance of the unit minus ``T0*Sgen``: it measures how consistent the
        exergy and entropy data of its streams are (0 for a consistent unit).
        """
        tab = {}
        for i, v in self.get_Exdest().items():
            r = lambda x: np.where(np.abs(x) > 1e-7, x, 0).round(5)
            tab[i] = {'Exdest': r(v.Exdest), 'Exloss': r(getattr(v, 'Exloss', 0)),
                      'Exbalance_error': r(getattr(v, 'Exbalance_error', 0))}
        if not tab:
            warnings.warn('exergy_df: the system has no units with exergy destruction, the DataFrame is empty')
            return pd.DataFrame(columns=['Exdest', 'Exloss', 'Exbalance_error'], dtype=float)
        df = pd.DataFrame(tab).transpose().astype(float)
        neg = df.index[df.Exdest < -1e-3]
        if len(neg):
            warnings.warn(f'exergy_df: negative exergy destruction (T0*Sgen < 0) in {list(neg)}: the second law is violated '
                          'inside the unit, the property models of its streams are not consistent '
                          '(for example a flash with NASA media: phase split from the Peng-Robinson equation)')
        return df

    def exergy_summary(self, products=None, hen=None) -> pd.Series:
        """
        Exergy balance of the system [kJ/s] and its efficiencies.

        ``products`` names the sinks or outlet flanges (name or path, or a list of them) whose stream is the useful product;
        by default every sink and every outlet flange of the system is a product. The other sinks and the heat that leaves the units at T > T0 count
        as exergy left over. Net work delivered by units (W < 0) is a product.

        ``Ex_in = Ex_product + Ex_unused + Exloss + Exdest`` (``Ex_balance`` is the remainder: the sum of the ``Exbalance_error`` of the units,
        i.e. how well the exergy and entropy data agree; ~0 for consistent models)
        ``efficiency      = Ex_product/Ex_in``
        ``efficiency_max  = 1 - (Ex_unused + Exloss)/Ex_in``, the efficiency with every irreversibility removed
        ``destruction_share = Exdest/Ex_in``

        ``hen`` is a solved ``celeste.HEN.HENSynthesiser`` built from ``pinch_stream_table()`` of this system: the
        summary is then that of the heat-integrated plant (see ``hen_exergy``): the recovered heat no longer leaves
        the coolers or has to be supplied to the heaters, the new process-process exchangers destroy exergy through
        their temperature difference. ``Q_recovered`` and ``efficiency_no_integration`` are added.
        """
        if isinstance(products, str):
            products = [products]
        out_prod = out_unused = 0.
        for i, v in self.get_Sinks().items():
            ex = float(v.F0.n*v.F0.Ex)
            if products is None or i in products or v.name in products or i.split('.')[-1] in products:
                out_prod += ex
            else:
                out_unused += ex
        owners = self._owner_map()
        feeds = 0.
        for o, i in self.streams:                          # flanges of the system itself: feeds and products
            if owners[id(o)] is self:
                feeds += float(o.n*o.Ex)
            elif owners[id(i)] is self:
                if products is None or i.name in products or i.name.split('.')[-1] in products:
                    out_prod += float(i.n*i.Ex)
                else:
                    out_unused += float(i.n*i.Ex)
        W = self.work_df()
        w_in = float(W[W > 0].sum())
        ex_in = float(self.tot_Ex_Flow_in()) + feeds + w_in + float(self.tot_exth_in())
        out_prod += -float(W[W < 0].sum())
        dest = float(self.tot_Exdest())
        left = float(self.tot_Exloss())
        tab = {'Ex_in': ex_in, 'Ex_product': out_prod, 'Ex_unused': out_unused, 'Exloss': left, 'Exdest': dest,
               'Ex_balance': ex_in - out_prod - out_unused - left - dest}
        if hen is not None:
            eff0 = out_prod/ex_in if abs(ex_in) > 1e-12 else np.nan
            exch = self.hen_exergy(hen)
            ex_in += exch.attrs['d_in']
            left += exch.attrs['d_loss']
            dest += exch.attrs['d_dest']
            tab = {'Ex_in': ex_in, 'Ex_product': out_prod, 'Ex_unused': out_unused, 'Exloss': left, 'Exdest': dest,
                   'Ex_balance': ex_in - out_prod - out_unused - left - dest, 'Q_recovered': float(exch.Q.sum())}
        if abs(ex_in) > 1e-12:
            tab.update(efficiency=out_prod/ex_in, efficiency_max=1 - (out_unused + left)/ex_in,
                       destruction_share=dest/ex_in)
        if hen is not None:
            tab['efficiency_no_integration'] = eff0
        return pd.Series(tab, name=self.name, dtype=float)

    def integrate(self, hen, mode: str = 'duty', **kw):
        """
        A solved copy of this system with the heat exchanger network ``hen`` built in, to verify it: every heater
        and cooler of the network becomes its exchanger sides and a trim heater or cooler. ``mode='duty'`` keeps
        the duties of the network, ``mode='area'`` its areas (the duties follow). The comparison is the copy's
        ``hen_report``. See ``celeste.HEN.integrate``.
        """
        from celeste.HEN.integrate import integrate
        return integrate(self, hen, mode=mode, **kw)

    def hen_exergy(self, hen) -> pd.DataFrame:
        """
        Exergy of the heat recovered by the process-process exchangers of a heat exchanger network [kJ/s].

        One row per exchanger of ``hen.network`` (built from this system's ``pinch_stream_table()``): the heat ``Q``,
        its exergy on the hot side ``Ex_hot = Q (1 - T0/T_lm,hot)`` and on the cold side ``Ex_cold = Q (1 - T0/T_lm,cold)``
        with the logarithmic mean temperatures of the stream sections, and the destruction of the exchanger
        ``Exdest = Ex_hot - Ex_cold = T0 Q (1/T_lm,cold - 1/T_lm,hot)``.

        Changes of the exergy balance of the system when the heat is recovered (``attrs``):

        * ``d_in``   the heater that is replaced no longer takes exergy: ``-Q`` (heaters count their duty as exergy,
          like electric heating, see ``Heater.Exth``),
        * ``d_loss`` the cooler no longer rejects the share ``Q/|Q_cooler|`` of its heat: ``-Exloss_cooler*share``,
        * ``d_dest`` the exchanger destroys ``Exdest``; the part of the cooler and heater destruction that went with
          the recovered heat disappears.

        The three changes keep ``Ex_in = Ex_product + Exloss + Exdest`` closed.
        """
        from celeste.CSL.exergy import T0
        net = hen.network
        if net is None:
            raise ValueError('hen_exergy: the HEN has not been solved (HENSynthesiser.solve())')

        def lm(t_in, t_out):                                  # logarithmic mean of two temperatures, K
            a, b = t_in + 273.15, t_out + 273.15
            return a if abs(a - b) < 1e-9 else (a - b)/np.log(a/b)

        rows, d_in, d_loss, d_dest = [], 0., 0., 0.
        for _, r in net.iterrows():
            Th, Tc = lm(r.T_hot_in, r.T_hot_out), lm(r.T_cold_in, r.T_cold_out)
            ex_h, ex_c = r.Q*(1 - T0/Th), r.Q*(1 - T0/Tc)
            try:
                hot, _ = self.rgetattr(r.hot), self.rgetattr(r.cold)
            except (AttributeError, KeyError) as err:
                raise ValueError(f'hen_exergy: the exchanger {r.hot} -> {r.cold} has a stream that is not a heater of '
                                 f'{self.name}; add the external streams to the system or remove them') from err
            ex_unit_h = r.Q*(1 - T0/hot.F1.T)                  # what the cooler accounted for this heat
            # stream exergy drop of the recovered part of the hot stream, and gain of the cold stream
            d_in += -r.Q
            d_loss += -ex_unit_h
            d_dest += (ex_h - ex_c) - (ex_h - ex_unit_h) - (r.Q - ex_c)
            rows.append(dict(hot=r.hot, cold=r.cold, Q=r.Q, T_lm_hot=Th, T_lm_cold=Tc, Ex_hot=ex_h, Ex_cold=ex_c,
                             Exdest=ex_h - ex_c))
        out = pd.DataFrame(rows, columns=['hot', 'cold', 'Q', 'T_lm_hot', 'T_lm_cold', 'Ex_hot', 'Ex_cold', 'Exdest'])
        out.attrs.update(d_in=d_in, d_loss=d_loss, d_dest=d_dest)
        return out

    @deprecated('exloss_df is replaced by exergy_df (Exdest and Exloss per unit); Exloss now means the exergy lost to the environment', removal=None)
    def exloss_df(self):
        return self.exergy_df()

    def tot_LHV_Flow_out(self):
        return np.sum([v.F0.LHV * v.F0.n for i, v in self.get_Sinks().items()])

    def tot_LHV_Flow_in(self):
        return np.sum([v.F1.LHV * v.F1.n for i, v in self.get_Sources().items()])

    def tot_Ex_Flow_out(self):
        return np.sum([v.F0.Ex * v.F0.n for i, v in self.get_Sinks().items()])

    def tot_Ex_Flow_in(self):
        return np.sum([v.F1.Ex * v.F1.n for i, v in self.get_Sources().items()])

    def tot_Exdest(self):
        return np.sum([v.Exdest for i, v in self.get_Exdest().items()])

    def tot_Exloss(self):
        """Exergy lost to the environment as heat at T > T0, kJ/s."""
        return np.sum([getattr(v, 'Exloss', 0.) for i, v in self.get_Exdest().items()])

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

    
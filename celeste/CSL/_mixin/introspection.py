import numpy as np
import pandas as pd

from celeste import CSL


class IntrospectionMixin:
    """Hierarchy traversal and component lookup (sources, sinks, heaters, flanges)."""

    def r_get(self, prefix=''):
        """
        Recursively retrieves the names of all attributes within the units in system.

        This method traverses the attributes of the current system, and for each attribute that is an instance of ModelBase,
        it adds its name to the output list. If the attribute is also an instance of SysBase (i.e., a subsystem),
        the method is called recursively on that subsystem, and the names are prefixed accordingly.

        Args:
            prefix (str, optional): String to prepend to each attribute name, used for hierarchical naming in nested systems.
                                    Defaults to ''.

        Returns:
            list of str: A list of dot-separated attribute names representing all ModelBase instances in the system hierarchy.
        """
        from .._base import ModelBase
        out = []
        for key, obj in self._members(ModelBase):
            if getattr(obj, "_is_system", False):
                out += obj.r_get(prefix + key + '.')
            else:
                out += [prefix + key]
        return out

    def _units(self):
        """(path, unit) pairs of every unit model in the hierarchy."""
        return [(path, self.rgetattr(path)) for path in self.r_get()]

    def get_AllFlanges(self):
        """
        Retrieves all flange objects (flow interfaces) within the system hierarchy.

        Recursively searches through all ModelBase components and subsystems, collecting
        every flange (flow connection point) defined in the system.

        Returns:
            list: A list of Flange objects from all components in the system.
        """
        flanges = []
        for i in self.r_get():
            flanges += self.rgetattr(i).get_Flanges()
        return flanges

    def ViewAllFlange(self):
        """
        Provides a summarized view of all flanges in the system.

        Returns:
            pandas.DataFrame or str: A structured view showing key properties of all flanges,
            typically including flow rates, pressures, temperatures, and compositions.
        """
        return self.ViewFlange(self.get_AllFlanges())

    def get_success(self):
        """
        Compiles the convergence status of all numerical solvers in the system.

        Checks components that have an 'f_sol' method (indicating they perform numerical solving)
        and returns their solver success status.

        Returns:
            dict: A dictionary mapping component paths to boolean success flags.
                True indicates successful convergence, False indicates failure.
        """
        success = {}
        for i, unit in self._units():
            if hasattr(unit, 'f_sol'):
                sol = getattr(unit, '_sol', None)
                # None: the unit has not been solved through a system yet, or it is evaluated explicitly
                success[i] = None if sol is None else bool(getattr(sol, 'success', True))
        return success

    def get_Electric(self):
        """
        Components that exchange work with the surroundings (anything exposing ``W``).

        Returns:
            dict: A dictionary mapping component paths to their object references.
        """
        return {i: unit for i, unit in self._units() if hasattr(unit, 'W')}

    def get_Sources(self):
        """
        Identifies all source components in the system hierarchy.

        Sources are defined as components that introduce material/energy flows into the system
        (instances of CSL.flow.SourceM).

        Returns:
            dict: A dictionary mapping source component paths to their object references.
        """
        sources = {}
        for i in self.r_get():
            if isinstance(self.rgetattr(i), CSL.flow.SourceM):
                sources[i] = self.rgetattr(i)
        return sources

    def get_Sinks(self):
        """
        Identifies all sink components in the system hierarchy.

        Sinks are defined as components that remove material/energy flows from the system
        (instances of CSL.flow.Sink).

        Returns:
            dict: A dictionary mapping sink component paths to their object references.
        """
        sinks = {}
        for i in self.r_get():
            if isinstance(self.rgetattr(i), CSL.flow.Sink):
                sinks[i] = self.rgetattr(i)
        return sinks

    def get_Heaters(self):
        heaters = {}
        for i in self.r_get():
            if isinstance(self.rgetattr(i), CSL.heat.Heater):
                heaters[f'{self.name}.{i}'] = self.rgetattr(i)
        return heaters

    def get_Heat(self):
        heat = {}
        for i in self.r_get():
            if 'Q' in dir(self.rgetattr(i)) and not isinstance(
                self.rgetattr(i), (CSL.heat.Heater, CSL.heat.HexBase, CSL.components.DAC)
            ):
                heat[f'{self.name}.{i}'] = self.rgetattr(i)
        return heat

    def get_Exloss(self):
        exloss = {}
        for i in self.r_get():
            if not isinstance(self.rgetattr(i), (CSL.flow.SourceM, CSL.flow.Sink)):
                exloss[f'{self.name}.{i}'] = self.rgetattr(i)
        return exloss

    def SBcheck(self):
        sg = {}
        for i in self.r_get():
            if not isinstance(self.rgetattr(i), (CSL.flow.SourceM, CSL.flow.Sink)):
                sgen = getattr(self.rgetattr(i), 'Sgen', np.nan)   # nan: the unit defines no entropy balance
                sg[i] = {
                    'Sgen': sgen,
                    'Sgen*T0': sgen*298.15
                    }
        return pd.DataFrame(sg).transpose()
    
    def exdest_df(self):
        """Exergy destruction grouped by component class, in kW and % of the total."""
        groups = {'Compressors': 0, 'SOC': 0, 'ASU': 0, 'DAC': 0, 'Heaters': 0, 'Coolers': 0, 'GPU': 0,
                  'Reactor': 0, 'Separation': 0, 'Split_Mix': 0, 'Valves': 0}
        tot = self.tot_Exloss()
        for i, v in self.get_Exloss().items():
            if isinstance(v, (CSL.components.Pump, CSL.components.Compressor)):
                groups['Compressors'] += v.Exloss
            if isinstance(v, CSL.electrochem.SOC):
                groups['SOC'] += v.Exloss
            if isinstance(v, CSL.components.ASU):
                groups['ASU'] += v.Exloss
            if isinstance(v, CSL.components.DAC):
                groups['DAC'] += v.Exloss
            if isinstance(v, CSL.heat.Heater):
                groups['Heaters'] += np.where(v.Q > 0, v.Exloss, 0)
                groups['Coolers'] += np.where(v.Q < 0, v.Exloss, 0)
            if isinstance(v, (CSL.reactors.EquilBase, CSL.reactors.ExtentReactor, CSL.reactors.HB_1D, CSL.reactors.Reformer)):
                groups['Reactor'] += v.Exloss
            if isinstance(v, CSL.components.GPUbase):
                groups['GPU'] += v.Exloss
            if isinstance(v, CSL.separation.Flash):
                groups['Separation'] += v.Exloss
            if isinstance(v, (CSL.flow.Mixer, CSL.flow.Splitter)):
                groups['Split_Mix'] += v.Exloss
            if isinstance(v, CSL.components.Valve) or 'valve' in i or 'Valve' in i:
                groups['Valves'] += v.Exloss
        groups['Total'] = tot
        ex = pd.DataFrame({'Ex. destruction / kW': {k: float(v) for k, v in groups.items()}})
        ex.insert(1, '%', ex.values*100/tot)
        return ex

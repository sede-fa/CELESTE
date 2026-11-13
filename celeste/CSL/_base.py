import numpy as np
import pandas as pd
import scipy.optimize as spo

from .interfaces import Flange
from .. import Base

from ._mixin.introspection import IntrospectionMixin
from ._mixin.graph import GraphMixin
from ._mixin.analysis import ReportingMixin
from ._mixin.recycle import RecycleMixin

from typing import Any
from numpy._typing._array_like import NDArray

class ModelBase(Base):
    '''
    Base for creating models such as pins and vessels with flexible attribute definition

    Conventions shared by all unit models (see docs/conventions.md):

    * ports are ``Flange`` attributes; a port whose name ends in ``0`` is an
      inlet (``F0``, ``A0``, ``H0``, ``C0``, ``FH0``), every other port is an outlet
    * ``run_sol()`` brings the outlets in line with the inlets and parameters
    * ``Q`` / ``W`` are heat and work added to the unit in kJ/s (kW)
    * ``Exloss`` is the exergy destroyed in kJ/s, ``Sgen`` the entropy generated in kJ/K/s
    '''

    def __init__(self, args):
        super().__init__(args)
        self.Eq = np.array([])

    def sol_seq(self):
        return None

    def Eqs(self) -> list[float] | NDArray[np.float64]:
        return []

    @property
    def Exloss(self):
        """Exergy destruction in kJ/s. Units override this; models that still assign it in ``ExBs()`` may do so."""
        return self.__dict__.get('_Exloss_value', 0)

    @Exloss.setter
    def Exloss(self, value):
        self.__dict__['_Exloss_value'] = value

    @staticmethod
    def ViewFlange(flanges):
        """
        View details of flanges in a tabular format.

        Args:
            flanges (Union[ModelBase, List[Flange]]): A ModelBase object or a list of Flange objects.

        Returns:
            pd.DataFrame: DataFrame containing flange details.
        
        Raises:
            ValueError: If an invalid object is passed.
        """
        if isinstance(flanges, ModelBase):
            flangeList = flanges.get_Flanges()
        elif isinstance(flanges, list) and all(hasattr(f, "get_nXTP") for f in flanges):
            flangeList = flanges
        else:
            raise ValueError("Invalid input: Expected a ModelBase object or a list of Flange objects.")

        return (pd.DataFrame([f.get_nXTP() for f in flangeList])
                .transpose()
                .fillna(0)
                #.clip(lower=1e-8)
        )

    def _members(self, kind) -> list:
        """(name, object) pairs of the instance attributes of a given type, sorted by name."""
        return [(k, v) for k, v in sorted(vars(self).items()) if isinstance(v, kind)]

    def get_Flanges(self) -> list[Flange]:
        """
        Get all Flange objects associated with this model.

        Returns:
            List[Flange]: A list of Flange objects.
        """
        return [v for _, v in self._members(Flange)]

    #: default state of a stand-alone unit, per port: {'F0': dict(n=..., Xi=..., T=..., P=...), 'F1': dict(P=...)}.
    #: Inlet entries are the feed; outlet entries are the operating specification (for example the outlet
    #: pressure of a compressor) or the starting point of an iterative solve.
    _default_state: dict = {}

    def _init_state(self, legacy: dict) -> None:
        '''
        Apply the default state (``_default_state``) and the former state keywords
        (``F0_n``, ``F0_Xi``, ``F0_T``, ``F0_P``, ``F1_P``, ...) to the flanges.
        The state keywords still work but are deprecated: the operating state belongs on the flanges.
        '''
        for port, state in self._default_state.items():
            for var, value in state.items():
                setattr(getattr(self, port), var, value)
        used = []
        for key, value in legacy.items():
            port, _, var = key.partition('_')
            flange = getattr(self, port, None)
            if not isinstance(flange, Flange) or var not in ('n', 'Xi', 'T', 'P'):
                raise TypeError(f"{type(self).__name__}() got an unexpected keyword argument '{key}'")
            if value is None:
                continue
            used.append(key)
            setattr(flange, var, value)
        if used:
            from .._deprecation import warn_deprecated
            port = used[0].partition('_')[0]
            var = used[0].partition('_')[2]
            warn_deprecated(f"{type(self).__name__}({', '.join(used)}=...): state keywords are deprecated; set the "
                            f"operating state on the flanges, e.g. unit.{port}.{var} = ...",
                            stacklevel=self._caller_level())

    def _caller_level(self) -> int:
        """``stacklevel`` that points a warning at the code that created this unit (past nested ``__init__``)."""
        import sys
        frame, level = sys._getframe(2), 3     # 1: warnings.warn in warn_deprecated, 2: _init_state
        while frame is not None and frame.f_locals.get('self') is self:
            frame, level = frame.f_back, level + 1
        return level

    @staticmethod
    def _spread(legacy: dict, ports, **common) -> dict:
        """Turn former keywords that applied to several ports (``T=``, ``P=``) into per-port state keywords."""
        for var, value in common.items():
            if value is not None:
                for port in ports:
                    legacy.setdefault(f'{port}_{var}', value)
        return legacy

    #: names of the inlet ports; None applies the naming convention (port name ends in ``0``)
    _inlet_ports: tuple | None = None

    def _is_inlet(self, port_name: str) -> bool:
        if self._inlet_ports is not None:
            return port_name in self._inlet_ports
        return port_name.endswith('0')

    @property
    def inlets(self) -> list[Flange]:
        """Inlet ports: flanges whose attribute name ends in ``0`` (or the ones listed in ``_inlet_ports``)."""
        return [v for k, v in self._members(Flange) if not k.startswith('_') and self._is_inlet(k)]

    @property
    def outlets(self) -> list[Flange]:
        """Outlet ports: every flange that is not an inlet."""
        return [v for k, v in self._members(Flange) if not k.startswith('_') and not self._is_inlet(k)]

    def mass_balance(self) -> float:
        """Mass flow into the unit minus mass flow out of it, g/s (no printing)."""
        return sum(f.m for f in self.inlets) - sum(f.m for f in self.outlets)

    def heat_and_work(self) -> float:
        """Heat plus work added to the unit, kJ/s (``Q`` + ``W``; a unit without them adds nothing)."""
        return float(getattr(self, 'Q', 0.0) or 0.0) + float(getattr(self, 'W', 0.0) or 0.0)

    def energy_balance(self) -> float:
        """Enthalpy in + heat and work added - enthalpy out, kJ/s; zero when the unit's energy balance closes."""
        H_in = sum(f.n*f.h for f in self.inlets)
        H_out = sum(f.n*f.h for f in self.outlets)
        return H_in - H_out + self.heat_and_work()

    def MBcheck(self) -> float:
        """Print the relative mass balance error and return the difference in - out, g/s."""
        m_in = sum(f.m for f in self.inlets)
        m_out = sum(f.m for f in self.outlets)
        print(f"{abs((m_in/m_out) - 1)*100}% difference ")
        return m_in - m_out


class SysBase(GraphMixin, ReportingMixin, IntrospectionMixin, RecycleMixin, ModelBase):
    _is_system=True
    def __init__(self, args): # name,
        super().__init__(args) #

        self.Eq = np.array([])
        self.Objs = np.array([], dtype=ModelBase)
        self._core = ['name', 'self','Medium']
        self.connections = []
        self._links = {}

    def _record_link(self, pin1, pin2) -> None:
        """Remember that two flanges carry the same stream (used by the boundary balances)."""
        if '_links' not in self.__dict__:
            self._links = {}
        self._links[(pin1, pin2)] = None

    @staticmethod
    def _check_connection(pin1, pin2) -> None:
        reason = pin1.incompatibility(pin2)
        if reason is not None:
            raise ValueError(f"Incorrect connection between {pin1.name} and {pin2.name}: {reason}")

    def connect(self, pin1, pin2) -> None:
        """
        Equation-based connection: append the residuals that tie two flanges together to ``self.Eq``
        (equality of n, T, P and zero sum of the flow variables Q, I).

        Raises:
            ValueError: if the two pins are not of the same type / medium.
        """
        Eq = np.array([])
        self._check_connection(pin1, pin2)
        self._record_link(pin1, pin2)
        
        for attr in pin1.__dict__.keys(): #for each flange of the other components 
            if any(k in attr for k in self._core):
                continue 

            if attr in ['Q','I']: # Flow variables sum to zero at interface
                Eq = np.append(Eq, [pin1.__dict__[attr] + pin2.__dict__[attr]])
            elif attr in ['n', 'T', 'P']: # add equations describing the equality on the flange
                Eq = np.append(Eq, [ pin1.__dict__[attr] - pin2.__dict__[attr] ])
            
        self.Eq = np.append(self.Eq, Eq)

    def connect_seq(self, pin1:Flange, pin2:Flange, direction:str='F', store:bool=False) -> None:
        '''
        forces assignment of one pins conditions to the other

        direction, str = F for forward, B for backward
        '''

        self._check_connection(pin1, pin2)
        self._record_link(pin1, pin2)

        if store:
            self.connections.append(['.'.join(pin1.name.split('.')[:-1]), pin1.name.split('.')[-1], pin2.name.split('.')[-1], '.'.join(pin2.name.split('.')[:-1])])
        
        for var in pin1._pin_vars:
            if var in pin1._flow_vars:
                if direction.lower() == 'b':
                    setattr(pin1, var, -getattr(pin2, var))
                else:
                    setattr(pin2, var, -getattr(pin1, var))
            else:
                if direction.lower() == 'b':
                    setattr(pin1, var, getattr(pin2, var))
                else:
                    setattr(pin2, var, getattr(pin1, var))

    def getObjs(self) -> list[ModelBase]:
        """
        Get all objects added to the system.

        Returns:
            List[object]: List of objects added to the system.
        """
        return [v for v in self.__dict__.values() if isinstance(v, ModelBase)]

    def seq_sol(self, store: bool = False) -> None:
        """
        Sequential-modular solution of the system: connect the units in flow
        order with ``connect_seq`` and call their ``run_sol``. Implemented by
        every concrete system.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement seq_sol()")

    def run_sol(self):
        """Solve the system (default: one sequential-modular pass)."""
        return self.seq_sol()

    def makeSys(self) -> None:
        """
        Aggregate equations from all objects in the system.
        """
        self.Eq = np.concatenate([obj.Eqs() for obj in self.getObjs()])

    @staticmethod
    def GradientDescent(fun, x, r=1e-2, m=0.03, eps=1e-6, Niter=5, print_steps=False):
        """
        Perform gradient descent optimization.

        Args:
            fun (callable): Objective function.
            x (np.ndarray): Initial guess.
            r (float): Learning rate.
            m (float): Momentum factor.
            eps (float): Perturbation size for gradient approximation.
            Niter (int): Number of iterations.
            print_steps (bool): Whether to print intermediate steps.

        Returns:
            np.ndarray: Optimized solution.
        """
        change = 0
        for i in range(Niter):
            print(i)
            f_val = fun(x)
            grad_approx = spo.approx_fprime(x, fun, eps)
            grad_step = np.linalg.inv(grad_approx) @ f_val
            
            new_change = r * grad_step + m * change
            x -= new_change
            change = new_change
            if print_steps:
                print(f"Iteration {i}: f={f_val}, g={new_change}")

            print('#'*10)

        return x
    
    # ---- balances over the system boundary --------------------------------------------------------
    def _all_links(self) -> list:
        links = list(getattr(self, '_links', {}))
        for _, sub in self._members(ModelBase):
            if getattr(sub, '_is_system', False):
                links += sub._all_links()
        return links

    def _process_units(self) -> list:
        from .flow import SourceM, Sink
        return [u for _, u in self._units() if not isinstance(u, (SourceM, Sink))]

    def boundary_flanges(self) -> dict:
        """
        Streams that cross the system boundary: ``{'feeds': [...], 'products': [...]}``.

        Every unit inlet that is not connected (directly, or through the ports of a sub-system) to the
        outlet of another unit is a feed; every unit outlet that does not lead to the inlet of another unit
        is a product. Sources and sinks only mark the boundary. Connections are the ones made by
        ``connect_seq`` / ``connect`` and the tear streams of ``Recycle_Tear``, so call ``seq_sol`` first.
        """
        parent = {}

        def find(f):
            parent.setdefault(f, f)
            while parent[f] is not f:
                parent[f] = parent[parent[f]]
                f = parent[f]
            return f

        for a, b in self._all_links():
            ra, rb = find(a), find(b)
            if ra is not rb:
                parent[ra] = rb
        units = self._process_units()
        group_in = {find(f) for u in units for f in u.inlets}
        group_out = {find(f) for u in units for f in u.outlets}
        feeds = [f for u in units for f in u.inlets if find(f) not in group_out]
        products = [f for u in units for f in u.outlets if find(f) not in group_in]
        return {'feeds': feeds, 'products': products}

    def unit_balances(self) -> pd.DataFrame:
        """Mass (g/s) and energy (kJ/s) balance residual of every unit, in - out; zero when it closes."""
        from .flow import SourceM, Sink
        rows = {path: {'mass': u.mass_balance(), 'energy': u.energy_balance(), 'Q+W': u.heat_and_work()}
                for path, u in self._units() if not isinstance(u, (SourceM, Sink))}
        return pd.DataFrame(rows).T

    def MBcheck(self) -> float:
        """Mass flow into the system minus mass flow out of it over the boundary streams, g/s."""
        b = self.boundary_flanges()
        return sum(f.m for f in b['feeds']) - sum(f.m for f in b['products'])

    def EBcheck(self) -> float:
        """
        Energy balance over the system boundary, kJ/s: enthalpy of the feeds + heat and work added to all
        units - enthalpy of the products. Zero when the balance closes; ``unit_balances()`` shows where not.
        The enthalpy flows are kept in ``q_in`` and ``q_out``.
        """
        b = self.boundary_flanges()
        self.q_in = sum(f.n*f.h for f in b['feeds'])
        self.q_out = sum(f.n*f.h for f in b['products'])
        return self.q_in - self.q_out + sum(u.heat_and_work() for u in self._process_units())


import numpy as np
import pandas as pd
import scipy.optimize as spo

from .interfaces import Flange
from .. import Base

from ._mixin.introspection import IntrospectionMixin
from ._mixin.graph import GraphMixin
from ._mixin.analysis import ReportingMixin
from ._mixin.recycle import RecycleMixin

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

    def Eqs(self):
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
            getattr(self, port).set(**state)
        used = []
        for key, value in legacy.items():
            port, _, var = key.partition('_')
            flange = getattr(self, port, None)
            if not isinstance(flange, Flange) or var not in ('n', 'Xi', 'T', 'P'):
                raise TypeError(f"{type(self).__name__}() got an unexpected keyword argument '{key}'")
            if value is None:
                continue
            used.append(key)
            flange.set(**{var: value})
        if used:
            from .._deprecation import warn_deprecated
            port = used[0].partition('_')[0]
            example = ', '.join(f"{k.partition('_')[2]}=..." for k in used if k.partition('_')[0] == port)
            warn_deprecated(f"{type(self).__name__}({', '.join(used)}=...): state keywords are deprecated; set the "
                            f"operating state on the flanges, e.g. unit.{port}.set({example})",
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
        
        for attr in pin1.__dict__.keys(): #for each flange of the other components 
            if any(k in attr for k in self._core):
                continue 

            if attr in ['Q','I']: # Flow variables sum to zero at interface
                Eq = np.append(Eq, [pin1.__dict__[attr] + pin2.__dict__[attr]])
            elif attr in ['n', 'T', 'P']: # add equations describing the equality on the flange
                Eq = np.append(Eq, [ pin1.__dict__[attr] - pin2.__dict__[attr] ])
            
        self.Eq = np.append(self.Eq, Eq)

    def connect_seq(self, pin1, pin2, direction='F', store=False) -> None:
        '''
        forces assignment of one pins conditions to the other

        direction, str = F for forward, B for backward
        '''

        self._check_connection(pin1, pin2)
        
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
    
    def MBcheck(self):
        sources = self.get_Sources().values()
        sinks = self.get_Sinks().values()
        
        m_in = 0
        for i in sources:
            m_in += i.F1.m

        m_out = 0
        for i in sinks:
            m_out += i.F0.m

        return m_out - m_in

    def EBcheck(self):
        sources = self.get_Sources().values()
        sinks = self.get_Sinks().values()
        
        self.q_in = 0
        for i in sources:
            self.q_in += i.F1.n*i.F1.h

        self.q_out = 0
        for i in sinks:
            self.q_out += i.F0.n*i.F0.h

        return (self.tot_heaters_in() + self.tot_heaters_out() + self.tot_heat_in() + self.tot_heat_out() +
                self.tot_Qloss() + self.tot_work_in() + self.tot_work_out() + self.q_in - self.q_out
              )


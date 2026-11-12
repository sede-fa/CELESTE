'''
Interfaces
'''
import copy

import numpy as np
from numpy.typing import NDArray as npArray
import pandas as pd

from . import media
from .. import Base, solvers
from . import exergy as _exergy
from ..functions import carbonFraction

class Flange(Base):
    '''
    Port of a unit model. A flange carries the state of a stream (molar flow n,
    mole fractions Xi, temperature T, pressure P) and owns a private copy of the
    medium that turns this state into properties.

    Parameters:
        name:   name of the port, by convention '<unit name>.<port name>'
        Medium: medium object from ``celeste.CSL.media`` (copied, never shared)
        n:      molar flow in [mol/s]
        Xi:     mole fractions; defaults to the reference composition of the medium
        T:      temperature in [K]
        P:      pressure in [bar]

    Properties:
        h, s, g, cp:  molar enthalpy, entropy, Gibbs energy [kJ/mol, kJ/mol/K]
        mw, m, Yi:    molar mass [g/mol], mass flow [g/s], mass fractions
        Ex:           molar exergy [kJ/mol];  LHV: lower heating value [kJ/mol]

    Setting T, P or Xi updates the properties.
    '''

    _pin_vars = ['n', 'Xi', 'T', 'P']
    _flow_vars = []
    
    def __init__(self, name:str, Medium:media.Medium_Base, n:float=1, Xi:npArray[np.float64]|None=None, T:float=298.15, P:float=1.):
        Medium=copy.copy(Medium) # to avoid weird behaviour when using the same medium variable for more than one flange
        super().__init__(locals())
        self.Medium: media.Medium_Base = Medium
        #self.name = f"{name}"
        
        if Xi is None:
            Xi  = self.Medium._reference_X

        self.Medium.PTX = P, T, np.array(Xi)
        
        #if np.sum(self.Xi)>1.01:
        #    print(f'Warning: {self.name}.Xi has sum greater than 1:  sum({self.name}.Xi) = {np.sum(self.Xi)}')

    @property
    def mw(self):
        '''
        Molar Mass /g/mol
        '''
        return self.Medium.getMW()

    @property
    def cp(self):
        '''
        Specific Heat Capacity /kJ/mol/K
        '''
        return self.Medium.cp
    
    @property
    def h(self):
        '''
        Specific Enthalpy /kJ/mol
        '''
        return self.Medium.h

    @property
    def s(self):
        '''
        Specific Entropy /kJ/mol/K
        '''
        return self.Medium.s

    @property
    def g(self):
        '''
        Specific Gibbs Energy /kJ/mol
        '''
        return self.Medium.g

    @property
    def T(self):
        '''
        Absolute Temperature /K
        '''
        return self.Medium.T
    
    @T.setter
    def T(self, T):
        self.__dict__['T'] = T
        self.Medium.T = T
        
    @property
    def Xi(self) -> npArray[np.float64]:
        '''
        Mole Fractions /-
        '''
        return self.Medium.Xi
    
    @Xi.setter
    def Xi(self, xi):
        if xi is None:
            xi = self.Medium._reference_X
        self.__dict__['Xi'] = xi
        self.Medium.Xi = np.array(xi)
        #if np.sum(self.Xi)>1.01:
        #    print(f'Warning: {self.name}.Xi has sum greater than 1:  sum({self.name}.Xi) = {np.sum(self.Xi)}')
        
    @property
    def P(self):
        '''
        Pressure /bar
        '''
        return self.Medium.P
    
    @P.setter
    def P(self, Ps):
        self.__dict__['P'] = Ps
        self.Medium.P = Ps
    
    @property
    def Yi(self):
        '''
        Mass Fraction /-
        '''
        return self.Medium.Yi

    @Yi.setter
    def Yi(self, y):
        self.Medium.Yi = y
        self.Xi = self.Medium.Xi

    @property
    def m(self):
        '''
        Mass Flow /g/s
        '''
        return self.n*self.mw

    @m.setter
    def m(self, massFlow):
        self.n = massFlow/self.mw

    def set(self, n=None, Xi=None, T=None, P=None):
        '''Set any of n, Xi, T, P in one call; returns the flange (``unit.F0.set(n=10, T=600)``).'''
        if n is not None:
            self.n = n
        if P is not None:
            self.P = P
        if T is not None:
            self.T = T
        if Xi is not None:
            self.Xi = np.asarray(Xi, dtype=float)
        return self

    def get_nXTP(self):
        ''' 
        Returns array of [n, Xi, T, P] as a 1 dimensional vector
        '''
        val_arr = np.concatenate(([self.n, self.m, self.T, self.P, self.h, self.cp], self.Xi))
        head_arr = ['n /mol/s', 'm /g/s', 'T /K', 'P /bar', 'h /kJ/mol', 'cp /kJ/mol/K'] + [f'Xi_{species}' for species in self.Medium.species]

        return pd.Series(dict(zip(head_arr, val_arr)), name = self.name )

    def __repr__(self):
        return self.get_nXTP().to_string()
    
    def incompatibility(self, other) -> str | None:
        """
        Why ``self`` cannot be connected to ``other``, or None if it can.

        Two flanges are compatible when they are the same kind of port (same flange class),
        carry the same kind of medium (same medium class) and list the same species in the
        same order. Use ``components.ChgMedium`` to pass a stream between different media.
        """
        if not isinstance(other, Flange):
            return f"{getattr(other, 'name', other)!r} is not a flange"
        if type(self) is not type(other):
            return f"different port types ({type(self).__name__} / {type(other).__name__})"
        if type(self.Medium) is not type(other.Medium):
            return f"different media ({type(self.Medium).__name__} / {type(other.Medium).__name__})"
        sp1, sp2 = [str(x) for x in self.Medium.species], [str(x) for x in other.Medium.species]
        if sp1 != sp2:
            return f"different species lists ({sp1} / {sp2})"
        return None

    def compatible(self, other) -> bool:
        """True if the two flanges can be connected (see ``incompatibility``)."""
        return self.incompatibility(other) is None

    # Flanges compare by identity (the default), so they can be used in sets and as dictionary keys.

    @property
    def ExergyObj(self):
        '''
        Exergy model of the stream; the class is chosen by the medium (``Medium.ExergyModel``),
        by default ``celeste.CSL.exergy.ThermoExergy``.
        '''
        model = getattr(self.Medium, 'ExergyModel', _exergy.ThermoExergy)
        return model(self.Medium, self.Xi, self.T, self.P)

    @property
    def Ex_ph(self):
        '''
        Specific Physical Exergy /kJ/mol
        '''
        exergy = getattr(self.Medium, 'exergy_parts', None)
        return exergy()[0] if exergy is not None else self.ExergyObj.Ex_ph

    @property
    def Ex_ch(self):
        '''
        Specific Chemical Exergy /kJ/mol
        '''
        exergy = getattr(self.Medium, 'exergy_parts', None)
        return exergy()[1] if exergy is not None else self.ExergyObj.Ex_ch

    @property
    def EX(self):
        '''
        Exergy Flow /kW
        '''
        return self.n*self.Ex
        
    @property
    def Ex(self):
        '''
        Specific Exergy (physical + chemical) /kJ/mol
        '''
        exergy = getattr(self.Medium, 'exergy', None)   # media with their own exergy model (e.g. Medium_PRbase)
        if exergy is not None:
            return exergy()
        return self.ExergyObj.Ex
    
    @property
    def LHV(self):
        '''
        Lower Heating Value /kJ/mol
        '''
        return self.Medium.LHVmix

    def _solve_T(self, residual, T_bounds, name):
        def fun(t):
            self.T = t
            return residual()
        sol = solvers.solve_scalar(fun, T_bounds[0], T_bounds[1], x0=min(max(self.T, T_bounds[0]), T_bounds[1]),
                                   name=f'{self.name}.{name}')
        self.T = float(sol.x[0])
        return sol

    def T_hx(self, h, X, T_bounds=(273.15, 2000.)):
        '''
        Set the composition to X and the temperature such that the molar enthalpy equals h [kJ/mol]
        '''
        self.Xi = X
        return self._solve_T(lambda: self.h - h, T_bounds, 'T_hx')

    def T_sPX(self, s, P, X, T_bounds=(273.15, 2000.)):
        '''
        Set pressure and composition and the temperature such that the molar entropy equals s [kJ/mol/K]
        '''
        self.P = P
        self.Xi = X
        return self._solve_T(lambda: (self.s - s)*1e3, T_bounds, 'T_sPX')

    def getCarbon(self, T_drop=0):
        return carbonFraction(self.T-T_drop, self.P, dict(zip(self.Medium.species, self.Xi)))
    

"""
Interactive process flow diagrams, independent of the modelling framework.

``draw(name, units, streams)`` lays out and draws any flowsheet; every unit and stream describes itself with a
``pfd_view()`` method returning a :class:`UnitView` or :class:`StreamView` (what kind of unit it is and which
rows its hover table shows). CELESTE systems call it through ``SysBase.pfd()``.
"""
from .draw import PFD, Terminal, draw
from .views import KINDS, Row, StreamView, UnitView

__all__ = ['draw', 'PFD', 'Terminal', 'UnitView', 'StreamView', 'Row', 'KINDS']

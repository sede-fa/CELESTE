"""
What the diagram shows of each object. An object describes itself with ``pfd_view()``:

* a unit returns a :class:`UnitView` - its kind (shape and colour) and the rows of its hover table,
* a stream returns a :class:`StreamView` - the rows of its hover table and its composition.

The drawing code only reads these views, so any object (CELESTE units and flanges, jax units and streams, or
a user's own class) can be drawn by giving it a ``pfd_view`` method. An object without one is drawn as a
plain box (unit) or an unlabelled line (stream).
"""
from dataclasses import dataclass, field

#: kind of unit -> legend text; the kind selects the colour of the box
KINDS = {
    'heat': 'heat exchange',
    'reaction': 'reactor',
    'electrochem': 'electrochemical',
    'separation': 'separation',
    'pressure': 'pressure change',
    'flow': 'mixer / splitter',
    'terminal': 'source / sink',
    'system': 'subsystem',
    'other': 'other',
}


@dataclass
class Row:
    """One line of a hover table: ``label`` (with its unit) and ``value`` (number, text or True/False)."""
    label: str
    value: object
    unit: str = ''


@dataclass
class UnitView:
    """How a unit is drawn: ``kind`` (a key of ``KINDS``), the ``rows`` of its table, optional ``subtitle``."""
    kind: str = 'other'
    rows: list = field(default_factory=list)
    subtitle: str | None = None          #: default: the class name

    def add(self, label: str, value, unit: str = '') -> 'UnitView':
        """Append a row (returns the view, so calls can be chained)."""
        self.rows.append(Row(label, value, unit))
        return self


@dataclass
class StreamView:
    """How a stream is shown: the ``rows`` of its table and its ``composition`` [(species, mole fraction)]."""
    rows: list = field(default_factory=list)
    composition: list = field(default_factory=list)
    subtitle: str | None = None          #: for example the medium

    def add(self, label: str, value, unit: str = '') -> 'StreamView':
        """Append a row (returns the view, so calls can be chained)."""
        self.rows.append(Row(label, value, unit))
        return self


def safe(obj, attr):
    """``getattr`` that gives None when the attribute is missing or cannot be evaluated in the current state."""
    try:
        return getattr(obj, attr)
    except Exception:
        return None


def view_of(obj, override=None):
    """The view of ``obj``: ``override(obj)`` if given, else ``obj.pfd_view()``, else None."""
    if override is not None:
        return override(obj)
    method = getattr(obj, 'pfd_view', None)
    if method is None:
        return None
    try:
        return method()
    except Exception:                    # a unit that has not been solved yet may fail to evaluate
        return None

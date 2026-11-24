"""
Process flow diagram of a system, drawn by :mod:`celeste.pfd` from the declared streams (or the connections of
the last run). What each unit and stream shows comes from its ``pfd_view()``.

    pfd = plant.pfd()          # after run_sol(), or for a flowsheet declared with link()
    pfd.save('plant.html')     # in Jupyter the object displays itself
"""
from ...pfd import PFD, Terminal, draw


def _short(name: str) -> str:
    """'plant.mix.F2' -> 'mix.F2' (drop the system prefix, and the decorations of sources and sinks)."""
    parts = name.replace('((', '.').replace('[(', '.').split('.')
    parts = [p.strip('()[]') for p in parts if p.strip('()[]')]
    return '.'.join(parts[-2:]) if len(parts) > 1 else name


class PFDMixin:
    """``pfd()``: interactive process flow diagram of the system."""

    def _drawn_streams(self) -> list:
        if self.streams:
            return self.streams
        links = self._run_streams()
        if not links:
            raise ValueError(f"{self.name}: nothing to draw - run the system once, or declare its streams with link()")
        return links

    def pfd(self, path: str | None = None, views: dict | None = None) -> PFD:
        """
        Draw the flowsheet (declared streams, or the connections made by the last run). ``path`` also writes
        the HTML file. ``views`` = {unit name or class: function(unit) -> UnitView} replaces the ``pfd_view()``
        of those units in this drawing. Returns a :class:`celeste.pfd.PFD`.
        """
        owners = self._owner_map()
        names = {id(v): k for k, v in self.__dict__.items() if hasattr(v, 'get_Flanges') and v is not self}
        units = {names.get(id(u), _short(u.name)): u for u in self.getObjs()}
        label = {id(u): k for k, u in units.items()}
        streams = []
        for outlet, inlet in self._drawn_streams():
            src, dst = owners.get(id(outlet)), owners.get(id(inlet))
            if src is None or dst is None:
                continue
            src = Terminal(_short(outlet.name), 'system feed') if src is self else label[id(src)]
            dst = Terminal(_short(inlet.name), 'system product') if dst is self else label[id(dst)]
            streams.append((src, dst, outlet, f'{_short(outlet.name)} → {_short(inlet.name)}'))
        return draw(self.name, units, streams, views=views, path=path)

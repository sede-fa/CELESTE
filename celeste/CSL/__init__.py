'''
CELESTE Standard Library

    components   compressors, pumps, valves, black-box separators (GPU), ASU, DAC
    electrochem  solid oxide cell (SOC) and alkaline electrolyser (AEL)
    flow         sources, sinks, mixers, splitters
    heat         heaters and heat exchangers
    interfaces   Flange: the port that connects units and carries the stream state
    media        fluid definitions (species sets + property model)
    reactors     equilibrium, yield and kinetic reactors
    separation   vapour-liquid flash drum
    thermo       property objects behind the media
    systems      ready-made subsystems ("boxes") and example systems
'''
import importlib

from ._base import ModelBase, SysBase

__all__ = ["ModelBase", "SysBase", "components", "electrochem", "flow", "heat", "interfaces", "media",
           "reactors", "separation", "thermo", "systems"]

_SUBMODULES = frozenset(__all__[2:])


def __getattr__(name):
    # submodules are imported on first access so that ``CSL.flow`` works after ``from celeste import CSL``
    if name in _SUBMODULES:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

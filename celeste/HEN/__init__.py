"""
Heat integration.

    PinchAnalyser    energy targets, composite curves, grand composite curve
    HENSynthesiser   heat exchanger network synthesis (matches, duties, areas, cost)

Both take the stream table produced by ``SysBase.pinch_stream_table()``.
"""
import importlib

__all__ = ["PinchAnalyser", "HENSynthesiser", "HENCosts", "Utility"]

_HOME = {"PinchAnalyser": "pinch", "HENSynthesiser": "synthesis", "HENCosts": "synthesis", "Utility": "synthesis"}


def __getattr__(name):
    if name in _HOME:
        return getattr(importlib.import_module(f"{__name__}.{_HOME[name]}"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

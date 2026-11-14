"""Helpers to retire names without breaking code that still uses them."""
import functools
import warnings

__all__ = ["CelesteDeprecationWarning", "deprecated", "warn_deprecated"]


class CelesteDeprecationWarning(DeprecationWarning):
    """Raised (as a warning) when a deprecated CELESTE name is used."""


def warn_deprecated(message: str, stacklevel: int = 3) -> None:
    warnings.warn(message, CelesteDeprecationWarning, stacklevel=stacklevel)


def deprecated(reason: str, removal: str | None = "a future release"):
    """
    Mark a function or method as deprecated.

    The wrapped callable keeps working and emits a
    :class:`CelesteDeprecationWarning` pointing at the caller.

    Parameters
    ----------
    reason  : what to use instead / why it is going away
    removal : when it will be removed; ``None`` for "kept as a no-op shim"
    """
    def decorate(fn):
        when = f" and will be removed in {removal}" if removal else ""
        msg = f"{fn.__qualname__} is deprecated{when}: {reason}"

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            warn_deprecated(msg)
            return fn(*args, **kwargs)

        wrapper.__deprecated__ = msg
        return wrapper
    return decorate

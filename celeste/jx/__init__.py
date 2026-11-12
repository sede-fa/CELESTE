"""
celeste.jx - numerical kernels shared by the whole framework, JAX ready.

===========  ==================================================================
module       content
===========  ==================================================================
``nasa``     NASA-9 ideal-gas properties (one kernel for NumPy and JAX)
``ideal``    jit/vmap/grad-ready ideal-gas mixture properties
``eos``      Peng-Robinson equation of state: Z, fugacities, departure functions
``flash``    Rachford-Rice, tangent-plane stability test and isothermal flash
``fluid``    real-fluid stream state (ideal gas + departure, phases from the flash)
``units``    Equinox process units (prototype of the next model generation)
===========  ==================================================================

``nasa`` only needs NumPy. Everything else imports JAX on first use and switches
it to double precision (``jax_enable_x64``), which thermodynamic calculations
require.
"""
import importlib

__all__ = ["nasa", "ideal", "eos", "flash", "fluid", "units", "enable_x64"]

_LAZY = ("ideal", "eos", "flash", "fluid", "units")


def enable_x64() -> None:
    """Switch JAX to 64-bit floats (idempotent). Called by every JAX module of celeste."""
    import jax
    jax.config.update("jax_enable_x64", True)


def __getattr__(name):
    if name in _LAZY or name == "nasa":
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

"""
Turbomachinery: compressors, turbines and their performance maps.

Compressors:  Compressor (base), CompIsot, CompIsent, CompPolytropic, CompressorMapped
Turbines:     Turbine (base), TurbineMapped
Maps:         CompressorMap, TurbineMap (calibration, plotting, Dymola export)

Typical use::

    from celeste.CSL.turbomachinery import CompressorMap, CompressorMapped

    cmap = CompressorMap(...)
    cmap.calibrate_design(...)
    comp = CompressorMapped('comp', cmap, Medium=media.Air(), use_map=True)
    cmap.export_dymola('out/compressor')
"""
from .components import (Compressor, CompIsot, CompIsent, CompPolytropic, CompressorMapped,
                         Turbine, TurbineMapped)
from .corrections import (Wcorr_from_mf, delta, mf_from_Wcorr, rel_error,
                          reynolds_number_index, sutherland_mu, theta)
from .dymola_io import export_1d_table, export_2d_map, export_all_maps
from .maps import CompressorMap, CompressorState, TurbineMap, TurbineState

__all__ = [
    "CompressorMap", "CompressorState", "TurbineMap", "TurbineState",
    "Compressor", "CompIsot", "CompIsent", "CompPolytropic", "CompressorMapped",
    "Turbine", "TurbineMapped",
    "theta", "delta", "Wcorr_from_mf", "mf_from_Wcorr",
    "sutherland_mu", "reynolds_number_index", "rel_error",
    "export_2d_map", "export_1d_table", "export_all_maps",
]
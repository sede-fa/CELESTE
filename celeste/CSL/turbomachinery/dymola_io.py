"""
Low-level writers for Dymola/Modelica CombiTable files.

The maps export themselves through ``CompressorMap.export_dymola`` and
``TurbineMap.export_dymola``; this module only knows the file format.
"""
import os

import numpy as np


def export_2d_map(filename, table_name, value_map, x_axis, y_axis):
    """
    Write a 2D table in Dymola's CombiTable2D text format.

    Parameters
    ----------
    filename : str
        Output file path.
    table_name : str
        Matrix name inside the file.
    value_map : array_like
        Values with shape ``(len(x_axis), len(y_axis))``. NaN is written as 0.
    x_axis : array_like
        First axis (rows), e.g. ``ncorr_map``.
    y_axis : array_like
        Second axis (columns), e.g. ``beta_map``.
    """
    x_axis = np.asarray(x_axis, dtype=float)
    y_axis = np.asarray(y_axis, dtype=float)
    value_map = np.nan_to_num(np.asarray(value_map, dtype=float), nan=0.0)

    if value_map.shape != (len(x_axis), len(y_axis)):
        raise ValueError(
            f"Shape mismatch: value_map shape is {value_map.shape}, "
            f"expected ({len(x_axis)}, {len(y_axis)})"
        )

    with open(filename, 'w', newline='') as f:
        f.write('#1\n')
        f.write(f'double {table_name}({len(x_axis) + 1},{len(y_axis) + 1})\n')

        # Header row: dummy zero followed by the y-axis values
        f.write(", ".join(map(str, [0.0] + list(y_axis))) + "\n")

        for i in range(len(x_axis)):
            f.write(", ".join(map(str, [x_axis[i]] + list(value_map[i]))) + "\n")


def export_1d_table(filename, table_name, x_values, y_values):
    """
    Write a two-column table in Dymola's CombiTable1D text format.

    Used for the compressor surge line (Wc, PR) and the turbine
    min/max PR lines (Nc, PRmin / PRmax).
    """
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)

    if len(x_values) != len(y_values):
        raise ValueError("x_values and y_values must have the same length.")

    with open(filename, mode='w', newline='') as f:
        f.write("#1\n")
        f.write(f"double {table_name}({len(x_values)},2)\n")
        for x, y in zip(x_values, y_values):
            f.write(f"{x}, {y}\n")


def export_all_maps(comp_map, hpt_map, lpt_map, comp_dir, hpt_dir, lpt_dir,
                    suffix="SOFC_LPTcycle"):
    """Export a compressor map and two turbine maps to their own folders."""
    comp_map.export_dymola(comp_dir, suffix)
    hpt_map.export_dymola(hpt_dir, suffix)
    lpt_map.export_dymola(lpt_dir, suffix)

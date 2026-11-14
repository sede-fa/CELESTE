"""
From a system model to a heat exchanger network.

A methanol synthesis loop is solved, its heaters and coolers are turned into a
stream table, the pinch analysis gives the energy targets and the network
synthesis proposes the heat exchangers.

Run with:  python examples/02_system_pinch_hens.py
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

import pandas as pd

from celeste.CSL.systems.ProcessBoxes import MethanolLoop
from celeste.HEN import HENSynthesiser, PinchAnalyser

FIG_DIR = Path(__file__).parent / 'figures'
FIG_DIR.mkdir(exist_ok=True)

pd.set_option('display.width', 200)

# --- 1. solve the system ------------------------------------------------------------------------------
loop = MethanolLoop('loop', n_feed=100.0, T_reac=523.15, P=50.0, T_cond=313.15, purge=0.02)
passes, error = loop.run_sol()
print(f'recycle converged in {passes} passes (error {error:.1e})')
print(f'recycle ratio {loop.recycle_ratio:.2f}, carbon efficiency {loop.carbon_efficiency:.3f}')
print(f'crude methanol: {loop.product.F0.n:.2f} mol/s, x_MeOH = {loop.product.F0.Xi[6]:.3f}\n')
print(loop.exdest_df().round(2), '\n')

# --- 2. stream table -> pinch analysis ----------------------------------------------------------------
streams = loop.pinch_stream_table()
print(streams.round(3), '\n')

pinch = PinchAnalyser(streams, dt_min=10).solve()
print(pinch.summary(), '\n')
fig, _ = pinch.plot_summary()
fig.savefig(FIG_DIR / 'pinch.png', dpi=120)

# --- 3. heat exchanger network ------------------------------------------------------------------------
hen = HENSynthesiser(streams, dt_min=10).solve()
print(hen.summary(), '\n')
print('process-process exchangers\n', hen.network.round(2), '\n')
print('utility exchangers\n', hen.utilities.round(2), '\n')
print(hen.check())
fig, _ = hen.plot_grid_diagram()
fig.savefig(FIG_DIR / 'network.png', dpi=120)
print(f'\nfigures written to {FIG_DIR}')

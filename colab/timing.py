"""Time one reservoir step for a given configuration (run on the GPU VM).

    python colab/timing.py --n-cells 2 2 --cell-xy 5e-9 [--fmr-duration 2e-9]
"""
import argparse, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asvi_rc import ASVIParams, ProtocolParams, tasks
from asvi_rc.mumaxplus_driver import ASVISimulation

ap = argparse.ArgumentParser()
ap.add_argument("--n-cells", type=int, nargs=2, default=(2, 2))
ap.add_argument("--cell-xy", type=float, default=5e-9)
ap.add_argument("--cell-z", type=float, default=5e-9)
ap.add_argument("--fmr-duration", type=float, default=2e-9)
ap.add_argument("--loop-step", type=float, default=1e-3)
a = ap.parse_args()
p = ASVIParams(n_cells=tuple(a.n_cells), cell_xy=a.cell_xy, cell_z=a.cell_z, fmr_duration=a.fmr_duration)
proto = ProtocolParams(loop_step=a.loop_step)
nx, ny, nz = p.grid
print(f"grid {nx}x{ny}x{nz} = {nx*ny*nz/1e6:.2f} Mcells, {p.n_islands} islands")
t0 = time.time(); sim = ASVISimulation(p, verbose=False); print(f"setup {time.time()-t0:.1f} s")
t0 = time.time(); sim.saturate(proto); print(f"saturate {time.time()-t0:.1f} s")
ramp = tasks.minor_loop(30e-3, proto, b_start=28e-3)[:5]
t0 = time.time(); sim.apply_ramp(ramp, proto); t_min = (time.time()-t0)/len(ramp)
print(f"minimisation {t_min:.2f} s each")
t0 = time.time(); freqs, P = sim.fmr(tasks.field_vector(30e-3, proto)); t_fmr = time.time()-t0
per_ns = t_fmr / (a.fmr_duration*1e9)
n_loop = len(tasks.minor_loop(32e-3, proto, b_start=32e-3))
for dur in (13, 26):
    step = n_loop*t_min + per_ns*dur
    print(f"FMR {dur} ns: {per_ns*dur:6.0f} s; full step ({n_loop} minimisations) ~{step/60:.1f} min; 100 steps ~{step*100/3600:.1f} h")

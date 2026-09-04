"""Run on the Colab VM with ``colab exec -s <name> -f colab/remote_setup.py``.

Expects ``/content/spinice.tar.gz`` (uploaded with ``colab upload``).  Installs
mumax+, unpacks the repo to /content/SpinIce and runs a small API smoke test on
the GPU (1x1 unit cell, 10 nm cells, 2 ns FMR) so that any mumax+ API mismatch
shows up before the long runs.
"""
import os
import subprocess
import sys
import tarfile
import time

print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"],
                     capture_output=True, text=True).stdout)

try:
    import mumaxplus  # noqa: F401
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "mumaxplus"], check=True)
    import mumaxplus  # noqa: F401
print("mumaxplus", getattr(mumaxplus, "__version__", "?"))

os.makedirs("/content/SpinIce", exist_ok=True)
with tarfile.open("/content/spinice.tar.gz") as tf:
    tf.extractall("/content/SpinIce")
os.chdir("/content/SpinIce")
sys.path.insert(0, "/content/SpinIce")
for m in [m for m in list(sys.modules) if m.startswith("asvi_rc")]:
    del sys.modules[m]

import numpy as np
from asvi_rc import ASVIParams, ProtocolParams, tasks
from asvi_rc.geometry import describe
from asvi_rc.mumaxplus_driver import ASVISimulation

p = ASVIParams(n_cells=(1, 1), cell_xy=10e-9, cell_z=5e-9, fmr_duration=2e-9)
proto = ProtocolParams(loop_step=5e-3)
sim = ASVISimulation(p)
print(describe(p, sim.islands))
t0 = time.time()
sim.saturate(proto)
print(f"saturate: {time.time() - t0:.1f} s; <m> per region:", np.round(sim.region_averages(), 3).tolist())
t0 = time.time()
sched = tasks.field_schedule(np.array([0.5]), proto)
sim.apply_ramp(sched[0]["ramp"], proto)
print(f"ramp of {len(sched[0]['ramp'])} minimisations: {time.time() - t0:.1f} s "
      f"({(time.time() - t0) / len(sched[0]['ramp']):.2f} s each)")
t0 = time.time()
freqs, P = sim.fmr(tasks.field_vector(sched[0]["b_meas"], proto))
dt = time.time() - t0
print(f"FMR {p.fmr_duration*1e9:g} ns: {dt:.1f} s -> {dt / p.fmr_duration * 26e-9:.0f} s per 26 ns run at this size")
print("spectrum shape", P.shape, "peak", freqs[P.sum(axis=1)[1:].argmax() + 1] / 1e9, "GHz")
print("SMOKE TEST OK")

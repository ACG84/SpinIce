#!/usr/bin/env python3
"""Reservoir-computing data collection with mumax+ (GPU required).

For every input value: quasi-static minor field loop -> measurement field ->
broadband FMR (sinc along z) -> region-averaged m(t) -> power spectrum.
The spectra of all steps are stored in one .npz that scripts/train_readout.py
consumes.  The run is resumable: pass --resume to continue from the last
saved step (the static magnetisation is checkpointed).

Example
-------
    python mumaxplus/run_reservoir.py --task mackey_glass_10 --n 400 \
        --out runs/mg10 --n-cells 2 2 --b-min 20e-3 --b-max 30e-3
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asvi_rc import ASVIParams, ProtocolParams, tasks          # noqa: E402
from asvi_rc.geometry import describe                           # noqa: E402
from asvi_rc.mumaxplus_driver import ASVISimulation             # noqa: E402
from asvi_rc.spectra import save_spectra                        # noqa: E402
from asvi_rc.tasks import field_schedule, field_vector          # noqa: E402


def add_common_args(ap: argparse.ArgumentParser):
    ap.add_argument("--n-cells", type=int, nargs=2, default=None, help="supercell in unit cells")
    ap.add_argument("--cell-xy", type=float, default=None)
    ap.add_argument("--cell-z", type=float, default=None)
    ap.add_argument("--disorder", type=float, default=None, help="relative std of island size")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--fmr-duration", type=float, default=None)
    ap.add_argument("--params", type=str, default=None, help="JSON file of ASVIParams")


def params_from_args(a) -> ASVIParams:
    p = ASVIParams.load(a.params) if a.params else ASVIParams()
    if a.n_cells: p.n_cells = tuple(a.n_cells)
    if a.cell_xy: p.cell_xy = a.cell_xy
    if a.cell_z: p.cell_z = a.cell_z
    if a.disorder is not None: p.disorder_sigma = a.disorder
    if a.seed is not None: p.seed = a.seed
    if a.alpha is not None: p.alpha = a.alpha
    if a.fmr_duration: p.fmr_duration = a.fmr_duration
    return p


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="mackey_glass_10", choices=list(tasks.TASKS))
    ap.add_argument("--n", type=int, default=400, help="number of input steps")
    ap.add_argument("--out", type=str, default="runs/reservoir")
    ap.add_argument("--b-min", type=float, default=None)
    ap.add_argument("--b-max", type=float, default=None)
    ap.add_argument("--loop-shape", default=None, choices=["bipolar", "unipolar", "return", "alternating"])
    ap.add_argument("--loop-angle", type=float, default=None)
    ap.add_argument("--loop-step", type=float, default=None)
    ap.add_argument("--coarse-step", type=float, default=None, help="larger increment used while |B| < --coarse-below")
    ap.add_argument("--coarse-below", type=float, default=None)
    ap.add_argument("--measure-at", default=None, choices=["loop_max", "bias"])
    ap.add_argument("--robust", action="store_true", help="use relax() instead of minimize() in loops")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--save-every", type=int, default=10)
    add_common_args(ap)
    a = ap.parse_args(argv)

    p = params_from_args(a)
    proto = ProtocolParams()
    if a.b_min is not None: proto.b_min = a.b_min
    if a.b_max is not None: proto.b_max = a.b_max
    if a.loop_shape: proto.loop_shape = a.loop_shape
    if a.loop_angle is not None: proto.loop_angle_deg = a.loop_angle
    if a.loop_step: proto.loop_step = a.loop_step
    if a.coarse_step: proto.coarse_step = a.coarse_step
    if a.coarse_below is not None: proto.coarse_below = a.coarse_below
    if a.measure_at: proto.measure_at = a.measure_at

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    u, y = tasks.make_task(a.task, a.n)
    sched = field_schedule(u, proto)

    sim = ASVISimulation(p)
    print(describe(p, sim.islands))
    p.save(out / "params.json")
    (out / "protocol.json").write_text(json.dumps(proto.to_dict(), indent=2))
    np.savez(out / "dataset.npz", u=u, y=y, task=a.task)

    start = 0
    power, b_loop = [], []
    ckpt, partial = out / "state.npy", out / "spectra_partial.npz"
    if a.resume and ckpt.exists() and partial.exists():
        with np.load(partial) as z:
            freqs, power, b_loop = z["freqs"], list(z["power"]), list(z["b_loop"])
        start = len(power)
        sim.load_state(ckpt)
        sim.set_field(field_vector(sched[start - 1]["b_meas"], proto))
        print(f"resuming at step {start}")
    else:
        sim.saturate(proto)

    t_start = time.time()
    for k in range(start, len(sched)):
        s = sched[k]
        sim.apply_ramp(s["ramp"], proto, robust=a.robust)
        freqs, P = sim.fmr(field_vector(s["b_meas"], proto))
        power.append(P)
        b_loop.append(s["b_loop"])
        el = time.time() - t_start
        print(f"step {k + 1}/{len(sched)}  B_loop {s['b_loop']*1e3:6.2f} mT  "
              f"peak {freqs[np.argmax(P.sum(axis=1)[1:]) + 1]/1e9:5.2f} GHz  "
              f"{el/(k + 1 - start):5.1f} s/step", flush=True)
        if (k + 1) % a.save_every == 0 or k + 1 == len(sched):
            save_spectra(partial, freqs, np.stack(power), np.array(b_loop), u=u[:k + 1], y=y[:k + 1])
            sim.save_state(ckpt)

    save_spectra(out / "spectra.npz", freqs, np.stack(power), np.array(b_loop), u=u, y=y,
                 region_ids=np.array(sim.region_ids), task=a.task)
    print("wrote", out / "spectra.npz")


if __name__ == "__main__":
    main()

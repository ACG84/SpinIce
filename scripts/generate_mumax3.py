#!/usr/bin/env python3
"""Generate mumax3 scripts for the multilayered ASVI.

Modes
-----
reservoir : unrolled reservoir-computing protocol for a benchmark task
sweep     : FMR-vs-field map + hysteresis (calibration of the input range)

Examples
--------
    python scripts/generate_mumax3.py reservoir --task mackey_glass_10 --n 400 --out mumax3/out/mg10
    python scripts/generate_mumax3.py sweep --b-start -30e-3 --b-stop 60e-3 --out mumax3/out/sweep
then
    mumax3 mumax3/out/mg10/asvi_reservoir.mx3
    python scripts/process_mumax3_table.py mumax3/out/mg10/asvi_reservoir.out/table.txt \
        mumax3/out/mg10/asvi_reservoir.meta.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asvi_rc import ASVIParams, ProtocolParams, tasks                                   # noqa: E402
from asvi_rc.geometry import build_islands, describe                                    # noqa: E402
from asvi_rc.mumax3_writer import write_reservoir_script, write_field_sweep_script      # noqa: E402


def _allow_negative_sci(parser: argparse.ArgumentParser):
    """Let argparse accept values like -30e-3 (default matcher only handles -30 / -0.03)."""
    import re
    parser._negative_number_matcher = re.compile(r"^-(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$")
    for act in parser._actions:
        if isinstance(act, argparse._SubParsersAction):
            for sp in act.choices.values():
                _allow_negative_sci(sp)


def common(ap):
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--n-cells", type=int, nargs=2, default=None)
    ap.add_argument("--cell-xy", type=float, default=None)
    ap.add_argument("--cell-z", type=float, default=None)
    ap.add_argument("--disorder", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--params", type=str, default=None, help="JSON file of ASVIParams")


def params_from(a):
    p = ASVIParams.load(a.params) if a.params else ASVIParams()
    if a.n_cells: p.n_cells = tuple(a.n_cells)
    if a.cell_xy: p.cell_xy = a.cell_xy
    if a.cell_z: p.cell_z = a.cell_z
    if a.disorder is not None: p.disorder_sigma = a.disorder
    if a.seed is not None: p.seed = a.seed
    if a.alpha is not None: p.alpha = a.alpha
    return p


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    r = sub.add_parser("reservoir")
    r.add_argument("--task", default="mackey_glass_10", choices=list(tasks.TASKS))
    r.add_argument("--n", type=int, default=400)
    r.add_argument("--b-min", type=float, default=None)
    r.add_argument("--b-max", type=float, default=None)
    r.add_argument("--loop-shape", default=None, choices=["bipolar", "unipolar", "return"])
    r.add_argument("--loop-angle", type=float, default=None)
    r.add_argument("--loop-step", type=float, default=None)
    r.add_argument("--coarse-step", type=float, default=None, help="larger increment used while |B| < --coarse-below")
    r.add_argument("--coarse-below", type=float, default=None)
    r.add_argument("--measure-at", default=None, choices=["loop_max", "bias"])
    r.add_argument("--minimizer", default="Minimize", choices=["Minimize", "Relax"])
    r.add_argument("--save-states", action="store_true", help="SaveAs(m) after every step")
    common(r)
    s = sub.add_parser("sweep")
    s.add_argument("--b-start", type=float, default=-30e-3)
    s.add_argument("--b-stop", type=float, default=60e-3)
    s.add_argument("--b-step", type=float, default=1e-3)
    s.add_argument("--angle", type=float, default=1.0)
    s.add_argument("--presaturate", type=float, default=-0.2)
    common(s)
    _allow_negative_sci(ap)
    a = ap.parse_args(argv)

    p = params_from(a)
    islands = build_islands(p)
    print(describe(p, islands))
    out = Path(a.out)
    if a.mode == "reservoir":
        proto = ProtocolParams()
        if a.b_min is not None: proto.b_min = a.b_min
        if a.b_max is not None: proto.b_max = a.b_max
        if a.loop_shape: proto.loop_shape = a.loop_shape
        if a.loop_angle is not None: proto.loop_angle_deg = a.loop_angle
        if a.loop_step: proto.loop_step = a.loop_step
    if a.coarse_step: proto.coarse_step = a.coarse_step
    if a.coarse_below is not None: proto.coarse_below = a.coarse_below
        if a.measure_at: proto.measure_at = a.measure_at
        u, y = tasks.make_task(a.task, a.n)
        script = write_reservoir_script(p, proto, u, out, islands, minimizer=a.minimizer,
                                        save_states=a.save_states)
        np.savez(out / "dataset.npz", u=u, y=y, task=a.task)
        n_min = sum(len(tasks.minor_loop(b, proto)) for b in tasks.input_to_field(u, proto))
        print(f"wrote {script}: {len(u)} reservoir steps, ~{n_min} energy minimisations, "
              f"{len(u)} x {p.fmr_duration*1e9:g} ns dynamic runs")
    else:
        script = write_field_sweep_script(p, a.b_start, a.b_stop, a.b_step, a.angle, out, islands,
                                          presaturate=a.presaturate)
        print(f"wrote {script}")


if __name__ == "__main__":
    main()

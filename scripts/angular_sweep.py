#!/usr/bin/env python
"""Switching astroid of one multilayer island: quasi-static field sweeps at several angles from the island
axis (CPU, magnum.np by default).  For each angle the field is ramped from -B_max to +B_max along the
direction (cos, sin) and the per-layer switching fields (jumps of the axis projection) are recorded, which
maps the (h_par, h_perp) astroid of every layer - the quantity that decides how much a neighbour's
perpendicular vertex field lowers the switching threshold.

    python scripts/angular_sweep.py --angles 45 60 70 80 --bmax 80e-3 --step 2e-3 --out runs/astroid_2layer
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asvi_rc import ASVIParams                                                  # noqa: E402
from asvi_rc.geometry import single_island, macrospin_magnetization            # noqa: E402
from asvi_rc.backend import make_simulation, BACKENDS                           # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--angles", type=float, nargs="+", default=[45, 60, 70, 80])
    ap.add_argument("--bmax", type=float, default=80e-3); ap.add_argument("--step", type=float, default=2e-3)
    ap.add_argument("--cell-xy", type=float, default=10e-9); ap.add_argument("--cell-z", type=float, default=5e-9)
    ap.add_argument("--t-top", type=float, default=None); ap.add_argument("--t-bottom", type=float, default=None)
    ap.add_argument("--width", type=float, default=None); ap.add_argument("--length", type=float, default=None)
    ap.add_argument("--extra-layers", type=float, nargs="+", default=None)
    ap.add_argument("--backend", choices=BACKENDS, default="magnumnp")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    p = ASVIParams(cell_xy=a.cell_xy, cell_z=a.cell_z)
    for attr, val in (("width", a.width), ("length", a.length), ("t_top", a.t_top), ("t_bottom", a.t_bottom)):
        if val is not None:
            setattr(p, attr, val)
    if a.extra_layers:
        p.extra_layers = tuple(zip(a.extra_layers[0::2], a.extra_layers[1::2]))
    p.box_override = (640e-9, 320e-9); p.pbc_repetitions = (0, 0, 0)
    islands = single_island(p)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    sim = make_simulation(p, islands, a.backend, verbose=False)
    fields = np.arange(-a.bmax, a.bmax + a.step / 2, a.step)
    results = {"params": p.to_dict(), "angles_deg": a.angles, "fields_T": fields.tolist(), "layers": {}}
    for ang in a.angles:
        th = np.radians(ang); d = np.array([np.cos(th), np.sin(th), 0.0])
        t0 = time.time()
        sim.set_magnetization(macrospin_magnetization(p, sim.regions, sim.islands, -1.0))
        sim.set_field(tuple(-0.3 * d)); sim.minimize()
        ms = []
        for b in fields:
            sim.set_field(tuple(b * d)); sim.minimize()
            ms.append(sim.region_averages())
        ms = np.stack(ms)                                                        # (nB, n_regions, 3)
        proj = ms[..., 0]                                                        # axis projection (island along x)
        line = []
        for j, isl in enumerate(sim.islands):
            dm = np.diff(proj[:, j])
            ev = [(float(fields[i + 1]), float(dm[i])) for i in np.where(np.abs(dm) > 0.3)[0]]
            results["layers"].setdefault(str(isl.layer), {})[str(ang)] = {"events": ev, "m_axis": proj[:, j].tolist()}
            line.append(f"layer {isl.layer}: " + ", ".join(f"{b*1e3:+.0f} mT (dm {x:+.2f})" for b, x in ev))
        print(f"angle {ang:5.1f} deg ({time.time() - t0:.0f} s): " + " | ".join(line), flush=True)
        (out / "astroid.json").write_text(json.dumps(results, indent=1, default=str))
        np.savez_compressed(out / f"sweep_{ang:g}.npz", fields=fields, m_static=ms, angle_deg=ang)
    print("wrote", out / "astroid.json")


if __name__ == "__main__":
    main()

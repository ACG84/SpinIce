#!/usr/bin/env python3
"""Catalogue the metastable states of a minimal unit and their energies (mumax+, GPU).

For every layer-island the nominal states are '+', '-' (macrospin along +-axis),
'V+', 'V-' (vortex, either circulation).  All combinations are seeded, relaxed,
classified again (a nominal state may collapse into another), and the total
energy recorded.  Run it at several cell sizes to see which textures survive
discretisation and how their energies converge.

    # one isolated 3D island (16 nominal states), cell-size scan
    python mumaxplus/state_catalogue.py --unit single --cell-xy 10e-9 --out runs/cat_single_10nm
    # the periodic 2-island unit cell (256 nominal states)
    python mumaxplus/state_catalogue.py --unit cell --cell-xy 10e-9 --out runs/cat_cell_10nm
Options: --bias BX BY (T) to catalogue in an applied field; --hysteresis to add a
+-60 mT sweep along x (single island) for the switching fields at this cell size;
--fast uses minimize() instead of relax().
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asvi_rc import ASVIParams                                                  # noqa: E402
from asvi_rc.geometry import (build_islands, single_island, macrospin_magnetization,   # noqa: E402
                              vortex_magnetization, classify_layer, describe)
from asvi_rc.mumaxplus_driver import ASVISimulation                             # noqa: E402

STATES = ("+", "-", "V+", "V-")


def seed_state(p, sim, labels):
    signs = {isl.region: (1.0 if lab == "+" else -1.0) for isl, lab in zip(sim.islands, labels) if lab in "+-"}
    m = macrospin_magnetization(p, sim.regions, sim.islands, signs or 1.0, tilt=0.02)
    chir = {isl.region: (1 if lab == "V+" else -1) for isl, lab in zip(sim.islands, labels) if lab.startswith("V")}
    if chir:
        m = vortex_magnetization(p, sim.regions, sim.islands, chirality=chir, base=m)
    sim.set_magnetization(m)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--unit", choices=["single", "cell"], default="single")
    ap.add_argument("--cell-xy", type=float, default=10e-9)
    ap.add_argument("--cell-z", type=float, default=5e-9)
    ap.add_argument("--box", type=float, nargs=2, default=(640e-9, 320e-9), help="single-island box (m)")
    ap.add_argument("--disorder", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bias", type=float, nargs=2, default=(0.0, 0.0), help="applied field Bx By (T)")
    ap.add_argument("--fast", action="store_true", help="minimize() instead of relax()")
    ap.add_argument("--hysteresis", action="store_true")
    ap.add_argument("--max-states", type=int, default=None)
    ap.add_argument("--out", type=str, required=True)
    a = ap.parse_args(argv)

    p = ASVIParams(cell_xy=a.cell_xy, cell_z=a.cell_z, disorder_sigma=a.disorder, seed=a.seed)
    if a.unit == "single":
        p.box_override = tuple(a.box)
        p.pbc_repetitions = (0, 0, 0)
        islands = single_island(p)
    else:
        p.n_cells = (1, 1)
        islands = build_islands(p)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    sim = ASVISimulation(p, islands=islands, verbose=False)
    print(describe(p, sim.islands), flush=True)
    sim.set_field((a.bias[0], a.bias[1], 0.0))

    combos = list(itertools.product(STATES, repeat=len(sim.islands)))
    if a.max_states:
        combos = combos[:a.max_states]
    rows = []
    t0 = time.time()
    for i, labels in enumerate(combos):
        seed_state(p, sim, labels)
        sim.minimize(robust=not a.fast)
        m = sim.get_magnetization()
        final = tuple(classify_layer(p, m, sim.regions, isl) for isl in sim.islands)
        e = sim.total_energy()
        mavg = sim.region_averages(m)
        rows.append({"initial": labels, "final": final, "energy_J": e, "stable": final == labels,
                     "m_axis": [float(mv[0] * isl.axis[0] + mv[1] * isl.axis[1]) for mv, isl in zip(mavg, sim.islands)]})
        print(f"[{i + 1:3d}/{len(combos)}] {'/'.join(labels):>14s} -> {'/'.join(final):>14s}  "
              f"E = {e * 1e18:9.3f} aJ  {'stable' if final == labels else 'collapsed'}  ({time.time() - t0:5.0f} s)", flush=True)

    # summary: distinct final states and their energies
    energies = {}
    for r in rows:
        energies.setdefault(r["final"], []).append(r["energy_J"])
    e_min = min(min(v) for v in energies.values())
    summary = sorted(((np.mean(v), k, len(v), np.std(v)) for k, v in energies.items()))
    print(f"\n{len(energies)} distinct relaxed states from {len(rows)} seeds (cell {a.cell_xy * 1e9:g} nm, "
          f"bias {a.bias[0] * 1e3:g},{a.bias[1] * 1e3:g} mT):")
    print("   E - E_min (aJ)   state (layer labels)          reached from N seeds   std (aJ)")
    for em, k, nn, sd in summary:
        print(f"   {(em - e_min) * 1e18:12.3f}    {'/'.join(k):>24s}   {nn:3d}   {sd * 1e18:.3f}")
    stable = [r for r in rows if r["stable"]]
    print(f"{len(stable)} of {len(rows)} nominal states are stable as seeded")
    spread = np.array([em for em, *_ in summary]) - e_min
    print(f"energy spread of distinct states: max {spread.max() * 1e18:.3f} aJ, "
          f"median gap between consecutive levels {np.median(np.diff(np.sort(spread))) * 1e18 if len(spread) > 1 else 0:.3f} aJ")

    res = {"params": p.to_dict(), "unit": a.unit, "bias": a.bias, "rows": rows,
           "summary": [{"dE_aJ": (em - e_min) * 1e18, "state": k, "n": nn} for em, k, nn, sd in summary]}
    (out / "catalogue.json").write_text(json.dumps(res, indent=1, default=str))

    if a.hysteresis:
        print("\nhysteresis sweep along x (+1 deg), -60..60 mT:", flush=True)
        d = (np.cos(np.radians(1)), np.sin(np.radians(1)))
        sim.set_magnetization(macrospin_magnetization(p, sim.regions, sim.islands, -1.0))
        sim.set_field((-0.2 * d[0], -0.2 * d[1], 0.0)); sim.minimize()
        fields = np.arange(-60e-3, 60e-3 + 0.5e-3, 1e-3)
        ms = []
        for b in fields:
            sim.set_field((b * d[0], b * d[1], 0.0)); sim.minimize()
            ms.append(sim.region_averages())
        ms = np.stack(ms)
        proj = ms @ np.array([*d, 0.0])
        for j, isl in enumerate(sim.islands):
            dm = np.diff(proj[:, j]); ev = [(float(fields[i + 1] * 1e3), float(dm[i])) for i in np.where(np.abs(dm) > 0.5)[0]]
            print(f"  island {isl.index} {'bottom' if isl.layer == 0 else 'top':6s}: events (mT, dm) {[(round(b, 1), round(x, 2)) for b, x in ev]}")
        np.savez_compressed(out / "sweep.npz", fields=fields, m_static=ms, region_ids=np.array(sim.region_ids), angle_deg=1.0)
    print("wrote", out)


if __name__ == "__main__":
    main()

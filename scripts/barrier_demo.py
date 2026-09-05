#!/usr/bin/env python
"""Switching barrier of the top layer (AP -> P) with the string method, its field dependence,
and its design gradient by the envelope theorem at the saddle (magnum.np, CPU).

    python scripts/barrier_demo.py --fields 0 15e-3 25e-3 --fd width --out runs/barrier_demo
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asvi_rc import ASVIParams                                                  # noqa: E402
from asvi_rc.geometry import single_island, macrospin_magnetization             # noqa: E402
from asvi_rc.magnumnp_driver import ASVISimulationNP                            # noqa: E402
from asvi_rc.barriers import string_barrier                                     # noqa: E402


def make(p, islands, design_vals):
    import torch
    design = {k: torch.tensor(v, dtype=torch.float64, requires_grad=True) for k, v in design_vals.items()}
    return ASVISimulationNP(p, islands, verbose=False, design=design)


def relax(sim, p, signs, b):
    sim.set_field((b, 0.0, 0.0))
    sim.set_magnetization(macrospin_magnetization(p, sim.regions, sim.islands, signs, tilt=0.02))
    sim.minimize()
    return sim.get_magnetization(), sim.total_energy()


def barrier(sim, p, b, n_images, n_iter, verbose):
    m_ap, e_ap = relax(sim, p, {1: 1.0, 2: -1.0}, b)
    m_p, e_p = relax(sim, p, {1: 1.0, 2: 1.0}, b)
    res = string_barrier(sim, m_ap, m_p, n_images=n_images, n_iter=n_iter, verbose=verbose)
    return res, e_ap, e_p


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell-xy", type=float, default=10e-9)
    ap.add_argument("--fields", type=float, nargs="+", default=[0.0], help="applied fields along +x (T)")
    ap.add_argument("--images", type=int, default=16)
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--fd", type=str, default=None, help="finite-difference check of the barrier gradient for this parameter")
    ap.add_argument("--fd-h", type=float, default=2e-9)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    p = ASVIParams(cell_xy=a.cell_xy, cell_z=5e-9, disorder_sigma=0.0, box_override=(640e-9, 320e-9),
                   pbc_repetitions=(0, 0, 0))
    p.t_spacer = 110e-9 - p.t_bottom - p.t_top          # grid headroom, the soft mask places the layers
    islands = single_island(p)
    vals = {"width": 140e-9, "t_spacer": 35e-9, "t_top": 20e-9, "length": 550e-9}
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    sim = make(p, islands, vals)
    rows = []
    for b in a.fields:
        t0 = time.time()
        res, e_ap, e_p = barrier(sim, p, b, a.images, a.iters, a.verbose)
        sim.set_magnetization(res["m_saddle"]); sim.set_material()
        g_s = sim.energy_grad(list(vals))
        sim.set_magnetization(res["images"][0]); sim.set_material()
        g_a = sim.energy_grad(list(vals))
        grad = {k: g_s[k] - g_a[k] for k in vals}
        row = {"B_mT": b * 1e3, "barrier_aJ": res["barrier_J"] * 1e18, "barrier_back_aJ": res["barrier_back_J"] * 1e18,
               "dE_P_minus_AP_aJ": (e_p - e_ap) * 1e18, "i_saddle": res["i_saddle"], "iterations": res["iterations"],
               "seconds": time.time() - t0, "grad_barrier_J_per_unit": grad,
               "path_aJ": ((res["energies"] - res["energies"][0]) * 1e18).round(3).tolist()}
        rows.append(row)
        print(f"B = {b * 1e3:5.1f} mT: barrier AP->P {row['barrier_aJ']:7.3f} aJ (back {row['barrier_back_aJ']:7.3f}), "
              f"dE(P-AP) {row['dE_P_minus_AP_aJ']:7.3f} aJ, saddle image {res['i_saddle']}/{a.images - 1}, "
              f"{res['iterations']} its, saddle torque {res['climb_torque']:.3g} A/m, {row['seconds']:.0f} s", flush=True)
        print("   d(barrier)/d: " + "  ".join(f"{k} {v * 1e18 * 1e-9:+.3f} aJ/nm" for k, v in grad.items()), flush=True)
        print("   path (aJ): " + " ".join(f"{x:.1f}" for x in row["path_aJ"]), flush=True)
    if a.fd:
        b = a.fields[0]
        vals_p, vals_m = dict(vals), dict(vals)
        vals_p[a.fd] += a.fd_h; vals_m[a.fd] -= a.fd_h
        bp = barrier(make(p, islands, vals_p), p, b, a.images, a.iters, False)[0]["barrier_J"]
        bm = barrier(make(p, islands, vals_m), p, b, a.images, a.iters, False)[0]["barrier_J"]
        fd = (bp - bm) / (2 * a.fd_h)
        print(f"FD d(barrier)/d{a.fd} at B = {b * 1e3:g} mT: {fd * 1e9:.4g} aJ/nm  "
              f"(autograd {rows[0]['grad_barrier_J_per_unit'][a.fd] * 1e9:.4g})", flush=True)
        rows[0]["fd_" + a.fd] = fd
    (out / "barriers.json").write_text(json.dumps(rows, indent=1))
    print("wrote", out / "barriers.json")


if __name__ == "__main__":
    main()

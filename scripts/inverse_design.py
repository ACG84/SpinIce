#!/usr/bin/env python
"""Gradient-based inverse design of the ASVI island with magnum.np (CPU is fine).

The design parameters (island width/length, layer thicknesses, top-layer offset,
Ms, A) enter through a differentiable anti-aliased mask, so the energy of every
relaxed state and its magnetic moment are torch scalars with gradients with
respect to the design.  Relaxed states are stationary in m, so the energy
gradients are exact at fixed m (envelope theorem); the moment gradients neglect
the response of m*, which is small for the macrospin/vortex states used here.

Objectives (all over the field-addressable states, i.e. those with a net moment
different from the ground state's):
  spread   : standard deviation of the level energies dE_i = E_i - E_0
             (well-clustered levels)
  reorder  : mean reordering field B_i = dE_i / max_g |M_i - M_g| along the island
             axis, g running over the two degenerate antiparallel ground states
             (the same definition as scripts/catalogue_analysis.py)
  escape   : reordering field of the *lowest* excited level only (the field
             that first lets the ground state reorder)
A penalty keeps every level at least --min-level above the ground state, so the
optimiser cannot "win" by turning a vortex state into the new ground state.

    python scripts/inverse_design.py --design width t_spacer --objective reorder --steps 15 --out runs/invdes
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asvi_rc import ASVIParams                                                  # noqa: E402
from asvi_rc.geometry import single_island, macrospin_magnetization, vortex_magnetization, classify_layer  # noqa: E402
from asvi_rc.magnumnp_driver import ASVISimulationNP, DESIGN_KEYS               # noqa: E402

GROUND = [("+", "-"), ("-", "+")]                    # degenerate antiparallel ground states
ADDRESSABLE = [("+", "+"), ("+", "V+"), ("+", "V-"), ("V+", "-"), ("V+", "+")]
SCALE = {"length": 100e-9, "width": 20e-9, "t_bottom": 5e-9, "t_spacer": 5e-9, "t_top": 5e-9,
         "offset_x": 20e-9, "offset_y": 20e-9, "msat_bottom": 100e3, "msat_top": 100e3, "aex": 2e-12}
BOUNDS = {"length": (350e-9, 700e-9), "width": (90e-9, 240e-9), "t_bottom": (10e-9, 45e-9),
          "t_spacer": (5e-9, 80e-9), "t_top": (5e-9, 45e-9), "offset_x": (-100e-9, 100e-9),
          "offset_y": (-100e-9, 100e-9), "msat_bottom": (300e3, 1200e3), "msat_top": (300e3, 1200e3),
          "aex": (5e-12, 25e-12)}
FMT = {"length": 1e9, "width": 1e9, "t_bottom": 1e9, "t_spacer": 1e9, "t_top": 1e9,
       "offset_x": 1e9, "offset_y": 1e9, "msat_bottom": 1e-3, "msat_top": 1e-3, "aex": 1e12}


def seed(p, sim, labels):
    signs = {isl.region: (1.0 if lab == "+" else -1.0) for isl, lab in zip(sim.islands, labels) if lab in "+-"}
    m = macrospin_magnetization(p, sim.regions, sim.islands, signs or 1.0, tilt=0.02)
    chir = {isl.region: (1 if lab == "V+" else -1) for isl, lab in zip(sim.islands, labels) if lab.startswith("V")}
    if chir:
        m = vortex_magnetization(p, sim.regions, sim.islands, chirality=chir, base=m)
    return m


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--design", nargs="+", default=["width", "t_spacer"], choices=DESIGN_KEYS)
    ap.add_argument("--objective", choices=["spread", "reorder", "escape"], default="reorder")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--lr", type=float, default=0.5, help="max change per step in units of SCALE[param]")
    ap.add_argument("--cell-xy", type=float, default=10e-9)
    ap.add_argument("--cell-z", type=float, default=5e-9)
    ap.add_argument("--box", type=float, nargs=2, default=(640e-9, 320e-9))
    ap.add_argument("--z-max", type=float, default=110e-9, help="grid height (headroom for thickness changes)")
    ap.add_argument("--init", nargs="*", default=[], help="initial values as key=value (SI)")
    ap.add_argument("--states", type=int, default=len(ADDRESSABLE), help="use the first N addressable states")
    ap.add_argument("--min-level", type=float, default=5e-18, help="penalise levels closer than this (J) to the ground state")
    ap.add_argument("--penalty", type=float, default=10.0, help="penalty weight per unit of (min_level - dE)/min_level")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    import torch

    p = ASVIParams(cell_xy=a.cell_xy, cell_z=a.cell_z, disorder_sigma=0.0, box_override=tuple(a.box),
                   pbc_repetitions=(0, 0, 0))
    nominal = {"length": p.length, "width": p.width, "t_bottom": p.t_bottom, "t_spacer": p.t_spacer,
               "t_top": p.t_top, "offset_x": p.top_offset[0], "offset_y": p.top_offset[1],
               "msat_bottom": p.msat, "msat_top": p.msat, "aex": p.aex}
    for kv in a.init:
        k, v = kv.split("=")
        nominal[k] = float(v)
    # grid headroom: the nominal stack only fixes nz; the soft mask places the layers
    p.t_spacer = a.z_max - p.t_bottom - p.t_top
    islands = single_island(p)
    design = {k: torch.tensor(nominal[k], dtype=torch.float64, requires_grad=True) for k in DESIGN_KEYS}
    free = list(a.design)
    states = GROUND + ADDRESSABLE[: a.states]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    sim = ASVISimulationNP(p, islands, verbose=False, design=design)
    axis = torch.tensor([*islands[0].axis, 0.0], dtype=torch.float64)
    history = []
    m_prev = {}
    t0 = time.time()
    for step in range(a.steps + 1):
        sim.set_material()
        E, M, ok = {}, {}, {}
        # 1. relax every tracked state (no graph)
        for lab in states:
            sim.set_magnetization(m_prev.get(lab, seed(p, sim, lab)))
            sim.minimize()
            m = sim.get_magnetization()
            final = tuple(classify_layer(p, m, sim.regions, isl) for isl in sim.islands)
            ok[lab] = final == lab
            if ok[lab]:
                m_prev[lab] = m
            else:
                m_prev.pop(lab, None)
            # 2. energy and moment with the design graph attached (m fixed)
            sim.set_material()
            E[lab] = sim.energy_tensor()
            M[lab] = (sim.moment() * axis).sum()
        grounds = [lab for lab in GROUND if ok[lab]]
        g = min(grounds, key=lambda l: float(E[l]))
        valid = [lab for lab in states if lab not in GROUND and ok[lab]]
        dE = {lab: E[lab] - E[g] for lab in valid}
        dM = {lab: torch.stack([(M[lab] - M[gg]).abs() for gg in grounds]).max() + 1e-3 * M[g].abs().detach()
              for lab in valid}                                     # guard M-degenerate levels
        B = {lab: dE[lab] / dM[lab] for lab in valid}
        if a.objective == "spread":
            J = torch.stack([dE[l] for l in valid]).std() * 1e18
        elif a.objective == "reorder":
            J = torch.stack([B[l] for l in valid]).mean() * 1e3
        else:
            low = min(valid, key=lambda l: float(dE[l]))
            J = B[low] * 1e3
        J = J + a.penalty * sum(torch.relu(a.min_level - dE[l]) / a.min_level for l in valid)
        grads = torch.autograd.grad(J, [design[k] for k in free], allow_unused=True)
        grads = {k: (0.0 if gr is None else float(gr)) for k, gr in zip(free, grads)}
        rec = {"step": step, "J": float(J), "design": {k: float(design[k]) for k in DESIGN_KEYS},
               "levels_aJ": {"/".join(l): float(dE[l]) * 1e18 for l in valid},
               "B_reorder_mT": {"/".join(l): float(B[l]) * 1e3 for l in valid},
               "stable": {"/".join(l): bool(ok[l]) for l in states}, "grad": grads, "t_s": time.time() - t0}
        history.append(rec)
        (out / "history.json").write_text(json.dumps(history, indent=1))
        print(f"step {step:2d}  J = {float(J):8.4f}  " +
              "  ".join(f"{k}={float(design[k]) * FMT[k]:.4g}" for k in free) +
              "  levels " + " ".join(f"{v:.1f}" for v in rec["levels_aJ"].values()) +
              "  B " + " ".join(f"{v:.1f}" for v in rec["B_reorder_mT"].values()) +
              ("" if all(ok.values()) else "  collapsed: " + ",".join("/".join(l) for l in states if not ok[l])) +
              f"  ({time.time() - t0:.0f} s)", flush=True)
        if step == a.steps:
            break
        # 3. normalised gradient step with bounds (each parameter moves at most lr*SCALE)
        gn = max(abs(grads[k]) * SCALE[k] for k in free) or 1.0
        with torch.no_grad():
            for k in free:
                new = float(design[k]) - a.lr * SCALE[k] * grads[k] * SCALE[k] / gn
                design[k].fill_(min(max(new, BOUNDS[k][0]), BOUNDS[k][1]))
    print("wrote", out / "history.json")


if __name__ == "__main__":
    main()

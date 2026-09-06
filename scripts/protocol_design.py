#!/usr/bin/env python
"""Optimise the field protocol for memory on a fixed energy landscape (soft-automaton proxy).

Reads a catalogue (energies + axis moments of the relaxed states), builds the soft
automaton, and maximises the memory capacity MC = sum_k R^2(k) of a ridge readout
over the protocol parameters (write window b_min/b_max, leak amplitude and jitter,
readout bias) by gradient ascent.  Seconds per step, no micromagnetics.

    python scripts/protocol_design.py docs/data/np_single_10/catalogue.json --steps 40 --out runs/protocol_design
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asvi_rc import ASVIParams                                                  # noqa: E402
from asvi_rc.geometry import build_islands, single_island                       # noqa: E402
from asvi_rc.softautomaton import memory_proxy                                  # noqa: E402
from scripts.catalogue_analysis import layer_moment                             # noqa: E402

PROTO = {"b_min": 28e-3, "b_max": 38e-3, "leak": 33e-3, "jitter": 5e-3, "bias": -20e-3}
BOUNDS = {"b_min": (5e-3, 80e-3), "b_max": (5e-3, 100e-3), "leak": (0.0, 80e-3), "jitter": (0.0, 30e-3),
          "bias": (-60e-3, 0.0)}


def landscape(path, angle_deg):
    """Distinct relaxed states of a catalogue with energies and moments projected on the drive axis.

    Moments: sum over layer-islands of m_axis * layer moment * (island axis . drive direction), so a
    45 deg drive of the unit cell couples to both sublattices with 1/sqrt(2) each."""
    import torch
    c = json.loads(Path(path).read_text())
    p = ASVIParams.from_dict(c["params"])
    islands = build_islands(p) if c.get("unit") == "cell" else single_island(p)
    d = np.array([np.cos(np.radians(angle_deg)), np.sin(np.radians(angle_deg))])
    proj = [layer_moment(p, isl.layer) * float(np.dot(isl.axis, d)) for isl in islands]
    best = {}
    for r in c["rows"]:
        k = tuple(r["final"])
        if k not in best or r["energy_J"] < best[k]["energy_J"]:
            best[k] = r
    rows = sorted(best.values(), key=lambda r: r["energy_J"])
    states = [tuple(r["final"]) for r in rows]
    E = torch.tensor([r["energy_J"] for r in rows], dtype=torch.float64)
    M = torch.tensor([sum(mx * mo for mx, mo in zip(r["m_axis"], proj)) for r in rows], dtype=torch.float64)
    return states, E, M


def stages_from(pr, kind):
    if kind == "leak":
        return [lambda ut, rt: -(pr["leak"] + pr["jitter"] * (2 * rt - 1)),
                lambda ut, rt: pr["b_min"] + (pr["b_max"] - pr["b_min"]) * ut,
                lambda ut, rt: pr["bias"]]
    return [lambda ut, rt: pr["b_min"] + (pr["b_max"] - pr["b_min"]) * ut, lambda ut, rt: pr["bias"]]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("catalogue")
    ap.add_argument("--protocol", choices=["leak", "unipolar"], default="leak")
    ap.add_argument("--angle", type=float, default=1.0, help="drive axis (deg from x); 45 for the unit cell")
    ap.add_argument("--free", nargs="+", default=list(PROTO), choices=list(PROTO))
    ap.add_argument("--init", nargs="*", default=[], help="key=value (T)")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3, help="max field change per step (T)")
    ap.add_argument("--barrier", type=float, default=70e-18)
    ap.add_argument("--coercive", type=float, nargs="+", default=None,
                    help="per-layer coercive fields (T) instead of one barrier energy")
    ap.add_argument("--width", type=float, default=2e-3)
    ap.add_argument("--n-steps", type=int, default=400)
    ap.add_argument("--k-max", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    import torch
    states, E, M = landscape(a.catalogue, a.angle)
    print(f"{len(states)} states, drive axis {a.angle:g} deg", flush=True)
    pr = {k: torch.tensor(v, dtype=torch.float64, requires_grad=True) for k, v in PROTO.items()}
    for kv in a.init:
        k, v = kv.split("="); pr[k].data.fill_(float(v))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    hist = []
    lr, best = a.lr, -np.inf
    for step in range(a.steps + 1):
        mc, r2, rho = memory_proxy(E, M, states, n_steps=a.n_steps, stages=stages_from(pr, a.protocol),
                                   barrier=a.coercive if a.coercive else a.barrier, width=a.width, k_max=a.k_max)
        g = torch.autograd.grad(mc, [pr[k] for k in a.free], allow_unused=True)
        g = {k: (0.0 if x is None else float(x)) for k, x in zip(a.free, g)}
        occ = rho.detach().mean(0).numpy()
        ent = float(-(occ[occ > 1e-9] * np.log2(occ[occ > 1e-9])).sum())
        rec = {"step": step, "MC": float(mc), "R2": r2.detach().numpy().round(3).tolist(),
               "protocol_mT": {k: float(v) * 1e3 for k, v in pr.items()}, "entropy_bits": ent, "grad": g}
        hist.append(rec)
        print(f"step {step:2d}  MC {float(mc):5.2f}  R2 {' '.join(f'{x:.2f}' for x in rec['R2'])}  "
              + " ".join(f"{k}={float(v) * 1e3:.1f}" for k, v in pr.items()) + f"  H {ent:.2f} bits", flush=True)
        if step == a.steps:
            break
        if float(mc) < best:                                   # overshoot: halve the step
            lr *= 0.5
        best = max(best, float(mc))
        gn = max(abs(g[k]) for k in a.free) or 1.0
        with torch.no_grad():
            for k in a.free:
                new = float(pr[k]) + lr * g[k] / gn
                pr[k].fill_(min(max(new, BOUNDS[k][0]), BOUNDS[k][1]))
    (out / "history.json").write_text(json.dumps(hist, indent=1))
    print("wrote", out / "history.json")


if __name__ == "__main__":
    main()

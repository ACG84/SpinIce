#!/usr/bin/env python
"""Memory curves of the ASVI lattice automaton (multilayer islands from a micromagnetic
catalogue, dumbbell coupling, per-layer coercive fields) under the field-loop protocols.

    python scripts/lattice_rc.py --catalogue docs/data/np_single_10/catalogue.json --coercive 0.05 0.0216 \\
        --cells 4 4 --angle 45 --window 30e-3 60e-3 --leak 40e-3 --jitter 5e-3 --out runs/lat_2layer
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asvi_rc.lattice_automaton import ASVILattice                              # noqa: E402
from scripts.flatspin_rc import ridge_r2                                       # noqa: E402


def run(a, lattice_constant, seed):
    lat = ASVILattice(a.catalogue, a.coercive, tuple(a.cells), lattice_constant, a.width, seed=seed)
    rng = np.random.default_rng(100 + seed)
    u = rng.random(a.n)
    d = np.array([np.cos(np.radians(a.angle)), np.sin(np.radians(a.angle))])
    feats, flips, keys = [], [], []
    t0 = time.time()
    for t in range(a.n):
        L = a.leak + a.jitter * rng.uniform(-1, 1)
        lat.relax(-L * d, sample=not a.deterministic)
        h = a.window[0] + (a.window[1] - a.window[0]) * u[t]
        flips.append(lat.relax(h * d, sample=not a.deterministic))
        lat.relax(a.bias * d, sample=not a.deterministic)
        feats.append(lat.features(a.features)); keys.append(lat.state_key())
    X = np.array(feats)
    r2, mc = ridge_r2(X, u, k_max=8)
    return {"a_nm": lat.a * 1e9, "islands": lat.n, "states_per_island": lat.K, "R2": r2.round(3).tolist(), "MC": mc,
            "flips": float(np.mean(flips)), "distinct": len(set(keys[a.n // 2:])),
            "vortex_frac": float(np.mean([f[-lat.n:].mean() for f in feats])) if a.features == "moment" else None,
            "seconds": time.time() - t0}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap._negative_number_matcher = re.compile(r"^-\d+$|^-\d*\.\d+$|^-\d+(\.\d*)?[eE][-+]?\d+$")
    ap.add_argument("--catalogue", required=True)
    ap.add_argument("--coercive", type=float, nargs="+", required=True, help="per-layer coercive fields (T)")
    ap.add_argument("--cells", type=int, nargs=2, default=(4, 4))
    ap.add_argument("--lattice", type=float, nargs="+", default=[None], help="lattice constants (m); default catalogue")
    ap.add_argument("--angle", type=float, default=45.0)
    ap.add_argument("--window", type=float, nargs=2, default=(30e-3, 60e-3), help="write amplitudes (T)")
    ap.add_argument("--leak", type=float, default=40e-3); ap.add_argument("--jitter", type=float, default=5e-3)
    ap.add_argument("--bias", type=float, default=0.0)
    ap.add_argument("--width", type=float, default=2e-3)
    ap.add_argument("--features", choices=["moment", "onehot"], default="moment")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = []
    print(f"{'a (nm)':>7s} {'islands':>7s} {'MC':>5s}  R2(k=0..8)                           distinct  flips  vortex  s/run")
    for lc in a.lattice:
        rs = [run(a, lc, s) for s in range(a.seeds)]
        r = dict(rs[0]); r["MC"] = float(np.mean([x["MC"] for x in rs]))
        r["R2"] = np.mean([x["R2"] for x in rs], axis=0).round(2).tolist()
        for k in ("flips", "distinct", "vortex_frac", "seconds"):
            r[k] = float(np.mean([x[k] for x in rs])) if rs[0][k] is not None else None
        rows.append(r)
        print(f"{r['a_nm']:7.0f} {r['islands']:7d} {r['MC']:5.2f}  {' '.join(f'{x:5.2f}' for x in r['R2'])}  "
              f"{r['distinct']:8.0f}  {r['flips']:5.1f}  {r['vortex_frac'] if r['vortex_frac'] is None else round(r['vortex_frac'], 2)!s:>6s}  {r['seconds']:5.0f}", flush=True)
    (out / "lattice_rc.json").write_text(json.dumps(rows, indent=1))
    print("wrote", out / "lattice_rc.json")


if __name__ == "__main__":
    main()

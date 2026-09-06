#!/usr/bin/env python
"""Lattice-scale memory curves with flatspin (macrospin ASI: Stoner-Wohlfarth switching,
dipolar coupling, quenched disorder), to find what lattice size / disorder / protocol gives
fading memory before mapping the multilayer islands onto it.

Input u_t in [0, 1] -> field-loop amplitude h = h_min + (h_max - h_min) u_t (units of hc)
applied along a fixed angle; protocols: unipolar (apply +h, relax, remove), alternating
(sign flips every step), leak (negative excursion of random amplitude before the write),
and rotate (angle advances by --rot degrees every step, the classic flatspin RC drive).
Features: the spin vector after the step.  Readout: ridge, test R^2(k) of u(t-k).

    python scripts/flatspin_rc.py --size 8 8 --disorder 0.05 --protocol alternating --out runs/fs_8x8
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def ridge_r2(X, y, k_max=8, alpha=1e-2, train_frac=0.6, washout=50):
    T = X.shape[0]
    Xb = np.hstack([X, np.ones((T, 1))])
    out = []
    for k in range(k_max + 1):
        idx = np.arange(washout + k, T)
        Xk, yk = Xb[idx], y[idx - k]
        n_tr = int(len(idx) * train_frac)
        A = Xk[:n_tr].T @ Xk[:n_tr] + alpha * np.eye(Xb.shape[1])
        w = np.linalg.solve(A, Xk[:n_tr].T @ yk[:n_tr])
        res = np.sum((yk[n_tr:] - Xk[n_tr:] @ w) ** 2)
        tot = np.sum((yk[n_tr:] - yk[n_tr:].mean()) ** 2)
        out.append(1 - res / tot)
    r2 = np.array(out)
    return r2, float(np.clip(r2, 0, None).sum())


def run(size, disorder, alpha, hc, angle, protocol, h_min, h_max, n, seed, leak, jitter, rot, model="square",
        angles=None, hard_frac=0.0, hard_ratio=1.0, stack=1):
    """angles     : list of drive angles cycled step by step (spatial multiplexing of time)
    hard_frac  : fraction of islands with a coercive field hard_ratio x hc (persistent population)
    stack      : number of past spin vectors concatenated in the readout (delay line, baseline)"""
    import flatspin.model as fm
    cls = {"square": fm.SquareSpinIceClosed, "pinwheel": fm.PinwheelSpinIceDiamond,
           "kagome": fm.KagomeSpinIce}[model]
    m = cls(size=tuple(size), alpha=alpha, disorder=disorder, hc=hc, use_opencl=0, random_seed=seed)
    rng = np.random.default_rng(seed)
    if hard_frac > 0:
        hard = rng.random(m.spin_count) < hard_frac
        m.threshold = np.where(hard, m.threshold * hard_ratio, m.threshold)
    u = rng.random(n)
    m.polarize()
    feats, nflips = [], []
    cyc = list(angles) if angles else [angle]
    for t in range(n):
        th = np.radians(cyc[t % len(cyc)])
        if protocol == "leak":
            L = (leak + jitter * rng.uniform(-1, 1)) * hc
            m.set_h_ext([-L * np.cos(th), -L * np.sin(th)]); m.relax()
        h = (h_min + (h_max - h_min) * u[t]) * hc
        sgn = (-1) ** t if protocol == "alternating" else 1.0
        if protocol == "rotate":
            th = np.radians(angle + rot * t)
        m.set_h_ext([sgn * h * np.cos(th), sgn * h * np.sin(th)])
        nflips.append(m.relax())
        m.set_h_ext([0.0, 0.0]); m.relax()
        feats.append(m.spin.copy())
    X = np.array(feats, dtype=float)
    if stack > 1:
        X = np.hstack([np.roll(X, k, axis=0) for k in range(stack)])
    r2, mc = ridge_r2(X, u)
    distinct = len({tuple(f) for f in feats[n // 2:]})
    return {"size": list(size), "disorder": disorder, "alpha": alpha, "protocol": protocol, "angle": angle,
            "angles": cyc, "hard_frac": hard_frac, "hard_ratio": hard_ratio, "stack": stack,
            "window": [h_min, h_max], "spins": int(m.spin_count), "R2": r2.round(3).tolist(), "MC": mc,
            "distinct_states_2nd_half": distinct, "mean_flips": float(np.mean(nflips))}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", type=int, nargs=2, default=(8, 8))
    ap.add_argument("--model", choices=["square", "pinwheel", "kagome"], default="square")
    ap.add_argument("--disorder", type=float, nargs="+", default=[0.05])
    ap.add_argument("--alpha", type=float, default=0.1, help="dipolar coupling strength (flatspin alpha)")
    ap.add_argument("--hc", type=float, default=0.03, help="mean coercive field (T)")
    ap.add_argument("--angle", type=float, default=45.0)
    ap.add_argument("--protocol", nargs="+", default=["alternating"], choices=["unipolar", "alternating", "leak", "rotate"])
    ap.add_argument("--window", type=float, nargs=2, default=(0.8, 1.2), help="amplitude range in units of hc")
    ap.add_argument("--leak", type=float, default=1.0); ap.add_argument("--jitter", type=float, default=0.15)
    ap.add_argument("--rot", type=float, default=7.0, help="degrees per step for the rotate protocol")
    ap.add_argument("--angles", type=float, nargs="+", default=None, help="cycle these drive angles step by step")
    ap.add_argument("--hard-frac", type=float, default=0.0); ap.add_argument("--hard-ratio", type=float, default=1.0)
    ap.add_argument("--stack", type=int, default=1, help="readout delay line: concatenate the last k spin vectors")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = []
    print(f"{'protocol':12s} {'disorder':>8s} {'spins':>6s} {'MC':>6s}  R2(k=0..8)                          states  flips")
    for proto in a.protocol:
        for dis in a.disorder:
            rs = [run(a.size, dis, a.alpha, a.hc, a.angle, proto, *a.window, a.n, s, a.leak, a.jitter, a.rot, a.model,
                      a.angles, a.hard_frac, a.hard_ratio, a.stack) for s in range(a.seeds)]
            r = dict(rs[0]); r["MC"] = float(np.mean([x["MC"] for x in rs]))
            r["R2"] = np.mean([x["R2"] for x in rs], axis=0).round(2).tolist()
            r["distinct_states_2nd_half"] = float(np.mean([x["distinct_states_2nd_half"] for x in rs]))
            rows.append(r)
            print(f"{proto:12s} {dis:8.3f} {r['spins']:6d} {r['MC']:6.2f}  {' '.join(f'{x:5.2f}' for x in r['R2'])}  "
                  f"{r['distinct_states_2nd_half']:6.0f}  {r['mean_flips']:5.1f}", flush=True)
    (out / "flatspin_rc.json").write_text(json.dumps(rows, indent=1))
    print("wrote", out / "flatspin_rc.json")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Gradient-free inverse design of a flatspin lattice reservoir for long memory.

flatspin evaluations are cheap (seconds) but discrete, so this uses differential
evolution (scipy, parallel workers) over a mixed set of lattice, island and
protocol parameters:

    w_lo, w_width : write window [w_lo, w_lo + w_width] in units of hc
    leak, jitter  : random-leak excursion amplitude and half-range (units of hc)
    disorder      : relative spread of the coercive fields
    alpha         : dipolar coupling (T); the island moment / spacing knob
    sw_b, sw_beta : Stoner-Wohlfarth astroid shape (hard-axis ratio, curvature),
                    i.e. the island aspect ratio / shape proxy

Objectives: 'mc' (sum_k R^2(k), k <= k_max), 'long' (sum_k k R^2(k): rewards late
lags), 'lag' (R^2 at --target-lag).  Every evaluation averages --seeds random
lattices and input sequences; the best design is re-evaluated with more seeds.

    python scripts/flatspin_design.py --size 12 12 --objective long --gens 12 --workers 3 --out runs/fs_design
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.flatspin_rc import ridge_r2                                        # noqa: E402

NAMES = ["w_lo", "w_width", "leak", "jitter", "disorder", "alpha", "sw_b", "sw_beta"]
BOUNDS = [(0.8, 2.2), (0.1, 1.2), (0.3, 2.0), (0.0, 0.6), (0.01, 0.4), (0.001, 0.012), (0.2, 1.0), (1.0, 3.0)]
K_MAX = 10


def evaluate(x, size=(12, 12), hc=0.03, n=500, seeds=2, angle=45.0, model="square"):
    """Memory curve of one design, averaged over seeds.  Returns (R2 array, flips per step)."""
    import flatspin.model as fm
    w_lo, w_width, leak, jitter, disorder, alpha, sw_b, sw_beta = x
    cls = {"square": fm.SquareSpinIceClosed, "pinwheel": fm.PinwheelSpinIceDiamond}[model]
    th = np.radians(angle)
    r2s, flips = [], []
    for seed in range(seeds):
        m = cls(size=tuple(size), alpha=alpha, disorder=disorder, hc=hc, sw_b=sw_b, sw_beta=sw_beta,
                use_opencl=0, random_seed=seed)
        rng = np.random.default_rng(100 + seed)
        u = rng.random(n)
        m.polarize()
        feats, nf = [], []
        for t in range(n):
            L = (leak + jitter * rng.uniform(-1, 1)) * hc
            m.set_h_ext([-L * np.cos(th), -L * np.sin(th)]); m.relax()
            h = (w_lo + w_width * u[t]) * hc
            m.set_h_ext([h * np.cos(th), h * np.sin(th)]); nf.append(m.relax())
            m.set_h_ext([0.0, 0.0]); m.relax()
            feats.append(m.spin.copy())
        r2, _ = ridge_r2(np.array(feats, dtype=float), u, k_max=K_MAX)
        r2s.append(r2); flips.append(np.mean(nf))
    return np.mean(r2s, axis=0), float(np.mean(flips))


def objective_value(r2, kind, target_lag):
    r = np.clip(r2, 0, None)
    if kind == "mc":
        return float(r.sum())
    if kind == "long":
        return float(sum(k * r[k] for k in range(len(r))))
    return float(r[target_lag])


def _loss(x, cfg):
    r2, fl = evaluate(x, tuple(cfg["size"]), cfg["hc"], cfg["n"], cfg["seeds"], cfg["angle"], cfg["model"])
    return -objective_value(r2, cfg["objective"], cfg["target_lag"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", type=int, nargs=2, default=(12, 12))
    ap.add_argument("--model", choices=["square", "pinwheel"], default="square")
    ap.add_argument("--hc", type=float, default=0.03)
    ap.add_argument("--angle", type=float, default=45.0)
    ap.add_argument("--objective", choices=["mc", "long", "lag"], default="long")
    ap.add_argument("--target-lag", type=int, default=5)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--gens", type=int, default=10)
    ap.add_argument("--popsize", type=int, default=6, help="DE population multiplier (x 8 parameters)")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from scipy.optimize import differential_evolution
    cfg = {"size": list(a.size), "hc": a.hc, "n": a.n, "seeds": a.seeds, "angle": a.angle, "model": a.model,
           "objective": a.objective, "target_lag": a.target_lag}
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    history = []
    t0 = time.time()
    # start from the best hand-found design so DE refines rather than rediscovers
    x0 = np.array([1.5, 1.0, 1.0, 0.15, 0.05, 0.005, 0.41, 1.5])

    def cb(xk, convergence=0.0):
        r2, fl = evaluate(xk, tuple(a.size), a.hc, a.n, a.seeds, a.angle, a.model)
        rec = {"t_s": time.time() - t0, "x": dict(zip(NAMES, [float(v) for v in xk])),
               "R2": r2.round(3).tolist(), "objective": objective_value(r2, a.objective, a.target_lag), "flips": fl}
        history.append(rec)
        (out / "history.json").write_text(json.dumps(history, indent=1))
        print(f"gen {len(history):2d} ({rec['t_s']:.0f} s): {a.objective} = {rec['objective']:.3f}  "
              f"R2 {' '.join(f'{v:.2f}' for v in r2[:8])}  flips {fl:.0f}  "
              + " ".join(f"{k}={v:.3g}" for k, v in rec["x"].items()), flush=True)

    res = differential_evolution(_loss, BOUNDS, args=(cfg,), x0=x0, popsize=a.popsize, maxiter=a.gens,
                                 workers=a.workers, updating="deferred", polish=False, seed=0,
                                 tol=0, callback=cb)
    r2, fl = evaluate(res.x, tuple(a.size), a.hc, a.n, max(4, a.seeds), a.angle, a.model)
    best = {"x": dict(zip(NAMES, [float(v) for v in res.x])), "R2": r2.round(3).tolist(),
            "objective": objective_value(r2, a.objective, a.target_lag), "MC": float(np.clip(r2, 0, None).sum()),
            "flips": fl, "evaluations": int(res.nfev), "seconds": time.time() - t0}
    (out / "best.json").write_text(json.dumps(best, indent=1))
    print("best design (re-evaluated with more seeds):", json.dumps(best, indent=1))


if __name__ == "__main__":
    main()

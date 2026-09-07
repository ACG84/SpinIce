#!/usr/bin/env python
"""Memory capacity of the macrospin-LLG square spin ice in the SUB-SWITCHING regime.

The reservoir is `asvi_rc.macrospin_llg.SpinIce` (point-dipole coupling, uniaxial anisotropy B_k,
Gilbert damping alpha).  Operating point in dimensionless units, as in the parallel session's
shaped-island study: the drive amplitude is a fraction of B_k (default 6 %) so nothing switches,
and the state is sampled every `sample_tau` relaxation times tau = (1 + alpha^2) / (alpha gamma B_k).
Input: i.i.d. uniform u_t in [0, 1] (a periodic or Mackey-Glass input would inflate the memory
through its own autocorrelation), held as a constant field for one sample interval; features: all
3N magnetisation components at the end of the interval (optionally also at mid-interval).
Readout: ridge, 60 % train / 40 % test, R^2(k) for k = 0..k_max, MC = sum R^2(k); a product task
u_{t-1} u_{t-2} (not linearly reachable from the input history) as a nonlinearity check.

    python scripts/macrospin_rc.py --nx 4 --ny 4 --drive 0.06 --sample-tau 0.35 --n 3000
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asvi_rc.macrospin_llg import SpinIce, GAMMA                               # noqa: E402
from scripts.flatspin_rc import ridge_r2                                       # noqa: E402


def product_r2(X, u, alpha=1e-2, train_frac=0.6, washout=50):
    """R^2 of predicting u_{t-1} u_{t-2} (centred) from the state, held-out."""
    y = (u - 0.5)
    tgt = np.roll(y, 1) * np.roll(y, 2)
    Xk, tk = X[washout:], tgt[washout:]
    n_tr = int(train_frac * len(Xk))
    Xb = np.hstack([Xk, np.ones((len(Xk), 1))])
    A = Xb[:n_tr].T @ Xb[:n_tr] + alpha * np.eye(Xb.shape[1])
    w = np.linalg.solve(A, Xb[:n_tr].T @ tk[:n_tr])
    pred = Xb[n_tr:] @ w
    return 1 - np.mean((pred - tk[n_tr:]) ** 2) / np.var(tk[n_tr:])


def run(a):
    si = SpinIce(nx=a.nx, ny=a.ny, a=a.lattice, Bk=a.bk, alpha=a.alpha, seed=a.seed)
    tau = (1 + a.alpha ** 2) / (a.alpha * GAMMA * a.bk)
    f_prec = GAMMA * a.bk / (2 * np.pi)
    dt = a.dt if a.dt else 0.02 / f_prec                                      # 50 steps per precession period
    T_sample = a.sample_tau * tau
    steps = max(int(round(T_sample / dt)), 1)
    th = np.radians(a.angle); d = np.array([np.cos(th), np.sin(th), 0.0])
    # relax the as-initialised lattice at zero field
    si.run(int(5 * tau / dt), dt, lambda n: np.zeros(3))
    Bd = (si.D @ si.m.reshape(-1)).reshape(si.N, 3)
    rng = np.random.default_rng(100 + a.seed)
    u = rng.random(a.n)
    feats = []
    t0 = time.time()
    for t in range(a.n):
        B = a.drive * a.bk * 2 * (u[t] - 0.5) * d
        if a.mid:
            si.run(steps // 2, dt, lambda n: B); m_mid = si.m.reshape(-1).copy()
            si.run(steps - steps // 2, dt, lambda n: B)
            feats.append(np.concatenate([m_mid, si.m.reshape(-1)]))
        else:
            si.run(steps, dt, lambda n: B)
            feats.append(si.m.reshape(-1).copy())
    X = np.array(feats)
    r2, mc = ridge_r2(X, u, k_max=a.kmax, alpha=a.ridge)
    nl = product_r2(X, u, alpha=a.ridge)
    res = {"nx": a.nx, "ny": a.ny, "islands": si.N, "Bk_mT": a.bk * 1e3, "alpha": a.alpha, "drive_frac": a.drive,
           "drive_mT": a.drive * a.bk * 1e3, "tau_ns": tau * 1e9, "sample_ns": T_sample * 1e9, "dt_ps": dt * 1e12,
           "steps_per_sample": steps, "dip_mean_mT": float(np.linalg.norm(Bd, axis=1).mean() * 1e3),
           "features": X.shape[1], "R2": r2.round(3).tolist(), "MC": float(mc), "product_R2": float(nl),
           "seconds": time.time() - t0}
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nx", type=int, default=4); ap.add_argument("--ny", type=int, default=4)
    ap.add_argument("--lattice", type=float, default=300e-9)
    ap.add_argument("--bk", type=float, default=0.05, help="anisotropy field (T)")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--drive", type=float, default=0.06, help="drive amplitude as a fraction of B_k")
    ap.add_argument("--sample-tau", type=float, default=0.35, help="sample interval in relaxation times")
    ap.add_argument("--angle", type=float, default=45.0)
    ap.add_argument("--dt", type=float, default=None)
    ap.add_argument("--mid", action="store_true", help="also read the state at mid-interval")
    ap.add_argument("--n", type=int, default=3000); ap.add_argument("--kmax", type=int, default=20)
    ap.add_argument("--ridge", type=float, default=1e-2); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    r = run(a)
    print(f"{r['islands']} islands, Bk {r['Bk_mT']:.0f} mT, drive {r['drive_mT']:.1f} mT, tau {r['tau_ns']:.2f} ns, "
          f"sample {r['sample_ns']:.2f} ns ({r['steps_per_sample']} steps of {r['dt_ps']:.1f} ps), dipolar {r['dip_mean_mT']:.1f} mT, "
          f"{r['features']} features")
    print(f"  MC {r['MC']:.2f}   R2(k=0..{a.kmax}): {' '.join(f'{x:.2f}' for x in r['R2'])}")
    print(f"  product task u(t-1)u(t-2): R2 {r['product_R2']:.2f}   ({r['seconds']:.0f} s)")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()

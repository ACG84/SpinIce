"""
Inverse design of island cross-section for reservoir quality (random search over Fourier shapes,
from the parallel session; imports adapted to this package).

The ellipse sweep showed geometry swings the memory window 14x, with MC peaking
near 1.5:1 and nonlinearity collapsing at high aspect. Ellipses are a 1-parameter
family; Fourier shapes open the rest of the space -- asymmetric, lobed, or
tilted-easy-axis islands that no aspect ratio can express.

Design variables: harmonics k=2,3,4 (k=1 mostly translates the shape).

    r(theta) = a0 * (1 + sum_{k=2..4} [a_k cos(k theta) + b_k sin(k theta)])

a0 is set per shape so the cross-sectional AREA is held fixed, so the search
cannot cheat by simply shrinking or growing the island -- only shape varies.

Objective: maximize MC subject to keeping real nonlinearity.

    score = MC - 20 * max(0, 0.70 - NL)

Every candidate is evaluated at its own matched dimensionless operating point
(drive = 6% of Bk, sample = 0.35 tau), so a low-Bk shape gains nothing from
being sampled more favourably.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asvi_rc.demag_fourier import boundary, ellipse_coeffs                     # noqa: E402
from asvi_rc.macrospin_shaped import measure                                   # noqa: E402

AREA_TARGET = None      # set from the 2.75:1 ellipse reference


def shape_area(coeffs, a0=1.0, n_seg=400):
    _, _, _, area = boundary(coeffs, a0, n_seg)
    return area


def normalize_area(coeffs, area_target):
    """Choose a0 so the cross-section has the target area."""
    a1 = shape_area(coeffs, 1.0)
    return float(np.sqrt(area_target / a1))


def to_coeffs(v):
    """v = [a2,b2,a3,b3,a4,b4] -> coefficient list for harmonics 1..4."""
    return [(0.0, 0.0), (v[0], v[1]), (v[2], v[3]), (v[4], v[5])]


def valid(v, cap=0.78):
    return np.abs(v).sum() < cap


def score(v, area_target, **kw):
    c = to_coeffs(v)
    a0 = normalize_area(c, area_target)
    try:
        r = measure(c, a0, **kw)
    except Exception:
        return -99.0, None
    s = r['MC'] - 20.0 * max(0.0, 0.70 - r['nl'])
    return s, r


if __name__ == "__main__":
    import sys, time
    rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
    n_iter = int(sys.argv[2]) if len(sys.argv) > 2 else 30

    ce, r0 = ellipse_coeffs(2.75)
    AREA_TARGET = shape_area(ce, 110e-9 / r0)

    print("=" * 78)
    print("INVERSE DESIGN over Fourier island shapes")
    print(f"        area held fixed at {AREA_TARGET*1e18:.0f} nm^2; harmonics k=2,3,4")
    print("=" * 78)

    # baselines from the ellipse family, area-matched
    print(f"{'candidate':>26}{'Bk mT':>8}{'MC':>7}{'NL':>7}{'window':>9}{'score':>8}")
    print("-" * 78)
    best_v, best_s, best_r = None, -1e9, None
    for asp in [1.2, 1.5, 2.0, 2.75]:
        c, r0b = ellipse_coeffs(asp)
        a0 = normalize_area(c, AREA_TARGET)
        r = measure(c, a0)
        s = r['MC'] - 20.0 * max(0.0, 0.70 - r['nl'])
        print(f"{'ellipse ' + format(asp, '.2f') + ':1':>26}{r['Bk_mT']:>8.1f}"
              f"{r['MC']:>7.2f}{r['nl']:>7.2f}{r['window_ns']:>8.1f}n{s:>8.2f}")
        if s > best_s:
            best_s, best_r, best_v = s, r, ('ellipse', asp)

    t0 = time.time()
    hits = 0
    for i in range(n_iter):
        v = rng.normal(0, 0.115, 6)
        if not valid(v):
            continue
        s, r = score(v, AREA_TARGET)
        if r is None:
            continue
        hits += 1
        if s > best_s:
            best_s, best_r, best_v = s, r, v.copy()
            print(f"{'fourier #' + str(i):>26}{r['Bk_mT']:>8.1f}{r['MC']:>7.2f}"
                  f"{r['nl']:>7.2f}{r['window_ns']:>8.1f}n{s:>8.2f}  <- new best")
    print("-" * 78)
    print(f"evaluated {hits} shapes in {time.time()-t0:.0f}s")
    print(f"best score {best_s:.2f}")
    if isinstance(best_v, np.ndarray):
        print("best coeffs (a2,b2,a3,b3,a4,b4):",
              np.round(best_v, 3).tolist())
        np.save('runs/best_shape.npy', best_v)
    else:
        print("best remained an ellipse:", best_v)

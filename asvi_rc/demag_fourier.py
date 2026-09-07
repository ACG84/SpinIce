"""
Shape -> demagnetizing tensor -> anisotropy field, for Fourier-defined islands (from the parallel session).

Island cross-section in polar form:

    r(theta) = a0 * (1 + sum_k [ a_k cos(k theta) + b_k sin(k theta) ])

extruded to thickness t. For uniform magnetization the demag tensor follows
from magnetic surface charges (sigma = M . n_hat):

    N_ab = 1/(4 pi V M^2) * int int sigma_a(s) sigma_b(s') / |s - s'| dS dS'

In-plane components come from the side walls; N_zz from the top and bottom
faces. The z-integral through the wall thickness has a closed form:

    int_0^t int_0^t dz dz' / sqrt(d^2 + u^2)
        = 2 [ t asinh(t/d) - sqrt(d^2 + t^2) + d ]

The in-plane uniaxial anisotropy field is then

    B_k = mu0 * Ms * (N_hard - N_easy)

with the easy axis the eigenvector of the in-plane block having the SMALLER
demag factor.

Validation is by the sum rule N_xx + N_yy + N_zz = 1, which is exact for any
uniformly magnetized body and is independent of the discretization -- so it
catches both algebra and regularization errors.
"""
import numpy as np

MU0 = 4e-7 * np.pi


def boundary(coeffs, a0, n_seg=400):
    """Return midpoints, outward normals and arc lengths of the cross-section."""
    th = np.linspace(0, 2 * np.pi, n_seg, endpoint=False)
    r = np.ones_like(th)
    for k, (ak, bk) in enumerate(coeffs, start=1):
        r += ak * np.cos(k * th) + bk * np.sin(k * th)
    r = a0 * np.clip(r, 0.05, None)
    x, y = r * np.cos(th), r * np.sin(th)
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    mx, my = 0.5 * (x + xn), 0.5 * (y + yn)
    tx, ty = xn - x, yn - y
    L = np.hypot(tx, ty)
    nx, ny = ty / L, -tx / L                      # outward for CCW traversal
    area = 0.5 * np.sum(x * yn - xn * y)
    if area < 0:                                   # enforce CCW
        nx, ny, area = -nx, -ny, -area
    return np.stack([mx, my], 1), np.stack([nx, ny], 1), L, area


def _wall_kernel(d, t):
    """int_0^t int_0^t dz dz'/sqrt(d^2+u^2), regularized at d->0."""
    d = np.maximum(d, 1e-12)
    return 2.0 * (t * np.arcsinh(t / d) - np.sqrt(d * d + t * t) + d)


def demag_tensor(coeffs, a0, t, n_seg=400, n_pix=64):
    pts, nrm, L, area = boundary(coeffs, a0, n_seg)
    V = area * t

    # ---- in-plane block from side-wall charges ----
    dxy = pts[:, None, :] - pts[None, :, :]
    d = np.hypot(dxy[..., 0], dxy[..., 1])
    np.fill_diagonal(d, 0.0)
    self_d = L / 4.0                                # segment self-distance proxy
    d = d + np.diag(self_d)
    K = _wall_kernel(d, t) * (L[:, None] * L[None, :])
    N_in = np.zeros((2, 2))
    for a in range(2):
        for b in range(2):
            N_in[a, b] = (nrm[:, a][:, None] * nrm[:, b][None, :] * K).sum()
    N_in /= (4 * np.pi * V)

    # ---- N_zz from top/bottom face charges ----
    lim = a0 * (1 + sum(abs(ak) + abs(bk) for ak, bk in coeffs)) * 1.02
    g = np.linspace(-lim, lim, n_pix)
    GX, GY = np.meshgrid(g, g, indexing='ij')
    cell = (g[1] - g[0]) ** 2
    th = np.arctan2(GY, GX)
    rb = np.ones_like(th)
    for k, (ak, bk) in enumerate(coeffs, start=1):
        rb += ak * np.cos(k * th) + bk * np.sin(k * th)
    inside = np.hypot(GX, GY) <= a0 * np.clip(rb, 0.05, None)
    px = np.stack([GX[inside], GY[inside]], 1)
    dd = np.hypot(px[:, None, 0] - px[None, :, 0], px[:, None, 1] - px[None, :, 1])
    eff = 0.3820 * np.sqrt(cell)                    # self-term for a square cell
    same = 1.0 / np.maximum(dd, eff)
    opp = 1.0 / np.sqrt(dd ** 2 + t ** 2)
    N_zz = (cell ** 2) * (same.sum() - opp.sum()) / (2 * np.pi * V)

    return N_in, float(N_zz), float(area)


def anisotropy(coeffs, a0, t, Ms=800e3, **kw):
    """Return (B_k in tesla, easy-axis angle in radians, sum-rule residual)."""
    N_in, N_zz, area = demag_tensor(coeffs, a0, t, **kw)
    w, V = np.linalg.eigh(N_in)                     # w[0] <= w[1]
    Bk = MU0 * Ms * (w[1] - w[0])
    easy = float(np.arctan2(V[1, 0], V[0, 0]))
    resid = float(N_in.trace() + N_zz - 1.0)
    return float(Bk), easy, resid, float(N_zz), area * t


def ellipse_coeffs(aspect, n_harm=6):
    """Fourier coefficients approximating an ellipse of the given aspect ratio."""
    th = np.linspace(0, 2 * np.pi, 2048, endpoint=False)
    a, b = aspect, 1.0
    r = a * b / np.sqrt((b * np.cos(th)) ** 2 + (a * np.sin(th)) ** 2)
    r0 = r.mean()
    rn = r / r0 - 1.0
    coeffs = []
    for k in range(1, n_harm + 1):
        ak = 2 * np.mean(rn * np.cos(k * th))
        bk = 2 * np.mean(rn * np.sin(k * th))
        coeffs.append((ak, bk))
    return coeffs, r0


if __name__ == "__main__":
    t = 25e-9
    print("=" * 74)
    print("DEMAG TENSOR FROM FOURIER SHAPES -- validation")
    print("=" * 74)
    print("\nsum rule N_xx+N_yy+N_zz = 1 must hold for every shape\n")
    print(f"{'shape':>22}{'N_xx':>8}{'N_yy':>8}{'N_zz':>8}{'sum':>8}{'Bk (mT)':>10}")
    print("-" * 74)

    cases = [("circle r=110nm", [(0.0, 0.0)], 110e-9)]
    for asp in [1.5, 2.0, 2.75, 4.0]:
        c, r0 = ellipse_coeffs(asp)
        cases.append((f"ellipse {asp:.2f}:1", c, 110e-9 / r0 * 1.0))
    for name, c, a0 in cases:
        N_in, N_zz, _ = demag_tensor(c, a0, t)
        w = np.linalg.eigvalsh(N_in)
        Bk = MU0 * 800e3 * (w[1] - w[0])
        print(f"{name:>22}{N_in[0,0]:>8.4f}{N_in[1,1]:>8.4f}{N_zz:>8.4f}"
              f"{N_in.trace()+N_zz:>8.4f}{Bk*1e3:>10.1f}")
    print("-" * 74)
    print("  reference: 220x80x25 nm island used so far had Bk = 50 mT assumed")

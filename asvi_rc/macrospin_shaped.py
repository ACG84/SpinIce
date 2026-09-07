"""
Spin ice with per-island geometry (from the parallel session; imports adapted to this package).

Extends SpinIce so that each island's anisotropy field, easy-axis angle and
magnetic moment come from its actual cross-section via demag.py, rather than
being a single assumed constant.

Two changes matter physically:
  - Bk becomes an array; the anisotropy term is Bk_i (m_i . e_i) e_i
  - the dipole tensor is built for UNIT moment and scaled by each island's
    volume at apply time, so shape changes rescale the coupling correctly

Comparisons across shapes are made at matched DIMENSIONLESS operating point:
drive amplitude is a fixed fraction of Bk, and the sample interval is a fixed
fraction of the decay time tau = 1/(alpha*gamma*Bk). Otherwise a shape with
lower Bk would look better purely because it was sampled more favourably.
"""
import numpy as np
from .macrospin_llg import SpinIce, square_asi, dipole_tensor, GAMMA, MU0_4PI
from .demag_fourier import anisotropy, ellipse_coeffs


class ShapedSpinIce(SpinIce):
    def __init__(self, coeffs, a0, thickness=25e-9, nx=4, ny=4, a=300e-9,
                 Ms=800e3, alpha=0.05, seed=0):
        self.pos, base_axis = square_asi(nx, ny, a)
        self.N = len(self.pos)
        Bk, easy_off, resid, N_zz, vol = anisotropy(coeffs, a0, thickness, Ms=Ms)

        # island orientation: 0 for x-sublattice, pi/2 for y-sublattice,
        # plus whatever tilt the shape itself carries
        phi0 = np.where(base_axis[:, 0] > 0.5, 0.0, np.pi / 2)
        phi = phi0 + easy_off
        self.axis = np.stack([np.cos(phi), np.sin(phi),
                              np.zeros(self.N)], axis=1)

        self.Bk = np.full(self.N, Bk)
        self.moment = Ms * vol
        self.volume = vol
        self.D = dipole_tensor(self.pos, self.moment)
        self.alpha = alpha
        self.gp = GAMMA / (1 + alpha ** 2)
        self.sum_rule_resid = resid

        rng = np.random.default_rng(seed)
        m = self.axis * rng.choice([-1.0, 1.0], size=(self.N, 1))
        m += 0.05 * rng.standard_normal((self.N, 3))
        self.m = m / np.linalg.norm(m, axis=1, keepdims=True)

    def B_eff(self, m, B_ext):
        proj = np.sum(m * self.axis, axis=1, keepdims=True)
        B_an = self.Bk[:, None] * proj * self.axis
        B_dp = (self.D @ m.reshape(-1)).reshape(self.N, 3)
        return B_an + B_dp + B_ext


def tau_of(Bk, alpha):
    """Precessional decay time 1/(alpha * gamma * Bk)."""
    return 1.0 / (alpha * GAMMA * Bk)


def measure(coeffs, a0, thickness=25e-9, alpha=0.05, nx=4, ny=4,
            n=500, washout=100, max_delay=40, drive_frac=0.06,
            samp_frac=0.35, dt_frac=0.02, seed=0):
    """MC and memory window at a matched dimensionless operating point.

    MC sums the held-out R^2 over delays 1..max_delay (lag 0 is reported separately as r2_0),
    using the 60/40 train/test ridge of `scripts.flatspin_rc.ridge_r2`."""
    from scripts.flatspin_rc import ridge_r2 as _ridge_curve
    si = ShapedSpinIce(coeffs, a0, thickness, nx=nx, ny=ny, alpha=alpha, seed=seed)
    Bk = float(si.Bk[0])
    tau = tau_of(Bk, alpha)
    T_prec = 2 * np.pi / (GAMMA * Bk)
    dt = dt_frac * T_prec
    sps = max(int(round(samp_frac * tau / dt)), 10)

    rng = np.random.default_rng(0)
    u = rng.uniform(-1, 1, n)
    d = np.array([1, 1, 0], float); d /= np.linalg.norm(d)
    B0 = drive_frac * Bk
    X = np.empty((n, 3 * si.N))
    for k in range(n):
        B = B0 * u[k] * d
        for _ in range(sps):
            si.step(dt, B)
        X[k] = si.m.reshape(-1)
    Xs, us = X[washout:], u[washout:]
    md = min(max_delay, len(us) // 3)
    r2, _ = _ridge_curve(Xs, us, k_max=md, alpha=1e-2, washout=0)
    mc = np.clip(r2[1:], 0, None)
    horizon = int((mc > 0.5).sum())
    samp_ns = sps * dt * 1e9
    prod = np.roll(us, 1) * np.roll(us, 2)
    nl, _ = _ridge_curve(Xs[2:], prod[2:], k_max=0, alpha=1e-2, washout=0)
    return dict(Bk_mT=Bk * 1e3, tau_ns=tau * 1e9, fmr_GHz=GAMMA * Bk / (2 * np.pi) / 1e9,
                MC=float(mc.sum()), r2_0=float(r2[0]), horizon=horizon, window_ns=horizon * samp_ns,
                samp_ns=samp_ns, nl=float(nl[0]), vol=si.volume, resid=si.sum_rule_resid)


if __name__ == "__main__":
    import sys, time
    asp = float(sys.argv[1])
    c, r0 = ellipse_coeffs(asp)
    a0 = 110e-9 / r0
    t0 = time.time()
    r = measure(c, a0)
    print(f"{asp:>7.2f}{r['Bk_mT']:>9.1f}{r['fmr_GHz']:>8.2f}{r['tau_ns']:>8.2f}"
          f"{r['samp_ns']:>9.2f}{r['MC']:>8.2f}{r['horizon']:>9}"
          f"{r['window_ns']:>10.1f}{r['nl']:>7.2f}   [{time.time()-t0:.0f}s]")

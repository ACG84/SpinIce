"""
Artificial square spin ice as a reservoir: macrospin LLG with dipolar coupling.

(Brought in from a parallel session; the shaped-island extension there scored a memory capacity of
about 10 in the sub-switching regime.  `scripts/macrospin_rc.py` reproduces that measurement here.)

Each nanoisland is one macrospin m_i (unit vector) with uniaxial shape
anisotropy along its easy axis. Islands couple through point-dipole fields.

    B_eff,i = B_ext(t) + B_k (m_i . e_i) e_i + B_dip,i
    dm/dt   = -g' (m x B) - g' a (m x (m x B)),     g' = gamma/(1+a^2)

The dipolar field is linear in m, so the whole coupling is precomputed once as
a 3N x 3N tensor D and applied as a matrix-vector product each step. That is
what makes this fast enough to run reservoir-length trajectories.

Square ASI geometry: horizontal islands on x-bonds (easy axis x), vertical
islands on y-bonds (easy axis y), meeting at 4-island vertices.
"""
import numpy as np

GAMMA = 1.760859e11        # rad s^-1 T^-1
MU0_4PI = 1e-7             # T m A^-1


def square_asi(nx=4, ny=4, a=300e-9):
    """Positions and easy axes for a square ASI lattice."""
    pos, axis = [], []
    for i in range(nx):
        for j in range(ny):
            pos.append([i * a + a / 2, j * a, 0.0]); axis.append([1.0, 0, 0])
    for i in range(nx):
        for j in range(ny):
            pos.append([i * a, j * a + a / 2, 0.0]); axis.append([0, 1.0, 0])
    return np.array(pos), np.array(axis)


def dipole_tensor(pos, moment):
    """D such that B_dip_flat = D @ m_flat, with m_flat the 3N stacked units."""
    N = len(pos)
    D = np.zeros((3 * N, 3 * N))
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            r = pos[i] - pos[j]
            d = np.linalg.norm(r)
            rh = r / d
            # B_i from unit moment of j: (mu0/4pi) m [3 rh rh^T - I] / d^3
            blk = MU0_4PI * moment * (3 * np.outer(rh, rh) - np.eye(3)) / d ** 3
            D[3 * i:3 * i + 3, 3 * j:3 * j + 3] = blk
    return D


class SpinIce:
    def __init__(self, nx=4, ny=4, a=300e-9, Ms=800e3,
                 vol=(220e-9 * 80e-9 * 25e-9), Bk=0.05, alpha=0.05, seed=0):
        self.pos, self.axis = square_asi(nx, ny, a)
        self.N = len(self.pos)
        self.moment = Ms * vol
        self.D = dipole_tensor(self.pos, self.moment)
        self.Bk, self.alpha = Bk, alpha
        self.gp = GAMMA / (1 + alpha ** 2)
        rng = np.random.default_rng(seed)
        m = self.axis * rng.choice([-1.0, 1.0], size=(self.N, 1))
        m += 0.05 * rng.standard_normal((self.N, 3))
        self.m = m / np.linalg.norm(m, axis=1, keepdims=True)

    def B_eff(self, m, B_ext):
        proj = np.sum(m * self.axis, axis=1, keepdims=True)
        B_an = self.Bk * proj * self.axis
        B_dp = (self.D @ m.reshape(-1)).reshape(self.N, 3)
        return B_an + B_dp + B_ext

    def rhs(self, m, B_ext):
        B = self.B_eff(m, B_ext)
        mxB = np.cross(m, B)
        return -self.gp * (mxB + self.alpha * np.cross(m, mxB))

    def step(self, dt, B_ext):
        """Heun (predictor-corrector), renormalized."""
        k1 = self.rhs(self.m, B_ext)
        mp = self.m + dt * k1
        mp /= np.linalg.norm(mp, axis=1, keepdims=True)
        k2 = self.rhs(mp, B_ext)
        m = self.m + 0.5 * dt * (k1 + k2)
        self.m = m / np.linalg.norm(m, axis=1, keepdims=True)

    def energy(self, B_ext=np.zeros(3)):
        proj = np.sum(self.m * self.axis, axis=1)
        E_an = -0.5 * self.Bk * np.sum(proj ** 2)
        B_dp = (self.D @ self.m.reshape(-1)).reshape(self.N, 3)
        E_dp = -0.5 * np.sum(self.m * B_dp)
        E_z = -np.sum(self.m * B_ext)
        return float(E_an + E_dp + E_z)

    def run(self, T_steps, dt, B_of_t, sample_every=1):
        """Integrate, returning sampled states (n_samples x 3N)."""
        out = []
        for n in range(T_steps):
            self.step(dt, B_of_t(n))
            if n % sample_every == 0:
                out.append(self.m.reshape(-1).copy())
        return np.array(out)

    def ice_rule(self, nx, ny):
        """Fraction of interior vertices obeying the 2-in-2-out ice rule."""
        nH = nx * ny
        sgnH = np.sign(self.m[:nH, 0]).reshape(nx, ny)
        sgnV = np.sign(self.m[nH:, 1]).reshape(nx, ny)
        good = tot = 0
        for i in range(1, nx):
            for j in range(1, ny):
                # vertex at (i*a, j*a): W/E horizontal, S/N vertical
                inflow = int(sgnH[i - 1, j] > 0) + int(sgnH[i, j] < 0) \
                       + int(sgnV[i, j - 1] > 0) + int(sgnV[i, j] < 0)
                tot += 1
                good += (inflow == 2)
        return good / max(tot, 1)


if __name__ == "__main__":
    import time
    print("=" * 70)
    print("SPIN-ICE LLG SOLVER -- physics validation")
    print("=" * 70)

    # --- 1. single-island precession frequency -------------------------
    si = SpinIce(nx=1, ny=1)
    si.D[:] = 0.0                      # isolate: no dipolar
    si.m = np.array([[0.98, 0.0, 0.2], [0.0, 0.98, 0.2]])   # near EASY axis
    si.m /= np.linalg.norm(si.m, axis=1, keepdims=True)
    si.alpha = 0.0; si.gp = GAMMA
    dt = 1e-13
    tr = si.run(120000, dt, lambda n: np.zeros(3), sample_every=1)
    sig = tr[:, 2] - tr[:, 2].mean()
    F = np.abs(np.fft.rfft(sig * np.hanning(len(sig))))
    fr = np.fft.rfftfreq(len(sig), dt)
    f_meas = fr[np.argmax(F)]
    print(f"\n1. undamped precession about easy axis")
    print(f"   measured  {f_meas/1e9:8.3f} GHz")
    print(f"   predicted {GAMMA*si.Bk/(2*np.pi)/1e9:8.3f} GHz   (gamma*Bk/2pi)")

    # --- 2. damping relaxes m onto the easy axis -----------------------
    si = SpinIce(nx=2, ny=2, alpha=0.1)
    e0 = si.energy()
    si.run(60000, 1e-13, lambda n: np.zeros(3))
    proj = np.abs(np.sum(si.m * si.axis, axis=1))
    print(f"\n2. relaxation with alpha=0.1")
    print(f"   energy {e0:+.4e} -> {si.energy():+.4e} T   (must decrease)")
    print(f"   min |m.easy_axis| over islands: {proj.min():.4f}  (1 = aligned)")

    # --- 3. dipolar field magnitude ------------------------------------
    si = SpinIce(nx=4, ny=4)
    Bd = (si.D @ si.m.reshape(-1)).reshape(si.N, 3)
    print(f"\n3. dipolar coupling scale")
    print(f"   |B_dip| mean {np.linalg.norm(Bd,axis=1).mean()*1e3:6.3f} mT   "
          f"max {np.linalg.norm(Bd,axis=1).max()*1e3:6.3f} mT")
    print(f"   anisotropy field Bk = {si.Bk*1e3:.1f} mT")

    # --- 4. ice rule after relaxation ----------------------------------
    fr_ = []
    for s in range(6):
        si = SpinIce(nx=4, ny=4, alpha=0.1, seed=s)
        si.run(40000, 1e-13, lambda n: np.zeros(3))
        fr_.append(si.ice_rule(4, 4))
    print(f"\n4. ice rule (2-in-2-out) at interior vertices after relaxation")
    print(f"   fraction obeying: {np.mean(fr_):.2f}  (random would be 0.375)")

    # --- 5. throughput --------------------------------------------------
    si = SpinIce(nx=4, ny=4)
    t0 = time.time()
    si.run(20000, 1e-13, lambda n: np.zeros(3), sample_every=100)
    el = time.time() - t0
    print(f"\n5. speed: {20000/el:,.0f} LLG steps/s  ({si.N} islands)")
    print(f"   a 3000-sample run at 100 steps/sample = {3e5/ (20000/el):.0f}s")

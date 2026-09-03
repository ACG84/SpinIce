"""A CPU toy stand-in for the micromagnetic reservoir.

This is NOT micromagnetics.  It exists so that the whole pipeline (input
encoding -> field loops -> spectra -> features -> ridge readout) can be run
and tested in seconds on a laptop, and so that readout code can be developed
before the GPU runs finish.  Replace it with the mumax3/mumax+ outputs for
real results.

Model
-----
Every magnetic layer of every 3D island is a macrospin s = +-1 along the
island axis with its own coercive field (quenched disorder).  Layers interact
through point-dipole fields (intra-island interlayer coupling is strong and
antiferromagnetic-like, inter-island coupling weaker).  A quasi-static field
ramp flips a macrospin when the projection of (applied + dipolar) field
opposes it by more than its coercive field.  Vortex nucleation is mimicked
by a probability of entering a "vortex" state (s = 0, no net moment, weak
stray field) when a soft layer is driven through a bipolar loop close to its
coercive field; a large field annihilates it.  The "FMR spectrum" is a sum of
Lorentzians at Kittel-like frequencies set by the local effective field
(with an acoustic/optical splitting for antiparallel bilayers) plus a
low-frequency gyrotropic-like mode for vortices.
"""
from __future__ import annotations

import numpy as np

from .geometry import Island, build_islands
from .params import ASVIParams, ProtocolParams
from .tasks import field_schedule

MU0 = 4e-7 * np.pi
GAMMA = 1.760859644e11  # rad/(s T)


class MockASVI:
    def __init__(self, p: ASVIParams, proto: ProtocolParams, islands: list[Island] | None = None,
                 seed: int | None = None, bc_bottom: float = 28e-3, bc_top: float = 8e-3,
                 bc_spread: float = 0.12, vortex_prob: float = 0.25, noise: float = 0.01):
        self.p, self.proto = p, proto
        self.islands = islands or build_islands(p)
        rng = np.random.default_rng(p.seed if seed is None else seed)
        n = len(self.islands)
        self.axis = np.array([i.axis for i in self.islands])          # (n, 2)
        self.pos = np.array([[i.cx, i.cy, p.t_bottom / 2 if i.layer == 0 else
                              p.t_bottom + p.t_spacer + p.t_top / 2] for i in self.islands])
        self.layer = np.array([i.layer for i in self.islands])
        vol = np.array([i.length * i.width * (p.t_bottom if i.layer == 0 else p.t_top)
                        for i in self.islands])
        self.moment = p.msat * vol                                      # A m^2
        base = np.where(self.layer == 0, bc_bottom, bc_top)
        # thinner/narrower islands are harder: scale by (width/length) deviation
        self.bc = base * (1 + bc_spread * rng.standard_normal(n))
        self.vortex_prob = vortex_prob
        self.noise = noise
        self.rng = rng
        self.s = -np.ones(n)                                            # start saturated along -axis
        self._dipole_tensor()
        self.b_prev = 0.0

    # ----------------------------------------------------------------- physics
    def _dipole_tensor(self):
        """Field (T) at island i along its axis per unit s_j: (n, n), periodic min-image."""
        n = len(self.islands)
        Lx, Ly = self.p.box
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                d = self.pos[j] - self.pos[i]
                d[0] = (d[0] + Lx / 2) % Lx - Lx / 2
                d[1] = (d[1] + Ly / 2) % Ly - Ly / 2
                # centre-to-centre; regularise very close (same island, other layer)
                r = np.linalg.norm(d)
                r_eff = max(r, 0.5 * self.islands[i].length)
                mj = self.moment[j] * np.array([*self.axis[j], 0.0])
                rhat = d / (r + 1e-30)
                B = MU0 / (4 * np.pi * r_eff ** 3) * (3 * rhat * np.dot(mj, rhat) - mj)
                K[i, j] = np.dot(B[:2], self.axis[i])
        # same-island interlayer coupling is strongly antiparallel-favouring:
        for i in range(n):
            for j in range(n):
                if i != j and self.islands[i].index == self.islands[j].index:
                    K[i, j] = -abs(K[i, j]) * 3.0
        self.K = K

    def local_field(self, b_app_vec: np.ndarray) -> np.ndarray:
        """Axis projection of applied + dipolar field (T) for every layer-island."""
        return self.axis @ b_app_vec[:2] + self.K @ self.s

    def _step_field(self, b: float):
        cx, cy = self.proto.loop_direction
        b_vec = np.array([b * cx, b * cy, 0.0])
        for _ in range(10):                                 # iterate to self-consistency
            h = self.local_field(b_vec)
            changed = False
            for i in np.argsort(-np.abs(h)):
                if self.s[i] == 0:                           # vortex: annihilate at high field
                    if abs(h[i]) > 1.6 * self.bc[i]:
                        self.s[i] = np.sign(h[i]); changed = True
                    continue
                if -self.s[i] * h[i] > self.bc[i]:           # field opposes moment beyond coercivity
                    # top (soft) layers swept through a bipolar loop may nucleate a vortex
                    if self.layer[i] == 1 and self.rng.random() < self.vortex_prob * \
                            np.exp(-((abs(h[i]) - self.bc[i]) / (0.3 * self.bc[i])) ** 2):
                        self.s[i] = 0.0
                    else:
                        self.s[i] = np.sign(h[i])
                    changed = True
            if not changed:
                break

    def apply_ramp(self, ramp):
        for b in ramp:
            self._step_field(float(b))

    # ------------------------------------------------------------------ spectrum
    def spectrum(self, b_meas: float, freqs: np.ndarray) -> np.ndarray:
        """(nf, nregions) toy power spectrum at the measurement field."""
        p = self.p
        cx, cy = self.proto.loop_direction
        b_vec = np.array([b_meas * cx, b_meas * cy, 0.0])
        h = self.local_field(b_vec)
        power = np.zeros((len(freqs), len(self.islands)))
        for i, isl in enumerate(self.islands):
            # effective shape-anisotropy field: thick bottom layer is "harder"
            b_shape = 0.20 if isl.layer == 0 else 0.15
            if self.s[i] == 0:                               # vortex: gyrotropic-like low mode
                f0, width, amp = 1.2e9 + 2e9 * abs(h[i]) / 30e-3, 0.25e9, 0.5
                power[:, i] += amp * width ** 2 / ((freqs - f0) ** 2 + width ** 2)
                continue
            beff = self.s[i] * h[i] + b_shape                # >0 aligned, softens when opposed
            beff = max(beff, 2e-3)
            f0 = GAMMA / (2 * np.pi) * np.sqrt(beff * (beff + 0.4 * MU0 * p.msat))
            # acoustic/optical splitting for antiparallel bilayers
            partner = [j for j, o in enumerate(self.islands) if o.index == isl.index and j != i]
            if partner and self.s[partner[0]] * self.s[i] < 0:
                f0 *= 0.8 if isl.layer == 1 else 1.12
            width = 0.08e9 + 0.01 * f0
            amp = 1.0 + 0.5 * self.layer[i]
            power[:, i] += amp * width ** 2 / ((freqs - f0) ** 2 + width ** 2)
        power *= 1 + self.noise * self.rng.standard_normal(power.shape)
        return np.clip(power, 0, None)


def run_mock_reservoir(p: ASVIParams, proto: ProtocolParams, u: np.ndarray, seed: int | None = None,
                       nf: int = 512, verbose: bool = False, **kw):
    """Drive the toy reservoir through the full protocol.  Returns (freqs, power, b_loop)."""
    freqs = np.linspace(0, 1 / (2 * p.fmr_dt), nf)
    res = MockASVI(p, proto, seed=seed, **kw)
    sched = field_schedule(u, proto)
    power, b_loop = [], []
    for k, s in enumerate(sched):
        res.apply_ramp(s["ramp"])
        power.append(res.spectrum(s["b_meas"], freqs))
        b_loop.append(s["b_loop"])
        if verbose and k % 50 == 0:
            print(f"step {k}: states {res.s.astype(int).tolist()}")
    return freqs, np.stack(power), np.array(b_loop)

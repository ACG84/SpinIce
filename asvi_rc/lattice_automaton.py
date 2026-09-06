"""ASVI lattice automaton: multilayer islands with their micromagnetic state landscape, coupled
through a magnetic-charge (dumbbell) model, driven by field loops.  "flatspin with ASVI islands".

Per island i: internal state s in a catalogue S (layer labels such as ('+','V-')) with
zero-field energy E_self(s) and per-layer axis magnetisation m_axis(s) from a single-island
micromagnetic catalogue.  Each macrospin layer carries charges +-q(s) = +-Ms A m_axis at its two
ends (a vortex layer has ~0 net charge); islands interact through E_int = mu0/4pi sum q_a q_b / r_ab
over charges on different islands (intra-island coupling is already in E_self).  The external
field enters as -M(s).B.

Switching: single-layer transitions s -> s' with gain
    g = -(E_self(s') - E_self(s)) + dM.B_ext - dQ . Phi_i
where Phi_i is the potential of all other islands' charges at island i's charge points; the
transition is taken when g exceeds the coercive barrier B_c[layer] |dM_layer| (per-layer coercive
fields calibrated from hysteresis sweeps), smoothed by a sigmoid of width w (tesla) and, in the
sampled mode, drawn at random.  Cascades: rounds of simultaneous updates on a random half of
the islands until nothing switches.  Features for the readout: per-island (m_x, m_y, vortex
count) or one-hot states.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

MU0 = 4e-7 * math.pi


def load_landscape(catalogue_path):
    """Distinct relaxed states of a single-island catalogue -> (labels, E (J), m_axis (K, L), params)."""
    from .params import ASVIParams
    c = json.loads(Path(catalogue_path).read_text())
    p = ASVIParams.from_dict(c["params"])
    best = {}
    for r in c["rows"]:
        k = tuple(r["final"])
        if k not in best or r["energy_J"] < best[k]["energy_J"]:
            best[k] = r
    rows = sorted(best.values(), key=lambda r: r["energy_J"])
    labels = [tuple(r["final"]) for r in rows]
    E = np.array([r["energy_J"] for r in rows])
    m_axis = np.array([r["m_axis"] for r in rows])                     # (K, L)
    return labels, E - E.min(), m_axis, p


class ASVILattice:
    def __init__(self, catalogue, coercive, n_cells=(4, 4), lattice_constant=None, width_T=2e-3,
                 charge_pos=0.87, seed=0, pbc=False):
        """coercive: per-layer coercive fields (T) along the island axis; the angular dependence follows
        the Stoner-Wohlfarth astroid, B_c(theta) = B_c / (|cos|^(2/3) + |sin|^(2/3))^(3/2) (0.5 B_c at 45 deg).
        lattice_constant overrides the catalogue's length + 2 vertex_gap.  charge_pos: charge position
        along the axis in units of length/2 (0.87 ~ the centre of the rounded end)."""
        from .geometry import build_islands
        self.labels, self.E, self.m_axis, self.p = load_landscape(catalogue)
        self.K, self.L = self.m_axis.shape
        self.Bc = np.asarray(coercive, dtype=float)
        self.width = width_T
        p = self.p
        p.n_cells = tuple(n_cells)
        p.disorder_sigma = 0.0
        if lattice_constant is not None:
            p.vertex_gap = (lattice_constant - p.length) / 2
        self.a = p.lattice_constant
        self.rng = np.random.default_rng(seed)
        layers = build_islands(p)
        self.n = p.n_islands
        # island geometry: centre, axis, per-layer offsets
        self.centre = np.zeros((self.n, 2)); self.axis = np.zeros((self.n, 2))
        self.layer_off = np.zeros((self.n, self.L, 2))
        for isl in layers:
            if isl.layer == 0:
                self.centre[isl.index] = (isl.cx, isl.cy); self.axis[isl.index] = isl.axis
            self.layer_off[isl.index, isl.layer] = (isl.cx, isl.cy)
        area = (p.length - p.width) * p.width + math.pi * (p.width / 2) ** 2 if p.rounded_ends else p.length * p.width
        self.vol = np.array([area * t for t in p.layer_thicknesses])              # per layer
        self.cross = np.array([p.width * t for t in p.layer_thicknesses])        # end cross-section
        # per-state per-layer moments (A m^2) along the axis and charges (A m) at the ends
        self.M_layer = self.m_axis * (p.msat * self.vol)[None, :]               # (K, L)
        self.Q_layer = self.m_axis * (p.msat * self.cross)[None, :]             # (K, L), +q at +end, -q at -end
        self.Mtot = self.M_layer.sum(1)                                          # (K,) axis moment
        # charge points: island i, layer l, end e (+/-): position (n, L, 2, 2)
        d = charge_pos * p.length / 2
        self.cpos = np.zeros((self.n, self.L, 2, 2))
        for i in range(self.n):
            for l in range(self.L):
                self.cpos[i, l, 0] = self.layer_off[i, l] + d * self.axis[i]
                self.cpos[i, l, 1] = self.layer_off[i, l] - d * self.axis[i]
        # Green's matrix between charge points of different islands (open boundaries)
        P = self.cpos.reshape(self.n, self.L * 2, 2)
        self.nq = self.L * 2
        flat = P.reshape(-1, 2)
        diff = flat[:, None, :] - flat[None, :, :]
        r = np.linalg.norm(diff, axis=-1)
        G = np.where(r > 0, MU0 / (4 * math.pi) / np.maximum(r, 1e-12), 0.0)
        same = (np.arange(self.n)[:, None] == np.arange(self.n)[None, :])
        G[np.repeat(np.repeat(same, self.nq, 0), self.nq, 1)] = 0.0             # no intra-island terms
        self.G = G                                                               # (n*nq, n*nq)
        # per-state charge vector (K, nq): [+q_l, -q_l] per layer
        self.Qs = np.zeros((self.K, self.nq))
        for l in range(self.L):
            self.Qs[:, 2 * l] = self.Q_layer[:, l]; self.Qs[:, 2 * l + 1] = -self.Q_layer[:, l]
        # single-layer transitions: for each state, list of (s', layer)
        self.trans = [[(j, next(k for k in range(self.L) if a[k] != b[k]))
                       for j, b in enumerate(self.labels) if j != i and sum(x != y for x, y in zip(a, b)) == 1]
                      for i, a in enumerate(self.labels)]
        self.ground = int(np.argmin(self.E))
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self, state=None):
        self.s = np.full(self.n, self.ground if state is None else state, dtype=int)

    def potentials(self):
        """Potential of all other islands' charges at each island's charge points: (n, nq)."""
        Q = self.Qs[self.s].reshape(-1)                                          # (n*nq,)
        return (self.G @ Q).reshape(self.n, self.nq)

    # ------------------------------------------------------------------ dynamics
    def gains(self, B_ext):
        """For every island and every allowed transition: gain (J) and barrier (J).  Returns
        lists per island of (s', layer, gain, barrier)."""
        Phi = self.potentials()                                                  # (n, nq)
        B = np.asarray(B_ext[:2], dtype=float)
        Bax = self.axis @ B                                                      # field along each axis (n,)
        Bmag = np.linalg.norm(B)
        if Bmag > 0:
            c = np.abs(Bax) / Bmag; sn = np.sqrt(np.clip(1 - c ** 2, 0, 1))
            astro = 1.0 / (c ** (2 / 3) + sn ** (2 / 3)) ** 1.5                  # SW astroid factor per island
        else:
            astro = np.ones(self.n)
        out = []
        for i in range(self.n):
            s = self.s[i]
            rows = []
            for j, l in self.trans[s]:
                dE = self.E[j] - self.E[s]
                dM = self.Mtot[j] - self.Mtot[s]
                dQ = self.Qs[j] - self.Qs[s]
                g = -dE + dM * Bax[i] - dQ @ Phi[i]
                bar = self.Bc[l] * astro[i] * abs(self.M_layer[j, l] - self.M_layer[s, l])
                rows.append((j, l, g, bar))
            out.append(rows)
        return out

    def relax(self, B_ext, max_rounds=30, sample=True):
        """Cascade at fixed field; returns the number of island switches."""
        n_flips = 0
        for _ in range(max_rounds):
            gl = self.gains(B_ext)
            cand = []
            for i, rows in enumerate(gl):
                if not rows:
                    continue
                j, l, g, bar = max(rows, key=lambda r: r[2] - r[3])
                x = (g - bar) / (self.width * abs(self.M_layer[j, l] - self.M_layer[self.s[i], l]) + 1e-30)
                p = 1.0 / (1.0 + math.exp(-max(min(x, 50), -50)))
                if (self.rng.random() < p) if sample else (p > 0.5):
                    cand.append((i, j))
            if not cand:
                break
            # simultaneous update of a random half (avoids two-island oscillations)
            keep = [c for c in cand if self.rng.random() < 0.5] or cand[:1]
            for i, j in keep:
                self.s[i] = j
            n_flips += len(keep)
        return n_flips

    # ------------------------------------------------------------------ features
    def features(self, kind="moment"):
        if kind == "onehot":
            f = np.zeros((self.n, self.K)); f[np.arange(self.n), self.s] = 1.0
            return f.reshape(-1)
        m = self.Mtot[self.s] / np.abs(self.Mtot).max()                          # normalised axis moment
        vort = np.array([sum(lab.startswith("V") for lab in self.labels[s]) for s in self.s], dtype=float)
        return np.concatenate([m * self.axis[:, 0], m * self.axis[:, 1], vort])

    def state_key(self):
        return self.s.tobytes()

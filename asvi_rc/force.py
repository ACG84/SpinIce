"""FORCE learning on the field-driven ASVI automaton (closed loop, online readout).

FORCE (Sussillo & Abbott 2009): a readout z_t = w . r_t is trained online by
recursive least squares while its output is fed back into the reservoir, and
because the error is kept small from the first step the feedback never
destabilises the dynamics.  Here the feedback path is the *field amplitude*:
the write excursion of step t+1 is

    B_{t+1} = b_min + (b_max - b_min) * clip(u_{t+1} + g * z_t, 0, 1)

so the previous output reaches the next magnetic state through the loop
amplitude - a controller between the FMR readout and the field coil in an
experiment.  With g = 0 this is the open-loop reservoir with an online
readout.  The reservoir can be

* a deterministic transition table from the GPU pipeline (``TableReservoir``,
  one-hot state features, amplitudes snapped to the table grid), or
* the soft automaton of the energy landscape, sampled as a stochastic
  automaton so the reservoir is always in one state (``SoftReservoir``).

Tasks: delayed recall u(t-k), NARMA-2/10 of the input, and autonomous pattern
generation (no input, the fed-back output has to sustain the target).
"""
from __future__ import annotations

import numpy as np

from .softautomaton import transition_matrix


class RLS:
    """Recursive least squares readout with forgetting factor lam and prior 1/alpha."""

    def __init__(self, n: int, alpha: float = 1.0, lam: float = 1.0):
        self.w = np.zeros(n)
        self.P = np.eye(n) / alpha
        self.lam = lam

    def predict(self, r):
        return float(self.w @ r)

    def update(self, r, target):
        z = self.predict(r)
        e = z - target
        Pr = self.P @ r
        k = Pr / (self.lam + r @ Pr)
        self.w -= e * k
        self.P = (self.P - np.outer(k, Pr)) / self.lam
        return e


class TableReservoir:
    """Deterministic automaton from transitions.json: T[state][amplitude_mT] -> state."""

    def __init__(self, table: dict, s0: int = 0):
        self.T = {int(k): {float(a): int(v) for a, v in row.items()} for k, row in table["table"].items()}
        self.n = len(table["states"])
        self.s0 = s0
        self.keys = {s: np.array(sorted(row)) for s, row in self.T.items()}
        self.reset()

    def reset(self):
        self.s = self.s0

    def apply(self, b):
        keys = self.keys[self.s]
        self.s = self.T[self.s][float(keys[np.argmin(np.abs(keys - b * 1e3))])]

    def features(self):
        r = np.zeros(self.n + 1); r[self.s] = 1.0; r[-1] = 1.0
        return r


class SoftReservoir:
    """Stochastic automaton sampled from the soft transition matrices of the energy landscape."""

    def __init__(self, E, M, states, barrier=70e-18, width=2e-3, cascade=3, s0: int = 0, seed: int = 0):
        import torch
        self.E, self.M, self.states = E.detach(), M.detach(), states
        self.n = len(states)
        self.barrier, self.width, self.cascade = barrier, width, cascade
        self.s0 = s0
        self.rng = np.random.default_rng(seed)
        self.cache = {}
        self.torch = torch
        self.reset()

    def reset(self):
        self.s = self.s0

    def matrix(self, b):
        key = round(float(b) * 1e4)
        if key not in self.cache:
            with self.torch.no_grad():
                self.cache[key] = transition_matrix(self.E, self.M, self.states, key * 1e-4, self.barrier,
                                                    self.width, self.cascade).numpy()
        return self.cache[key]

    def apply(self, b):
        p = np.clip(self.matrix(b)[self.s], 0.0, None)          # rounding can leave -1e-16 on the diagonal
        self.s = int(self.rng.choice(self.n, p=p / p.sum()))

    def features(self):
        r = np.zeros(self.n + 1); r[self.s] = 1.0; r[-1] = 1.0
        return r


class FlatspinReservoir:
    """flatspin macrospin lattice (Stoner-Wohlfarth switching, dipolar coupling, disorder) driven
    by field loops along a fixed angle; fields are given in tesla like the other reservoirs and
    the state feature is the spin vector (+-1) plus a bias term."""

    def __init__(self, size=(8, 8), alpha=0.005, disorder=0.05, hc=0.03, angle_deg=45.0, seed=0, model="square"):
        import flatspin.model as fm
        cls = {"square": fm.SquareSpinIceClosed, "pinwheel": fm.PinwheelSpinIceDiamond, "kagome": fm.KagomeSpinIce}[model]
        self.m = cls(size=tuple(size), alpha=alpha, disorder=disorder, hc=hc, use_opencl=0, random_seed=seed)
        self.n = int(self.m.spin_count)
        self.d = np.array([np.cos(np.radians(angle_deg)), np.sin(np.radians(angle_deg))])
        self.reset()

    def reset(self):
        self.m.polarize()
        self.s = 0

    def apply(self, b):
        self.m.set_h_ext(list(b * self.d)); self.m.relax()
        self.m.set_h_ext([0.0, 0.0]); self.m.relax()
        self.s = hash(self.m.spin.tobytes())

    def features(self):
        return np.concatenate([self.m.spin.astype(float), [1.0]])


def closed_loop(res, u, target, protocol, gain: float, train_steps: int, alpha: float = 1.0,
                seed: int = 0, washout: int = 20):
    """Run FORCE on a reservoir. protocol = dict(b_min, b_max, leak, jitter, bias) in tesla.

    Returns dict(z, target, train_nrmse, test_nrmse, test_r2, states)."""
    rng = np.random.default_rng(seed)
    res.reset()
    rls = RLS(res.n + 1, alpha=alpha)
    z_prev = 0.0
    zs, states = [], []
    T = len(u)
    for t in range(T):
        drive = float(np.clip(u[t] + gain * z_prev, 0.0, 1.0))
        res.apply(-(protocol["leak"] + protocol["jitter"] * rng.uniform(-1, 1)))
        res.apply(protocol["b_min"] + (protocol["b_max"] - protocol["b_min"]) * drive)
        res.apply(protocol["bias"])
        r = res.features()
        if t < train_steps:
            rls.update(r, target[t])            # FORCE: update before the output is used
        z = rls.predict(r)
        zs.append(z); states.append(res.s)
        z_prev = z
    z = np.array(zs)
    def nrmse(a, b):
        return float(np.sqrt(np.mean((a - b) ** 2)) / (np.std(b) + 1e-12))
    tr = slice(washout, train_steps); te = slice(train_steps, T)
    r2 = 1 - np.sum((z[te] - target[te]) ** 2) / (np.sum((target[te] - target[te].mean()) ** 2) + 1e-12)
    return {"z": z, "target": np.asarray(target), "train_nrmse": nrmse(z[tr], target[tr]),
            "test_nrmse": nrmse(z[te], target[te]), "test_r2": float(r2), "states": np.array(states),
            "n_visited": len(set(states[train_steps:]))}


def make_target(task: str, u: np.ndarray, k: int = 1):
    from .tasks import narma
    if task == "recall":
        y = np.roll(u, k); y[:k] = 0.0
        return y
    if task.startswith("narma"):
        order = int(task[5:]) if len(task) > 5 else 10
        return narma(u * 0.5, order=order)             # scaled input keeps NARMA-10 bounded
    if task == "sine":
        return 0.5 + 0.4 * np.sin(2 * np.pi * np.arange(len(u)) / 12)
    raise ValueError(task)

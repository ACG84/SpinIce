"""Differentiable memory-curve proxy: a soft automaton on the energy landscape.

The GPU transition tables showed that the reservoir *is* a finite automaton on
the relaxed states, driven by the field-loop amplitude.  This module builds a
smooth surrogate of that automaton from quantities the differentiable
micromagnetic model already provides (state energies E_i and axis moments M_i
of the relaxed states), so the linear memory curve of the reservoir becomes a
differentiable function of the design parameters.

Switching rule (single-layer events between "neighbour" states i -> j that
differ in one layer label): under a field B along the island axis the Zeeman
gain of the transition is dM*B - dE (dE = E_j - E_i, dM = M_j - M_i); the
transition happens with probability

    p_ij(B) = sigmoid((dM*B - dE - barrier) / (|dM| * width))

i.e. it needs the field-driven gain to exceed an effective barrier (calibrated
so that the top layer of the nominal island switches at ~30 mT), with a
switching-field width `width` (disorder / thermal smearing, ~2 mT).  Metastable
states therefore persist at zero field, and every reordering field of the
level analysis maps to a switching field.  A few cascade rounds at each field
allow sequential switching within one excursion.

Reservoir: the state probability vector rho_t is propagated through the field
stages of each protocol step (e.g. leak excursion, write excursion, bias);
features are rho_t at the bias field; the memory curve is the test R^2 of a
ridge readout of u(t-k) from rho_t, and MC = sum_k R^2(k).
"""
from __future__ import annotations

import numpy as np


def neighbours(states):
    """Pairs (i, j) of states that differ in exactly one layer label."""
    out = []
    for i, a in enumerate(states):
        for j, b in enumerate(states):
            if i != j and sum(x != y for x, y in zip(a, b)) == 1:
                out.append((i, j))
    return out


def transition_matrix(E, M, states, B, barrier, width, cascade=3, pairs=None):
    """(N, N) row-stochastic transition matrix at field B (scalar torch or float, tesla)."""
    import torch
    n = len(states)
    pairs = neighbours(states) if pairs is None else pairs
    ii = torch.tensor([i for i, _ in pairs]); jj = torch.tensor([j for _, j in pairs])
    dE = E[jj] - E[ii]
    dM = M[jj] - M[ii]
    p = torch.sigmoid((dM * B - dE - barrier) / (dM.abs() * width + 1e-30))
    P = torch.zeros(n, n, dtype=E.dtype).index_put((ii, jj), p)
    row = P.sum(dim=1)
    P = P / torch.clamp(row, min=1.0)[:, None]                  # competing switches share the probability
    P = P + torch.diag(1.0 - P.sum(dim=1))
    Pk = P
    for _ in range(cascade - 1):
        Pk = Pk @ P
    return Pk


def simulate(E, M, states, u, stages, barrier, width, cascade=3, rho0=None, pairs=None):
    """Propagate rho_t through the protocol.

    stages : list of callables f(u_t, r_t) -> field (torch scalar); r_t is an
             independent uniform random number per step (for the leak jitter).
    Returns rho (T, N) after the last stage of every step.
    """
    import torch
    n = len(states)
    pairs = neighbours(states) if pairs is None else pairs
    rho = torch.full((n,), 1.0 / n, dtype=E.dtype) if rho0 is None else rho0
    rng = np.random.default_rng(1)
    r = rng.random(len(u))
    out = []
    cache = {}
    for t, ut in enumerate(u):
        for f in stages:
            B = f(float(ut), float(r[t]))
            if torch.is_tensor(B) and B.requires_grad:               # protocol parameters being optimised
                rho = rho @ transition_matrix(E, M, states, B, barrier, width, cascade, pairs)
                continue
            key = round(float(B) * 1e4)                              # 0.1 mT input quantisation (cache key)
            if key not in cache:
                cache[key] = transition_matrix(E, M, states, key * 1e-4, barrier, width, cascade, pairs)
            rho = rho @ cache[key]
        out.append(rho)
    return torch.stack(out)


def memory_curve(rho, u, k_max=8, alpha=1e-3, train_frac=0.6, washout=50):
    """Test R^2 of a ridge readout of u(t-k) from rho_t for k = 0..k_max (torch)."""
    import torch
    T = rho.shape[0]
    u = torch.as_tensor(np.asarray(u), dtype=rho.dtype)
    X = torch.cat([rho, torch.ones(T, 1, dtype=rho.dtype)], dim=1)
    r2 = []
    for k in range(k_max + 1):
        idx = torch.arange(washout + k, T)
        Xk, yk = X[idx], u[idx - k]
        n_tr = int(len(idx) * train_frac)
        Xtr, ytr, Xte, yte = Xk[:n_tr], yk[:n_tr], Xk[n_tr:], yk[n_tr:]
        A = Xtr.T @ Xtr + alpha * torch.eye(X.shape[1], dtype=rho.dtype)
        w = torch.linalg.solve(A, Xtr.T @ ytr)
        res = ((yte - Xte @ w) ** 2).sum()
        tot = ((yte - yte.mean()) ** 2).sum()
        r2.append(1 - res / tot)
    r2 = torch.stack(r2)
    # smooth clamp at 0 (softplus with slope 20) so a negative-R^2 lag still gives a gradient
    return r2, (torch.nn.functional.softplus(20 * r2) / 20).sum()


def leak_stages(b_min, b_max, leak, leak_jitter, bias):
    """Field stages of the random-leak protocol (all along the island axis)."""
    return [lambda ut, rt: -(leak + leak_jitter * (2 * rt - 1)),
            lambda ut, rt: b_min + (b_max - b_min) * ut,
            lambda ut, rt: bias]


def unipolar_stages(b_min, b_max, bias):
    return [lambda ut, rt: b_min + (b_max - b_min) * ut, lambda ut, rt: bias]


def memory_proxy(E, M, states, n_steps=400, seed=0, stages=None, barrier=70e-18, width=2e-3,
                 cascade=3, k_max=8, alpha=1e-3):
    """MC and R^2(k) of the soft automaton for a uniform random input (torch scalars)."""
    rng = np.random.default_rng(seed)
    u = rng.random(n_steps)
    stages = leak_stages(28e-3, 38e-3, 33e-3, 5e-3, -20e-3) if stages is None else stages
    rho = simulate(E, M, states, u, stages, barrier, width, cascade)
    r2, mc = memory_curve(rho, u, k_max=k_max, alpha=alpha)
    return mc, r2, rho

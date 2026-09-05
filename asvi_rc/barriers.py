"""Energy barriers between ASVI states with a simplified string method (magnum.np backend).

The saddle configuration returned here is stationary, so, exactly like the
minima, d(E_saddle)/d(design) = dE/d(design) at fixed m (envelope theorem):
``ASVISimulationNP.energy_grad`` at the saddle gives the design gradient of the
barrier, and barrier - level combinations become differentiable objectives.

Algorithm (E, Ren & Vanden-Eijnden 2007): every interior image takes a few
Barzilai-Borwein semi-implicit descent steps (magnum.np's minimiser update),
then the string is re-parametrised to equal arc length in m.  The endpoints
are the two relaxed states and stay fixed.
"""
from __future__ import annotations

import time

import numpy as np


def _to_t(sim, m_np):
    torch = sim.torch
    t = torch.from_numpy(np.ascontiguousarray(m_np.transpose(3, 2, 1, 0), dtype=np.float64)).to(sim.device)
    return t


def _normalize(sim, m):
    torch = sim.torch
    n = torch.linalg.norm(m, dim=-1, keepdim=True)
    m = torch.where(n > 0, m / n.clamp(min=1e-30), torch.zeros_like(m))
    m[~sim._inside()] = 0.0
    return m


def _bb_steps(mn, state, m, k, tau, tau_min=1e-13, tau_max=1e-5):
    """k Barzilai-Borwein midpoint steps from m; returns (m, tau, |torque|_max)."""
    import torch
    state.m = m
    h = mn.h(state)
    dm = torch.linalg.cross(m, torch.linalg.cross(m, h))
    for i in range(k):
        m_new = mn._midpoint(m, h, tau)
        state.m = m_new
        h_new = mn.h(state)
        dm_new = torch.linalg.cross(m_new, torch.linalg.cross(m_new, h_new))
        s, y = m_new - m, dm_new - dm
        num, den = ((s * s).sum(), (s * y).sum()) if i % 2 == 0 else ((s * y).sum(), (y * y).sum())
        tau = float(min(max(abs(num / den) if den != 0 else tau_max, tau_min), tau_max))
        m, h, dm = m_new, h_new, dm_new
    return m, tau, float(dm.abs().max())


def _reparametrise(sim, images):
    """Equal arc-length re-distribution of the images (linear interpolation in m, renormalised)."""
    torch = sim.torch
    n = len(images)
    d = torch.tensor([0.0] + [float(torch.linalg.norm(images[i] - images[i - 1])) for i in range(1, n)])
    s = torch.cumsum(d, 0)
    if float(s[-1]) == 0:
        return images
    targets = torch.linspace(0, float(s[-1]), n)
    out = [images[0]]
    j = 1
    for k in range(1, n - 1):
        t = float(targets[k])
        while j < n - 1 and float(s[j]) < t:
            j += 1
        w = (t - float(s[j - 1])) / max(float(s[j] - s[j - 1]), 1e-30)
        out.append(_normalize(sim, (1 - w) * images[j - 1] + w * images[j]))
    out.append(images[-1])
    return out


def string_barrier(sim, m_a: np.ndarray, m_b: np.ndarray, n_images: int = 16, n_iter: int = 80,
                   sub_steps: int = 4, tol_J: float = 2e-21, verbose: bool = False) -> dict:
    """Minimum-energy path between two relaxed states m_a, m_b (numpy (3, nz, ny, nx)).

    Returns dict(energies (J, per image), images (numpy), i_saddle, m_saddle,
    barrier_J = E_saddle - E_a, barrier_back_J = E_saddle - E_b, iterations, seconds).
    """
    torch, mnp = sim.torch, sim.mnp
    mn = mnp.MinimizerBB(sim.terms)
    state = sim.state
    t0 = time.time()
    with torch.no_grad():
        a, b = _normalize(sim, _to_t(sim, m_a)), _normalize(sim, _to_t(sim, m_b))
        images = [_normalize(sim, (1 - w) * a + w * b) for w in np.linspace(0, 1, n_images)]
        images = _reparametrise(sim, images)
        taus = [1e-13] * n_images
        prev = None
        stable = 0
        for it in range(n_iter):
            for i in range(1, n_images - 1):
                images[i], taus[i], _ = _bb_steps(mn, state, images[i], sub_steps, taus[i])
            images = _reparametrise(sim, images)
            E = []
            for img in images:
                state.m = img
                E.append(float(mn.E(state)))
            E = np.array(E)
            change = np.abs(E - prev).max() if prev is not None else np.inf
            prev = E
            if verbose:
                print(f"  string it {it:3d}: barrier {(E.max() - E[0]) * 1e18:8.3f} aJ  max dE {change * 1e18:.4f} aJ", flush=True)
            stable = stable + 1 if change < tol_J else 0
            if stable >= 3:
                break
        i_s = int(np.argmax(E))
        state.m = images[i_s]
    imgs_np = [img.detach().cpu().numpy().transpose(3, 2, 1, 0).copy() for img in images]
    return {"energies": E, "images": imgs_np, "i_saddle": i_s, "m_saddle": imgs_np[i_s],
            "barrier_J": float(E[i_s] - E[0]), "barrier_back_J": float(E[i_s] - E[-1]),
            "iterations": it + 1, "seconds": time.time() - t0}

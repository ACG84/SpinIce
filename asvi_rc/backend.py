"""Choose the micromagnetic backend: mumax+ (CUDA) or magnum.np (PyTorch, CPU/GPU, differentiable)."""
from __future__ import annotations

BACKENDS = ("mumaxplus", "magnumnp")


def make_simulation(p, islands=None, backend: str = "mumaxplus", **kw):
    if backend == "magnumnp":
        from .magnumnp_driver import ASVISimulationNP
        return ASVISimulationNP(p, islands=islands, **kw)
    if backend == "mumaxplus":
        from .mumaxplus_driver import ASVISimulation
        return ASVISimulation(p, islands=islands, **kw)
    raise ValueError(f"unknown backend {backend!r}; choose from {BACKENDS}")

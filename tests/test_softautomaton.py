"""Soft automaton memory proxy: stochastic matrix, finite MC, gradient flow (pure torch, fast)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from asvi_rc.softautomaton import neighbours, transition_matrix, memory_proxy, unipolar_stages  # noqa: E402

STATES = [("+", "-"), ("-", "+"), ("+", "+"), ("-", "-"), ("+", "V+"), ("-", "V-"), ("V+", "+"), ("V-", "-")]
E = torch.tensor([0, 0, 27.6, 27.6, 26.4, 26.4, 36.8, 36.8], dtype=torch.float64) * 1e-18
M = torch.tensor([0.57, -0.57, 2.75, -2.75, 1.54, -1.54, 0.99, -0.99], dtype=torch.float64) * 1e-15


def test_neighbours_single_layer_changes():
    pairs = neighbours(STATES)
    assert (0, 2) in pairs and (0, 4) in pairs and (0, 1) not in pairs


def test_transition_matrix_is_row_stochastic_and_field_dependent():
    P0 = transition_matrix(E, M, STATES, 0.0, 70e-18, 2e-3)
    P1 = transition_matrix(E, M, STATES, 60e-3, 70e-18, 2e-3)
    assert torch.allclose(P0.sum(1), torch.ones(8, dtype=torch.float64))
    assert torch.allclose(P1.sum(1), torch.ones(8, dtype=torch.float64))
    assert float(P0[1, 2]) < 1e-3                      # metastable at zero field
    assert float(P1[1, 2]) > 0.5                       # -/+ -> +/+ switches at 60 mT


def test_memory_proxy_gradient_flows():
    e = E.clone().requires_grad_(True)
    m = M.clone().requires_grad_(True)
    mc, r2, rho = memory_proxy(e, m, STATES, n_steps=150, stages=unipolar_stages(20e-3, 40e-3, -20e-3),
                               barrier=70e-18, width=2e-3, k_max=3)
    assert torch.isfinite(mc) and 0 <= float(mc) <= 4
    g = torch.autograd.grad(mc, [e, m])
    assert all(torch.isfinite(x).all() for x in g)
    assert float(g[0].abs().sum()) > 0
    assert torch.allclose(rho.sum(1), torch.ones(150, dtype=torch.float64))

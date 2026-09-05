"""magnum.np backend: hard-mask energies, soft-mask consistency and design gradients (CPU).

Uses a coarse 20 nm island so the whole file runs in well under a minute; skipped
when torch/magnumnp are not installed.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("magnumnp")

from asvi_rc import ASVIParams                                                  # noqa: E402
from asvi_rc.geometry import single_island, macrospin_magnetization             # noqa: E402
from asvi_rc.magnumnp_driver import ASVISimulationNP                            # noqa: E402


@pytest.fixture(scope="module")
def coarse():
    p = ASVIParams(cell_xy=20e-9, cell_z=10e-9, t_bottom=30e-9, t_spacer=30e-9, t_top=20e-9,
                   disorder_sigma=0.0, box_override=(640e-9, 320e-9), pbc_repetitions=(0, 0, 0))
    return p, single_island(p)


def relax(sim, p, signs):
    sim.set_magnetization(macrospin_magnetization(p, sim.regions, sim.islands, signs, tilt=0.02))
    sim.minimize()
    return sim.total_energy()


def test_hard_mask_antiparallel_ground_state(coarse):
    p, isl = coarse
    sim = ASVISimulationNP(p, isl, verbose=False)
    e_ap = relax(sim, p, {1: 1.0, 2: -1.0})
    e_p = relax(sim, p, {1: 1.0, 2: 1.0})
    assert e_p - e_ap > 5e-18                       # parallel stack costs tens of aJ
    avg = sim.region_averages()
    assert avg[0, 0] > 0.8 and avg[1, 0] > 0.8      # stays a parallel macrospin pair


def test_soft_mask_matches_hard_mask(coarse):
    p, isl = coarse
    hard = ASVISimulationNP(p, isl, verbose=False)
    design = {"width": torch.tensor(p.width, dtype=torch.float64, requires_grad=True),
              "t_spacer": torch.tensor(p.t_spacer, dtype=torch.float64, requires_grad=True)}
    soft = ASVISimulationNP(p, isl, verbose=False, design=design)
    d_hard = relax(hard, p, {1: 1.0, 2: 1.0}) - relax(hard, p, {1: 1.0, 2: -1.0})
    d_soft = relax(soft, p, {1: 1.0, 2: 1.0}) - relax(soft, p, {1: 1.0, 2: -1.0})
    assert abs(d_soft - d_hard) < 0.2 * d_hard      # boundary cells only


def test_design_gradient_matches_finite_difference(coarse):
    p, isl = coarse

    def splitting(width):
        design = {"width": torch.tensor(width, dtype=torch.float64, requires_grad=True)}
        sim = ASVISimulationNP(p, isl, verbose=False, design=design)
        e, g = {}, {}
        for lab, signs in (("ap", {1: 1.0, 2: -1.0}), ("p", {1: 1.0, 2: 1.0})):
            e[lab] = relax(sim, p, signs)
            g[lab] = sim.energy_grad()["width"]
        return e["p"] - e["ap"], g["p"] - g["ap"]

    d0, grad = splitting(p.width)
    assert np.isfinite(grad)
    h = 2e-9
    fd = (splitting(p.width + h)[0] - splitting(p.width - h)[0]) / (2 * h)
    assert abs(grad - fd) < 0.15 * abs(fd) + 1e-12


def test_string_barrier_runs_and_is_positive(coarse):
    from asvi_rc.barriers import string_barrier
    p, isl = coarse
    sim = ASVISimulationNP(p, isl, verbose=False)
    sim.set_magnetization(macrospin_magnetization(p, sim.regions, sim.islands, {1: 1.0, 2: -1.0}, tilt=0.02))
    sim.minimize(); m_ap = sim.get_magnetization()
    sim.set_magnetization(macrospin_magnetization(p, sim.regions, sim.islands, {1: 1.0, 2: 1.0}, tilt=0.02))
    sim.minimize(); m_p = sim.get_magnetization()
    res = string_barrier(sim, m_ap, m_p, n_images=6, n_iter=8, n_climb=4)
    assert res["barrier_J"] > 0 and res["barrier_back_J"] >= -1e-21
    assert 0 < res["i_saddle"] < 5
    assert res["m_saddle"].shape == m_ap.shape

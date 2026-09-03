"""mumax3 script generation and a dry run of the mumax+ driver on a fake API."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from asvi_rc import ASVIParams, ProtocolParams, tasks
from asvi_rc.geometry import build_islands
from asvi_rc.mumax3_writer import write_reservoir_script, write_field_sweep_script
from asvi_rc.mock_reservoir import run_mock_reservoir
from asvi_rc.spectra import reservoir_features
from asvi_rc import readout


def small():
    return ASVIParams(n_cells=(1, 1), cell_xy=10e-9, cell_z=5e-9)


def test_mumax3_reservoir_script(tmp_path):
    p, proto = small(), ProtocolParams(loop_step=5e-3)
    u = np.array([0.0, 0.5, 1.0])
    script = write_reservoir_script(p, proto, u, tmp_path)
    txt = script.read_text()
    assert txt.count("{") == txt.count("}")
    assert "SetGridSize(80, 80, 17)" in txt
    assert "SetPBC(4, 4, 0)" in txt
    assert txt.count("DefRegion(") == 4 and "TableAdd(m.Region(4))" in txt
    assert txt.count("TableAutoSave(") == 2 * len(u)   # start + stop per step
    assert txt.count("Run(2.6e-08)") == len(u)
    assert "sinc(2*pi*1.5e+10*(t-5e-10))" in txt
    assert txt.count("Repeat(8e-07, 8e-07, 0)") == 4
    meta = json.loads((tmp_path / "asvi_reservoir.meta.json").read_text())
    assert meta["region_ids"] == [1, 2, 3, 4] and len(meta["b_loop"]) == 3
    assert meta["b_loop"][0] == pytest.approx(proto.b_min)


def test_mumax3_sweep_script(tmp_path):
    p = small()
    script = write_field_sweep_script(p, -10e-3, 10e-3, 5e-3, 1.0, tmp_path)
    txt = script.read_text()
    assert txt.count("Run(") == 5 and txt.count("SaveAs(m, \"m_field") == 5
    meta = json.loads((tmp_path / "asvi_field_sweep.meta.json").read_text())
    assert len(meta["fields"]) == 5


def test_mumaxplus_driver_dry_run(monkeypatch):
    fake = Path(__file__).parent / "fake_mumaxplus"
    monkeypatch.syspath_prepend(str(fake))
    for mod in [m for m in sys.modules if m.startswith("mumaxplus")]:
        del sys.modules[mod]
    from asvi_rc.mumaxplus_driver import ASVISimulation

    p = small()
    p.fmr_duration, p.fmr_dt = 2e-9, 50e-12
    proto = ProtocolParams(loop_step=10e-3)
    sim = ASVISimulation(p, verbose=False)
    assert sim.magnet.geometry.sum() == sim.mask.sum()
    sim.saturate(proto)
    m = sim.get_magnetization()
    assert m[0][sim.mask].mean() < -0.99
    sched = tasks.field_schedule(np.array([0.2, 0.9]), proto)
    n0 = sim.world.n_minimize
    sim.apply_ramp(sched[0]["ramp"], proto)
    assert sim.world.n_minimize - n0 == len(sched[0]["ramp"])
    freqs, P, m_t = sim.fmr(tasks.field_vector(sched[0]["b_meas"], proto), return_timeseries=True)
    assert m_t.shape == (p.fmr_nt, 4, 3) and P.shape == (len(freqs), 4)
    assert freqs[P[1:, 0].argmax() + 1] == pytest.approx(7e9, rel=0.15)   # fake dynamics at 7 GHz
    assert sim.magnet.bias_magnetic_field.time_terms == []                 # excitation removed
    assert sim.world.timesolver.time == pytest.approx(p.fmr_duration)


def test_mock_pipeline_end_to_end():
    p, proto = ASVIParams(n_cells=(2, 2)), ProtocolParams(loop_step=2e-3)
    u, y = tasks.make_task("sine_to_square", 160)
    freqs, power, b_loop = run_mock_reservoir(p, proto, u, seed=1, nf=256)
    assert power.shape == (160, 256, 16) and np.all(power >= 0)
    X = reservoir_features(freqs, power, proto)
    r = readout.evaluate(X, y, n_train=70, alpha=1e-2, washout=10, u=u)
    assert np.isfinite(r["mse_test"])

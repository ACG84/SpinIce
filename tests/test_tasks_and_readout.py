import numpy as np
import pytest

from asvi_rc import ProtocolParams, tasks, readout, spectra


def test_all_tasks_finite():
    for name in tasks.TASKS:
        u, y = tasks.make_task(name, 300)
        assert u.shape == y.shape == (300,)
        assert np.all(np.isfinite(y)) and u.min() == 0 and u.max() == 1


def test_mackey_glass_is_chaotic_and_bounded():
    x = tasks.mackey_glass(2000)
    assert 0.2 < x.min() and x.max() < 1.6
    assert np.std(x) > 0.1


def test_minor_loop_shapes():
    pr = ProtocolParams(b_min=20e-3, b_max=30e-3, loop_step=1e-3)
    ramp = tasks.minor_loop(25e-3, pr, b_start=20e-3)
    assert ramp[-1] == pytest.approx(25e-3)
    assert ramp.min() == pytest.approx(-25e-3)
    assert np.all(np.abs(np.diff(ramp)) <= 1e-3 + 1e-12)
    pr.measure_at = "bias"
    ramp = tasks.minor_loop(25e-3, pr, b_start=-1.2e-3)
    assert ramp[-1] == pytest.approx(-1.2e-3)
    pr.loop_shape = "unipolar"
    ramp = tasks.minor_loop(25e-3, pr, b_start=-1.2e-3)
    assert ramp.max() == pytest.approx(25e-3) and ramp.min() > -1.3e-3


def test_adaptive_loop_stepping():
    pr = ProtocolParams(loop_step=2e-3, coarse_step=5e-3, coarse_below=22e-3)
    ramp = tasks.minor_loop(34e-3, pr, b_start=34e-3)
    fine = ramp[np.abs(ramp) >= 22e-3 - 1e-9]
    assert ramp[-1] == pytest.approx(34e-3) and ramp.min() == pytest.approx(-34e-3)
    assert np.all(np.abs(np.diff(ramp)) <= 5e-3 + 1e-9)
    # fine region visited in 2 mT steps, coarse region in larger ones
    assert np.any(np.isclose(fine, 24e-3)) and np.any(np.isclose(fine, -26e-3))
    coarse = ramp[np.abs(ramp) <= 21e-3]
    assert len(coarse) == 4 * 4     # 4 passes through |B|<22 mT in ~5 mT steps
    plain = tasks.minor_loop(34e-3, ProtocolParams(loop_step=2e-3), b_start=34e-3)
    assert len(ramp) < len(plain)


def test_alternating_protocol():
    pr = ProtocolParams(loop_shape="alternating", measure_at="bias", bias_field=-1.2e-3, loop_step=2e-3)
    u = np.array([0.0, 1.0, 0.5, 1.0])
    sched = tasks.field_schedule(u, pr)
    signs = [s["sign"] for s in sched]
    assert signs == [1.0, -1.0, 1.0, -1.0]
    for s in sched:
        assert s["ramp"][-1] == pytest.approx(-1.2e-3)                # measured at the bias field
        assert np.max(np.abs(s["ramp"])) == pytest.approx(s["b_loop"]) or s is sched[0]
        assert np.sign(s["ramp"][np.argmax(np.abs(s["ramp"]))]) == s["sign"] or s is sched[0]
    # ramps are contiguous: each starts near the previous measurement field
    for a, b in zip(sched[:-1], sched[1:]):
        assert abs(b["ramp"][0] - a["b_meas"]) <= pr.loop_step + 1e-9
    pr2 = ProtocolParams(loop_shape="alternating", measure_at="loop_max")
    s2 = tasks.field_schedule(u, pr2)
    assert s2[1]["b_meas"] == pytest.approx(-pr2.b_max)


def test_leak_protocol():
    pr = ProtocolParams(loop_shape="leak", leak_field=30e-3, bias_field=-1.2e-3, loop_step=2e-3)
    sched = tasks.field_schedule(np.array([0.3, 1.0, 0.0]), pr)
    for s in sched:
        assert s["b_meas"] == pytest.approx(-1.2e-3) and s["ramp"][-1] == pytest.approx(-1.2e-3)
        assert s["ramp"].max() == pytest.approx(s["b_loop"])
    for s in sched[1:]:                      # step 0 starts with the approach from -B_sat
        assert s["ramp"].min() == pytest.approx(-30e-3)
        # order: up to +B, down through -leak, back to bias
        i_max, i_min = np.argmax(s["ramp"]), np.argmin(s["ramp"])
        assert i_max < i_min < len(s["ramp"]) - 1


def test_field_schedule_continuity():
    pr = ProtocolParams()
    u = np.linspace(0, 1, 5)
    sched = tasks.field_schedule(u, pr)
    # coarse approach from -B_sat, then the fine loop; ends at the measurement field
    assert sched[0]["ramp"][0] == pytest.approx(-pr.saturation_field + pr.approach_step, abs=1e-6)
    assert np.all(np.abs(np.diff(sched[0]["ramp"])) <= pr.approach_step + 1e-12)
    assert sched[0]["ramp"][-1] == pytest.approx(sched[0]["b_meas"])
    assert sched[0]["b_loop"] == pytest.approx(pr.b_min) and sched[-1]["b_loop"] == pytest.approx(pr.b_max)
    v = tasks.field_vector(1.0, pr)
    assert v[0] == pytest.approx(np.cos(np.radians(1))) and v[2] == 0


def test_spectrum_peak_and_features():
    dt, nt = 33e-12, 789
    t = np.arange(nt) * dt
    m = np.zeros((nt, 2, 3))
    m[:, 0, 2] = 0.01 * np.sin(2 * np.pi * 5e9 * t)
    m[:, 1, 2] = 0.5 + 0.01 * np.sin(2 * np.pi * 9e9 * t)
    f, P = spectra.power_spectrum(m, dt)
    assert f[P[:, 0].argmax()] == pytest.approx(5e9, rel=0.02)
    assert f[P[:, 1].argmax()] == pytest.approx(9e9, rel=0.02)
    c, v = spectra.features(f, P, 2e9, 14e9, 40e6)
    assert len(c) == 300 and v.shape == (300,)
    c, v = spectra.features(f, P, 2e9, 14e9, 40e6, per_region=True)
    assert v.shape == (600,)


def test_ridge_recovers_linear_map():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 5))
    w = np.array([1.0, -2.0, 0.5, 0.0, 3.0])
    y = X @ w + 0.7
    r = readout.evaluate(X, y, n_train=100, alpha=1e-6, u=X[:, 0])
    assert r["mse_test"] < 1e-8
    assert readout.nrmse(y, r["y_pred"]) < 1e-3
    assert r["mse_test_baseline"] > r["mse_test"]
    a = readout.select_alpha(X, y, 100)
    assert a > 0


def test_mumax3_table_roundtrip(tmp_path):
    names = ["t (s)", "m.region1x ()", "m.region1y ()", "m.region1z ()", "step ()", "B_loop (T)"]
    rows = []
    dt = 33e-12
    for s in range(2):
        for i in range(64):
            rows.append([i * dt, 1.0, 0.0, 0.01 * np.sin(2 * np.pi * 6e9 * i * dt), s, 0.02 + 0.005 * s])
    path = tmp_path / "table.txt"
    with open(path, "w") as f:
        f.write("# " + "\t".join(names) + "\n")
        for r in rows:
            f.write("\t".join(f"{x:g}" for x in r) + "\n")
    tab = spectra.read_mumax3_table(path)
    assert set(tab) == {"t", "m.region1x", "m.region1y", "m.region1z", "step", "B_loop"}
    f, P, steps, b = spectra.spectra_from_mumax3_table(path, [1], dt)
    assert P.shape == (2, 33, 1) and list(steps) == [0, 1] and b[1] == pytest.approx(0.025)
    assert f[P[0, 1:, 0].argmax() + 1] == pytest.approx(6e9, rel=0.1)

#!/usr/bin/env python3
"""End-to-end demo on the CPU toy reservoir (no GPU needed).

    python scripts/demo_mock.py --task sine_to_square --n 600 --out runs/mock_demo

Writes spectra.npz in the same format as the micromagnetic runners and a
figure with the input, the spectral fingerprints and the readout results.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asvi_rc import ASVIParams, ProtocolParams, tasks, readout        # noqa: E402
from asvi_rc.mock_reservoir import run_mock_reservoir                 # noqa: E402
from asvi_rc.spectra import save_spectra, reservoir_features          # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", default="sine_to_square", choices=list(tasks.TASKS))
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--n-cells", type=int, nargs=2, default=(3, 3))
    ap.add_argument("--b-min", type=float, default=20e-3)
    ap.add_argument("--b-max", type=float, default=30e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/mock_demo")
    a = ap.parse_args(argv)

    p = ASVIParams(n_cells=tuple(a.n_cells), seed=a.seed)
    proto = ProtocolParams(b_min=a.b_min, b_max=a.b_max)
    u, y = tasks.make_task(a.task, a.n, seed=a.seed)
    freqs, power, b_loop = run_mock_reservoir(p, proto, u, seed=a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    save_spectra(out / "spectra.npz", freqs, power, b_loop, u=u, y=y, task=a.task)

    X = reservoir_features(freqs, power, proto, per_region=False)
    washout = 20
    n_train = a.n_train or (a.n - washout) // 2
    alpha = readout.select_alpha(X, y, n_train, washout=washout)
    r = readout.evaluate(X, y, n_train, alpha=alpha, washout=washout, u=u)
    print(f"task {a.task}: {X.shape[1]} outputs, alpha {alpha:g}")
    print(f"  reservoir MSE test {r['mse_test']:.3e} (NRMSE {r['nrmse_test']:.3f}); "
          f"raw-input baseline MSE {r['mse_test_baseline']:.3e}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    ax[0].plot(b_loop * 1e3, "k", lw=0.8); ax[0].set_ylabel("loop amplitude (mT)")
    S = np.log10(1 + power.sum(axis=2) / power.sum(axis=2).max() * 1e3)
    ax[1].imshow(S.T, aspect="auto", origin="lower", cmap="inferno",
                 extent=[0, len(u), freqs[0] / 1e9, freqs[-1] / 1e9])
    ax[1].set_ylim(0, 14); ax[1].set_ylabel("f (GHz)")
    te = r["test_idx"]
    ax[2].plot(y, "k", lw=1, label="target")
    ax[2].plot(te, r["y_pred"][te], "C3", lw=1, label=f"reservoir (MSE {r['mse_test']:.2e})")
    ax[2].plot(te, r["y_baseline"][te], "C0", lw=0.8, alpha=0.7, label=f"raw input (MSE {r['mse_test_baseline']:.2e})")
    ax[2].axvspan(r["train_idx"][0], r["train_idx"][-1], color="0.9")
    ax[2].legend(fontsize=8); ax[2].set_xlabel("time step"); ax[2].set_ylabel("output")
    fig.suptitle(f"toy ASVI reservoir - {a.task} (NOT micromagnetics)")
    fig.tight_layout(); fig.savefig(out / "demo.png", dpi=130)
    print("wrote", out / "demo.png")


if __name__ == "__main__":
    main()

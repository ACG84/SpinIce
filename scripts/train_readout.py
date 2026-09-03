#!/usr/bin/env python3
"""Train and evaluate the linear readout on a spectra.npz file.

    python scripts/train_readout.py runs/mg10/spectra.npz --n-train 200 [--per-region] [--plot out.png]

The file must contain ``freqs``, ``power`` (steps x nf x regions), ``u`` and
``y`` (written by mumaxplus/run_reservoir.py, process_mumax3_table.py or the
mock demo).  Ridge parameter is chosen on a validation slice of the training
data unless --alpha is given.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asvi_rc import ProtocolParams, readout                          # noqa: E402
from asvi_rc.spectra import load_spectra, reservoir_features         # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("spectra")
    ap.add_argument("--n-train", type=int, default=None, help="default: half of the data")
    ap.add_argument("--washout", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--per-region", action="store_true", help="concatenate region spectra")
    ap.add_argument("--f-min", type=float, default=2e9)
    ap.add_argument("--f-max", type=float, default=14e9)
    ap.add_argument("--f-bin", type=float, default=40e6)
    ap.add_argument("--no-log", action="store_true")
    ap.add_argument("--memory", action="store_true", help="also compute linear memory capacity")
    ap.add_argument("--plot", type=str, default=None)
    a = ap.parse_args(argv)

    d = load_spectra(a.spectra)
    u, y = d["u"], d["y"]
    if len(u) == 0:
        sys.exit("spectra file has no inputs/targets")
    proto = ProtocolParams(f_min=a.f_min, f_max=a.f_max, f_bin=a.f_bin)
    X = reservoir_features(d["freqs"], d["power"], proto, per_region=a.per_region, log=not a.no_log)
    n = len(y)
    n_train = a.n_train or (n - a.washout) // 2
    alpha = a.alpha or readout.select_alpha(X, y, n_train, washout=a.washout)
    r = readout.evaluate(X, y, n_train, alpha=alpha, washout=a.washout, u=u)
    print(f"{n} steps, {X.shape[1]} reservoir outputs, train {n_train}, test {len(r['test_idx'])}, alpha {alpha:g}")
    print(f"reservoir : MSE test {r['mse_test']:.3e}  NRMSE test {r['nrmse_test']:.3f}  (train MSE {r['mse_train']:.3e})")
    print(f"raw input : MSE test {r['mse_test_baseline']:.3e}  NRMSE test {r['nrmse_test_baseline']:.3f}")
    if a.memory:
        caps, mc = readout.memory_capacity(X, u, n_train, alpha=alpha, washout=a.washout)
        print("memory capacity", round(mc, 2), "per delay", np.round(caps, 2).tolist())
    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        te = r["test_idx"]
        fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        ax[0].plot(u, "k", lw=0.8, label="input u")
        ax[0].set_ylabel("input"); ax[0].legend(loc="upper right")
        ax[1].plot(y, "k", lw=1, label="target")
        ax[1].plot(te, r["y_pred"][te], "C3", lw=1, label=f"reservoir (test MSE {r['mse_test']:.2e})")
        ax[1].plot(te, r["y_baseline"][te], "C0", lw=0.8, alpha=0.7,
                   label=f"raw-input regression (MSE {r['mse_test_baseline']:.2e})")
        ax[1].axvspan(r["train_idx"][0], r["train_idx"][-1], color="0.9", label="train")
        ax[1].set_xlabel("time step"); ax[1].set_ylabel("output"); ax[1].legend(loc="upper right", fontsize=8)
        fig.tight_layout(); fig.savefig(a.plot, dpi=150)
        print("wrote", a.plot)


if __name__ == "__main__":
    main()

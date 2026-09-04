#!/usr/bin/env python3
"""Plot a field sweep produced by mumaxplus/run_field_sweep.py (sweep.npz).

    python scripts/plot_sweep.py runs/colab/hyst2_sweep.npz --out hyst2.png

Top panel: per-region static magnetisation projected on the field axis
(hysteresis of every layer-island).  Bottom panel (if the sweep contains
spectra): FMR power vs field and frequency, the simulated analogue of the
experimental colour maps in Dion et al. Fig. 2.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sweep")
    ap.add_argument("--out", default=None)
    ap.add_argument("--f-max", type=float, default=15e9)
    a = ap.parse_args(argv)
    z = np.load(a.sweep)
    B, m = z["fields"], z["m_static"]
    ang = float(z["angle_deg"]) if "angle_deg" in z else 1.0
    d = np.array([np.cos(np.radians(ang)), np.sin(np.radians(ang)), 0.0])
    proj = m @ d
    regions = z["region_ids"] if "region_ids" in z else np.arange(1, proj.shape[1] + 1)
    has_fmr = "power" in z and z["power"].size > 0

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2 if has_fmr else 1, 1, figsize=(8, 8 if has_fmr else 4.5), sharex=True, squeeze=False)
    ax = axes[0, 0]
    for j, r in enumerate(regions):
        isl, layer = (int(r) - 1) // 2, (int(r) - 1) % 2
        ax.plot(B * 1e3, proj[:, j], lw=1.2, ls="-" if layer == 0 else "--",
                label=f"island {isl} {'bottom' if layer == 0 else 'top'}")
    ax.set_ylabel("m · field axis")
    ax.legend(fontsize=7, ncol=2)
    ax.set_title(f"quasi-static sweep, field at {ang:g}° from x")
    if has_fmr:
        f, P = z["freqs"], z["power"]                      # (nB, nf, nreg)
        S = P.sum(axis=2)
        S = np.log10(1 + S / S.max() * 1e4)
        ax2 = axes[1, 0]
        sel = f <= a.f_max
        ax2.pcolormesh(B * 1e3, f[sel] / 1e9, S[:, sel].T, cmap="inferno", shading="auto")
        ax2.set_ylabel("f (GHz)")
    axes[-1, 0].set_xlabel("μ0H (mT)")
    fig.tight_layout()
    out = a.out or str(Path(a.sweep).with_suffix(".png"))
    fig.savefig(out, dpi=140)
    print("wrote", out)


if __name__ == "__main__":
    main()

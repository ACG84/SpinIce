#!/usr/bin/env python3
"""Extract per-layer switching fields from a sweep.npz and suggest protocol settings.

    python scripts/switching_fields.py runs/colab/hyst3_sweep.npz [--layer top]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sweep", nargs="+")
    ap.add_argument("--jump", type=float, default=0.8, help="min |Δ(m·d)| counted as a switching event")
    a = ap.parse_args(argv)
    events = {"top": [], "bottom": []}
    for path in a.sweep:
        z = np.load(path)
        B, m = z["fields"], z["m_static"]
        ang = float(z["angle_deg"]) if "angle_deg" in z else 1.0
        d = np.array([np.cos(np.radians(ang)), np.sin(np.radians(ang)), 0.0])
        proj = m @ d
        regions = z["region_ids"] if "region_ids" in z else np.arange(1, proj.shape[1] + 1)
        print(f"== {path}")
        for j, r in enumerate(regions):
            isl, layer = (int(r) - 1) // 2, "bottom" if (int(r) - 1) % 2 == 0 else "top"
            dm = np.diff(proj[:, j])
            for i in np.where(np.abs(dm) > a.jump)[0]:
                print(f"  island {isl} {layer:6s}: {B[i + 1] * 1e3:6.1f} mT  (Δm = {dm[i]:+.2f})")
                events[layer].append(B[i + 1])
    for layer in ("top", "bottom"):
        if events[layer]:
            e = np.sort(np.array(events[layer])) * 1e3
            print(f"{layer:6s} layers: {len(e)} events, {e.min():.1f} .. {e.max():.1f} mT, "
                  f"median {np.median(e):.1f} mT, values {np.round(e, 1).tolist()}")
    top = np.sort(np.array(events["top"])) * 1e3
    if len(top) >= 2:
        lo, hi = top.min(), top.max()
        print("\nsuggested leak protocol:")
        print(f"  --b-min {lo - 2:.0f}e-3 --b-max {hi + 2:.0f}e-3 --leak-field {lo + 0.3 * (hi - lo):.0f}e-3 "
              f"--coarse-below {lo - 6:.0f}e-3")


if __name__ == "__main__":
    main()

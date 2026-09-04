#!/usr/bin/env python3
"""Energy-landscape proxies from state catalogues (no GPU).

For each catalogue.json: energies relative to the ground state, net moment
along the island axis, the "reordering field" B_r = dE / (dM) at which the
Zeeman energy would make a state degenerate with the ground state, and
spread / clustering metrics of those quantities.  These are the quantities
to minimise (spread) or maximise (number of states inside a field window)
in an inverse-design loop.

    python scripts/catalogue_analysis.py runs/colab/cat_single_10/catalogue.json runs/colab/cat_single_8/catalogue.json
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asvi_rc import ASVIParams   # noqa: E402


def layer_moment(p: ASVIParams, layer: int) -> float:
    """Saturation moment (A m^2) of one layer of a stadium island."""
    area = (p.length - p.width) * p.width + math.pi * (p.width / 2) ** 2 if p.rounded_ends else p.length * p.width
    t = p.t_bottom if layer == 0 else p.t_top
    return p.msat * area * t


def analyse(path, window_mT=None):
    res = json.loads(Path(path).read_text())
    p = ASVIParams.from_dict(res["params"])
    rows = res["rows"]
    # distinct relaxed states: take the lowest-energy representative of each label tuple
    best = {}
    for r in rows:
        k = tuple(r["final"])
        if k not in best or r["energy_J"] < best[k]["energy_J"]:
            best[k] = r
    states = sorted(best.values(), key=lambda r: r["energy_J"])
    e0 = states[0]["energy_J"]
    n_layers = len(states[0]["final"])
    moments = [layer_moment(p, j % 2) for j in range(n_layers)]      # regions alternate bottom/top
    M = np.array([sum(m_ax * mom for m_ax, mom in zip(r["m_axis"], moments)) for r in states])  # A m^2 along axis
    dE = np.array([r["energy_J"] - e0 for r in states])
    # degenerate ground states (e.g. +/- and -/+): refer each state to the ground state it
    # separates from most in moment, i.e. the transition a field along the axis enables first
    gs = np.flatnonzero(dE < 1e-3 * max(dE.max(), 1e-30))
    dM = np.max(np.abs(M[:, None] - M[gs][None, :]), axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        B_r = np.where(dM > 1e-3 * max(moments), dE / dM, np.nan)      # tesla: dE = dM * B
    B_r[gs] = 0.0
    print(f"== {path}  (cell {p.cell_xy * 1e9:g} nm, {len(states)} distinct states, {sum(r['stable'] for r in rows)}/{len(rows)} seeds stable)")
    print("   dE (aJ)   M/Ms_tot   B_reorder (mT)   state")
    Mtot = sum(moments)
    for r, e, m, b in zip(states, dE, M, B_r):
        print(f"   {e * 1e18:7.2f}   {m / Mtot:+6.2f}   {'   n/a  ' if np.isnan(b) else f'{b * 1e3:8.1f}'}       {'/'.join(r['final'])}")
    ex = dE[1:] * 1e18
    levels = np.unique(np.round(dE * 1e18, 2))
    gaps = np.diff(levels)
    br = B_r[~np.isnan(B_r)] * 1e3
    out = {
        "n_states": len(states),
        "dE_max_aJ": float(dE.max() * 1e18),
        "dE_std_aJ": float(np.std(ex)) if len(ex) else 0.0,
        "n_levels": int(len(levels)),
        "median_gap_aJ": float(np.median(gaps)) if len(gaps) else 0.0,
        "B_reorder_median_mT": float(np.median(br)) if len(br) else float("nan"),
        "B_reorder_spread_mT": float(np.std(br)) if len(br) else float("nan"),
    }
    if window_mT:
        lo, hi = window_mT
        out["states_in_window"] = int(np.sum((br >= lo) & (br <= hi)))
    print("   metrics:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items()})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("catalogues", nargs="+")
    ap.add_argument("--window", type=float, nargs=2, default=None, help="field window (mT) to count reorderable states in")
    a = ap.parse_args(argv)
    for c in a.catalogues:
        analyse(c, a.window)


if __name__ == "__main__":
    main()

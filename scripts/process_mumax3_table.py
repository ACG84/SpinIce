#!/usr/bin/env python3
"""Convert a mumax3 table.txt written by a generated script into spectra.npz.

    python scripts/process_mumax3_table.py <table.txt> <meta.json> [--out spectra.npz]

For reservoir runs the dataset.npz next to the meta file is merged in so
that train_readout.py has inputs and targets.  For sweep runs the output
also contains the static per-region magnetisation at every field.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asvi_rc.spectra import read_mumax3_table, spectra_from_mumax3_table, save_spectra   # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("table")
    ap.add_argument("meta")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    meta = json.loads(Path(a.meta).read_text())
    regions = meta["region_ids"]
    freqs, power, steps, b_loop = spectra_from_mumax3_table(a.table, regions, meta["fmr_dt"], meta["fmr_nt"])
    tab = read_mumax3_table(a.table)
    # static state at t=0 of every step (first row of the step)
    m_static = np.stack([np.stack([[tab[f"m.region{r}{c}"][tab["step"] == s][0] for c in "xyz"]
                                   for r in regions]) for s in steps])
    out = Path(a.out) if a.out else Path(a.meta).with_name("spectra.npz")
    ds = Path(a.meta).with_name("dataset.npz")
    u = y = None
    if ds.exists():
        with np.load(ds) as z:
            u, y = z["u"], z["y"]
        if len(u) != len(steps):
            print(f"warning: dataset has {len(u)} steps, table has {len(steps)} (partial run?)")
            u, y = u[:len(steps)], y[:len(steps)]
    save_spectra(out, freqs, power, b_loop, u=u, y=y, steps=steps, m_static=m_static,
                 region_ids=np.array(regions))
    print(f"wrote {out}: {len(steps)} steps, {len(freqs)} frequencies, {len(regions)} regions")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Collect the per-run JSONs of the lattice-automaton / flatspin scans (runs/<family>_<config>/...) into one
JSON per family under docs/data/lattice_scans/ (config name -> rows)."""
import json, sys
from pathlib import Path

FAMILIES = {
    "lat2c": "2-layer 45 deg, corrected thresholds (axial rule)", "lat3c": "3-layer 45 deg (axial rule)",
    "lat2d": "2-layer symmetry-breaking angles 15/25/35", "lat3d": "3-layer symmetry-breaking angles",
    "lat2e": "2-layer partial-reset leaks straddling the top curve", "lat3e": "3-layer low-field windows",
    "lat2f": "2-layer deterministic switching / small noise", "lat2g": "2-layer coupling multiplier (axial rule)",
    "lat2h": "2-layer near-zero leaks (axial rule)", "lat2s": "2-layer SW local field at the centre",
    "lat2e2": "2-layer SW local field at the island ends", "lat2sub": "2-layer sub-threshold leaks (sw_ends)",
    "lat2gsw": "2-layer generalised astroid (stadium fit), sw_ends", "lat2strong": "2-layer strong coupling, flatspin-scaled windows",
    "fs_sw1": "flatspin 8x8 with the ideal SW astroid", "fs_def": "flatspin 8x8 default astroid, retuned windows",
    "fsp": "flatspin pinwheel", "fs8_rw": "flatspin rewrite-fraction scan",
}
out = Path("docs/data/lattice_scans"); out.mkdir(parents=True, exist_ok=True)
for fam, desc in FAMILIES.items():
    rows = {}
    for d in sorted(Path("runs").glob(fam + "_*")):
        for f in d.glob("*.json"):
            try:
                rows[d.name[len(fam) + 1:]] = json.loads(f.read_text())
            except Exception as e:
                print("skip", f, e, file=sys.stderr)
    if rows:
        (out / f"{fam}.json").write_text(json.dumps({"description": desc, "runs": rows}, indent=1))
        print(f"{fam:10s} {len(rows):4d} configs -> {out / (fam + '.json')}")

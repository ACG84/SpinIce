#!/usr/bin/env python
"""FORCE learning (online RLS readout with output feedback into the field amplitude) on the
ASVI automaton, from a GPU transition table or from an energy-landscape catalogue.

    python scripts/force_rc.py --catalogue docs/data/np_single_10/catalogue.json --angle 1 \\
        --protocol 25.2e-3 29.9e-3 33.4e-3 1.2e-3 -26.8e-3 --tasks recall:1 recall:2 narma2 sine \\
        --gains 0 0.3 1 --out runs/force_single
    python scripts/force_rc.py --table docs/data/trans_cell_10_45deg/transitions.json --out runs/force_table45
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asvi_rc.force import TableReservoir, SoftReservoir, closed_loop, make_target   # noqa: E402
from scripts.protocol_design import landscape                                       # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap._negative_number_matcher = re.compile(r"^-\d+$|^-\d*\.\d+$|^-\d+(\.\d*)?[eE][-+]?\d+$")  # accept -26.8e-3
    ap.add_argument("--table", type=str, default=None, help="transitions.json (deterministic automaton)")
    ap.add_argument("--catalogue", type=str, default=None, help="catalogue.json (soft automaton, sampled)")
    ap.add_argument("--angle", type=float, default=1.0)
    ap.add_argument("--barrier", type=float, default=70e-18)
    ap.add_argument("--width", type=float, default=2e-3)
    ap.add_argument("--protocol", nargs=5, type=float, default=(25.2e-3, 29.9e-3, 33.4e-3, 1.2e-3, -26.8e-3),
                    metavar=("B_MIN", "B_MAX", "LEAK", "JITTER", "BIAS"))
    ap.add_argument("--tasks", nargs="+", default=["recall:1", "recall:2", "recall:3", "narma2", "sine"])
    ap.add_argument("--gains", nargs="+", type=float, default=[0.0, 0.3, 1.0])
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    proto = dict(zip(("b_min", "b_max", "leak", "jitter", "bias"), a.protocol))
    if a.table:
        tab = json.loads(Path(a.table).read_text())
        make_res = lambda seed: TableReservoir(tab)
        print(f"table automaton: {len(tab['states'])} states, amplitudes {tab['amplitudes_T']}")
    else:
        states, E, M = landscape(a.catalogue, a.angle)
        make_res = lambda seed: SoftReservoir(E, M, states, a.barrier, a.width, seed=seed)
        print(f"soft automaton: {len(states)} states, drive {a.angle:g} deg")
    print("protocol (mT):", {k: round(v * 1e3, 1) for k, v in proto.items()})
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = []
    print(f"{'task':10s} {'gain':>5s} {'train NRMSE':>12s} {'test NRMSE':>11s} {'test R2':>8s} {'visited':>8s}")
    for task in a.tasks:
        name, _, k = task.partition(":")
        k = int(k) if k else 1
        for g in a.gains:
            res_seeds = []
            for seed in range(a.seeds):
                rng = np.random.default_rng(100 + seed)
                u = np.zeros(a.n) if name == "sine" else rng.random(a.n)
                y = make_target(name, u, k)
                r = closed_loop(make_res(seed), u, y, proto, g, int(a.train_frac * a.n), seed=seed)
                res_seeds.append(r)
            row = {"task": task, "gain": g,
                   "train_nrmse": float(np.mean([r["train_nrmse"] for r in res_seeds])),
                   "test_nrmse": float(np.mean([r["test_nrmse"] for r in res_seeds])),
                   "test_r2": float(np.mean([r["test_r2"] for r in res_seeds])),
                   "n_visited": float(np.mean([r["n_visited"] for r in res_seeds]))}
            rows.append(row)
            print(f"{task:10s} {g:5.2f} {row['train_nrmse']:12.3f} {row['test_nrmse']:11.3f} {row['test_r2']:8.3f} "
                  f"{row['n_visited']:8.1f}", flush=True)
    (out / "force.json").write_text(json.dumps({"protocol": proto, "rows": rows}, indent=1))
    print("wrote", out / "force.json")


if __name__ == "__main__":
    main()

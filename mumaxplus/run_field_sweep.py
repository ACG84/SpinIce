#!/usr/bin/env python3
"""FMR-vs-field map and hysteresis loop with mumax+ (GPU required).

Reproduces the kind of data in Fig. 2 of Dion et al. 2024: starting from
negative saturation the in-plane field is swept in steps; at every field the
system is minimised, the static per-region magnetisation is recorded (the
hysteresis loop of every layer-island) and a broadband FMR spectrum is
computed.  Use the resulting switching fields to choose the reservoir input
range (ProtocolParams.b_min / b_max).

Example
-------
    python mumaxplus/run_field_sweep.py --b-start -30e-3 --b-stop 60e-3 --b-step 1e-3 \
        --angle 1 --out runs/sweep
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asvi_rc.geometry import describe                              # noqa: E402
from asvi_rc.mumaxplus_driver import ASVISimulation                # noqa: E402
from run_reservoir import add_common_args, params_from_args        # noqa: E402


def _allow_negative_sci(parser: argparse.ArgumentParser):
    """Let argparse accept values like -30e-3 (default matcher only handles -30 / -0.03)."""
    import re
    parser._negative_number_matcher = re.compile(r"^-(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$")
    for act in parser._actions:
        if isinstance(act, argparse._SubParsersAction):
            for sp in act.choices.values():
                _allow_negative_sci(sp)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--b-start", type=float, default=-30e-3)
    ap.add_argument("--b-stop", type=float, default=60e-3)
    ap.add_argument("--b-step", type=float, default=1e-3)
    ap.add_argument("--angle", type=float, default=1.0, help="field angle from +x (deg)")
    ap.add_argument("--presaturate", type=float, default=-0.2)
    ap.add_argument("--no-fmr", action="store_true", help="hysteresis loop only")
    ap.add_argument("--out", type=str, default="runs/sweep")
    add_common_args(ap)
    _allow_negative_sci(ap)
    a = ap.parse_args(argv)

    p = params_from_args(a)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sim = ASVISimulation(p)
    print(describe(p, sim.islands))
    p.save(out / "params.json")

    th = math.radians(a.angle)
    d = (math.cos(th), math.sin(th))
    # presaturate < 0 means "saturate along -d"
    sim.saturate(direction=d if a.presaturate < 0 else (-d[0], -d[1]), amplitude=abs(a.presaturate))
    fields = np.arange(a.b_start, a.b_stop + 0.5 * a.b_step, a.b_step)
    m_static, power, freqs = [], [], None
    for k, b in enumerate(fields):
        bvec = (b * d[0], b * d[1], 0.0)
        sim.set_field(bvec)
        sim.minimize()
        m_static.append(sim.region_averages())
        if not a.no_fmr:
            freqs, P = sim.fmr(bvec)
            power.append(P)
            pk = freqs[np.argmax(P.sum(axis=1)[1:]) + 1] / 1e9
        else:
            pk = float("nan")
        mx = m_static[-1] @ np.array([*d, 0.0])
        print(f"B {b*1e3:7.2f} mT  <m.d> per region {np.round(mx, 2).tolist()}  peak {pk:5.2f} GHz", flush=True)
        np.savez_compressed(out / "sweep.npz", fields=fields[:k + 1], m_static=np.stack(m_static),
                            freqs=freqs if freqs is not None else np.array([]),
                            power=np.stack(power) if power else np.array([]),
                            region_ids=np.array(sim.region_ids), angle_deg=a.angle)
    print("wrote", out / "sweep.npz")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Evaluate reservoir proxies from a state-transition table (no GPU).

Loads transitions.json (from state_catalogue.py --transitions) and, optionally,
state_spectra.npz, then runs the field protocol as a finite automaton on any
input series: s <- T[s][-L_k] (leak) ; s <- T[s][+B_k] (write).  Reports the
number of states visited, their occupancy, stationarity, the recall of past
inputs and Mackey-Glass / NARMA scores with a linear readout on the per-state
spectra (or a one-hot of the state if no spectra are given).

    python scripts/automaton_rc.py runs/colab/cat_cell_10/transitions.json \
        --b-min 28e-3 --b-max 38e-3 --leak 33e-3 --leak-jitter 5e-3 --n 2000
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asvi_rc import tasks, readout                     # noqa: E402
from asvi_rc.spectra import features                   # noqa: E402


def nearest_key(table_row, value_mT):
    keys = np.array([float(k) for k in table_row])
    return list(table_row)[int(np.argmin(np.abs(keys - value_mT)))]


def run_automaton(T, u, b_min, b_max, leak, jitter, seed=0, s0=0, shape="leak"):
    rng = np.random.default_rng(seed)
    B = b_min + (b_max - b_min) * u
    s = s0
    states, leaks = [], []
    for k, b in enumerate(B):
        if shape == "leak":
            L = leak + jitter * rng.uniform(-1, 1)
            s = T[str(s)][nearest_key(T[str(s)], -L * 1e3)]
            leaks.append(L)
        elif shape == "alternating":
            b = b * (-1) ** k
        s = T[str(s)][nearest_key(T[str(s)], b * 1e3)]
        states.append(s)
    return np.array(states), np.array(leaks)


def accessibility(T, S, lo_mT, hi_mT, start=None):
    """Reachability structure of the transition graph restricted to |B| in [lo, hi] mT.

    Returns (reachable-from-start set, largest strongly connected component,
    edges used).  Both signs of the excursion are allowed inside the window.
    """
    n = len(S)
    adj = {i: set() for i in range(n)}
    for i in range(n):
        for key, j in T[str(i)].items():
            if lo_mT - 1e-9 <= abs(float(key)) <= hi_mT + 1e-9 and j != i:
                adj[i].add(j)
    start = 0 if start is None else start
    seen, stack = {start}, [start]
    while stack:
        for j in adj[stack.pop()]:
            if j not in seen:
                seen.add(j); stack.append(j)
    # strongly connected components (Kosaraju)
    order, visited = [], set()
    def dfs(u):
        visited.add(u)
        for v in adj[u]:
            if v not in visited: dfs(v)
        order.append(u)
    for u in range(n):
        if u not in visited: dfs(u)
    radj = {i: set() for i in range(n)}
    for u in range(n):
        for v in adj[u]: radj[v].add(u)
    comp, assigned = [], set()
    for u in reversed(order):
        if u in assigned: continue
        c, st = set(), [u]
        while st:
            x = st.pop()
            if x in assigned: continue
            assigned.add(x); c.add(x); st.extend(radj[x] - assigned)
        comp.append(c)
    scc = max(comp, key=len)
    return seen, scc, sum(len(v) for v in adj.values())


def report_accessibility(T, S, energies=None):
    amps = sorted({abs(float(k)) for row in T.values() for k in row})
    print("accessibility vs amplitude window (states reachable from ground / largest mutually-reachable set / edges):")
    for lo, hi in ((amps[0], amps[-1]), (24, 40), (28, 38), (31, 47), (36, 60)):
        reach, scc, ne = accessibility(T, S, lo, hi)
        print(f"  |B| in [{lo:g}, {hi:g}] mT: {len(reach):3d} / {len(scc):3d} / {ne:4d}   of {len(S)} states")
    # minimal window width (centred scan) to reach >= k states from the ground state
    print("minimal window [lo, hi] reaching >= k states:")
    for k in (4, 8, 16, 32):
        best = None
        for lo in amps:
            for hi in amps:
                if hi < lo: continue
                reach, scc, _ = accessibility(T, S, lo, hi)
                if len(reach) >= k and (best is None or hi - lo < best[1] - best[0]):
                    best = (lo, hi, len(reach), len(scc))
        print(f"  k={k:2d}: " + (f"[{best[0]:g}, {best[1]:g}] mT (reaches {best[2]}, mutually reachable {best[3]})" if best else "not reachable"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("transitions")
    ap.add_argument("--spectra", default=None, help="state_spectra.npz (default: alongside transitions.json)")
    ap.add_argument("--task", default="mackey_glass_10", choices=list(tasks.TASKS))
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--b-min", type=float, default=28e-3)
    ap.add_argument("--b-max", type=float, default=38e-3)
    ap.add_argument("--leak", type=float, default=33e-3)
    ap.add_argument("--leak-jitter", type=float, default=5e-3)
    ap.add_argument("--shape", default="leak", choices=["leak", "unipolar", "alternating"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    tr = json.loads(Path(a.transitions).read_text())
    T, S = tr["table"], tr["states"]
    report_accessibility(T, S)
    spec_path = Path(a.spectra) if a.spectra else Path(a.transitions).with_name("state_spectra.npz")
    if spec_path.exists():
        z = np.load(spec_path)
        F = np.stack([features(z["freqs"], z["power"][i][:, None], 2e9, 14e9, 100e6)[1] for i in range(len(S))])
        feat_kind = f"spectra ({F.shape[1]} bins)"
    else:
        F = np.eye(len(S)); feat_kind = "one-hot state"
    u, y = tasks.make_task(a.task, a.n, seed=a.seed)
    st, _ = run_automaton(T, u, a.b_min, a.b_max, a.leak, a.leak_jitter, seed=a.seed, shape=a.shape)
    X = F[st]
    occ = np.bincount(st, minlength=len(S)) / len(st)
    visited = np.flatnonzero(occ)
    ent = -np.sum(occ[visited] * np.log2(occ[visited]))
    first_seen = [int(np.argmax(st == v)) for v in visited]
    print(f"{len(S)} states in table ({tr['axis_deg']:g} deg axis, bias {tr['bias_T']*1e3:g} mT); features: {feat_kind}")
    print(f"{a.task}, n={a.n}, window {a.b_min*1e3:g}-{a.b_max*1e3:g} mT, {a.shape} protocol, leak {a.leak*1e3:g}+-{a.leak_jitter*1e3:g} mT")
    print(f"visited states: {len(visited)} of {len(S)}; occupancy entropy {ent:.2f} bits; "
          f"last new state first seen at step {max(first_seen)}")
    print("occupancy:", {int(v): round(float(occ[v]), 3) for v in visited})
    n_train = a.n // 2
    r = readout.evaluate(X, y, n_train, alpha=1e-2, washout=20, u=u)
    print(f"target {a.task}: reservoir NRMSE {r['nrmse_test']:.3f}   raw-input NRMSE {r['nrmse_test_baseline']:.3f}")
    r2 = lambda t, p: 1 - np.mean((t - p) ** 2) / np.var(t)
    print("recall of u(t-k), test R2 (reservoir / raw input):")
    U = np.stack([u, u**2, u**3], 1)
    for k in (0, 1, 2, 3, 5, 8):
        yk = np.roll(u, k); tri, tei = readout.split(a.n, n_train, 20 + k)
        m = readout.RidgeReadout(alpha=1e-2).fit(X[tri], yk[tri]); mb = readout.RidgeReadout(alpha=1e-2).fit(U[tri], yk[tri])
        print(f"  k={k}: {r2(yk[tei], m.predict(X[tei])):6.2f} / {r2(yk[tei], mb.predict(U[tei])):6.2f}")
    caps, mc = readout.memory_capacity(X, np.random.default_rng(1).uniform(0, 1, a.n) if False else u, n_train, alpha=1e-2, washout=20)
    print(f"linear memory capacity (input-driven, periodicity-contaminated for MG): {mc:.2f}")


if __name__ == "__main__":
    main()

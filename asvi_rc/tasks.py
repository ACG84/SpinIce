"""Input signals, benchmark targets and the input -> field-loop encoding."""
from __future__ import annotations

import math

import numpy as np

from .params import ProtocolParams


# ---------------------------------------------------------------------------
# input waveforms (Gartside et al. use ~30 points per period)
# ---------------------------------------------------------------------------
def sine(n: int, points_per_period: int = 30, phase: float = 0.0) -> np.ndarray:
    t = np.arange(n)
    return np.sin(2 * np.pi * t / points_per_period + phase)


def square(n: int, points_per_period: int = 30) -> np.ndarray:
    return np.sign(sine(n, points_per_period) + 1e-12)


def saw(n: int, points_per_period: int = 30) -> np.ndarray:
    t = np.arange(n) % points_per_period
    return 2 * t / points_per_period - 1


def inverse_saw(n: int, points_per_period: int = 30) -> np.ndarray:
    return -saw(n, points_per_period)


def triangle(n: int, points_per_period: int = 30) -> np.ndarray:
    t = (np.arange(n) % points_per_period) / points_per_period
    return 4 * np.abs(t - 0.5) - 1


def mackey_glass(n: int, tau: int = 17, beta: float = 0.2, gamma: float = 0.1,
                 n_exp: int = 10, dt: float = 1.0, x0: float = 1.2,
                 discard: int = 500, subsample: int = 1, seed: int | None = None) -> np.ndarray:
    """Mackey-Glass series dx/dt = beta x(t-tau)/(1+x(t-tau)^n) - gamma x.

    Integrated with RK4 on a fixed grid of step ``dt`` (tau/dt must be an
    integer).  ``discard`` transient steps are dropped; the result is
    optionally sub-sampled.
    """
    if seed is not None:
        x0 = x0 + 0.1 * np.random.default_rng(seed).standard_normal()
    d = int(round(tau / dt))
    total = discard + n * subsample
    x = np.empty(total + d + 1)
    x[: d + 1] = x0

    def f(xt, xd):
        return beta * xd / (1 + xd ** n_exp) - gamma * xt

    for i in range(d, total + d):
        xd = x[i - d]
        k1 = f(x[i], xd)
        k2 = f(x[i] + 0.5 * dt * k1, xd)
        k3 = f(x[i] + 0.5 * dt * k2, xd)
        k4 = f(x[i] + dt * k3, xd)
        x[i + 1] = x[i] + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    out = x[d + 1 + discard:]
    return out[::subsample][:n]


def narma(u: np.ndarray, order: int = 10, a=0.3, b=0.05, c=1.5, d=0.1) -> np.ndarray:
    """NARMA-``order`` target driven by input ``u`` (u should lie in [0, 0.5])."""
    n = len(u)
    y = np.zeros(n)
    for t in range(order, n):
        y[t] = (a * y[t - 1] + b * y[t - 1] * np.sum(y[t - order:t])
                + c * u[t - order] * u[t - 1] + d)
    return y


def second_order_hysteretic(I: np.ndarray) -> np.ndarray:
    """y(t) = 0.4 y(t-1) + 0.4 y(t-1) y(t-2) + 0.6 I^3(t) + 0.1  (Gartside et al.)."""
    y = np.zeros(len(I))
    for t in range(len(I)):
        y1 = y[t - 1] if t >= 1 else 0.0
        y2 = y[t - 2] if t >= 2 else 0.0
        y[t] = 0.4 * y1 + 0.4 * y1 * y2 + 0.6 * I[t] ** 3 + 0.1
    return y


def normalise(x: np.ndarray, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    """Affine map of x onto [lo, hi]."""
    xmin, xmax = np.min(x), np.max(x)
    if xmax - xmin < 1e-15:
        return np.full_like(x, 0.5 * (lo + hi), dtype=float)
    return lo + (hi - lo) * (x - xmin) / (xmax - xmin)


# ---------------------------------------------------------------------------
# canned datasets
# ---------------------------------------------------------------------------
TASKS = {
    "sine_to_square":     "sine input, square-wave target",
    "sine_to_saw":        "sine input, saw-wave target",
    "sine_to_nonlinear":  "sine input, 2nd-order hysteretic target",
    "isaw_to_sine":       "inverse-saw input, sine target",
    "isaw_to_square":     "inverse-saw input, square target",
    "isaw_to_nonlinear":  "inverse-saw input, 2nd-order hysteretic target",
    "mackey_glass_1":     "Mackey-Glass, predict t+1",
    "mackey_glass_10":    "Mackey-Glass, predict t+10",
    "narma10":            "NARMA-10 driven by uniform random input",
}


def make_task(name: str, n: int = 600, points_per_period: int = 30, seed: int = 0):
    """Return (u, y) input and target series of length n, u normalised to [0, 1]."""
    rng = np.random.default_rng(seed)
    if name.startswith("sine_to_") or name.startswith("isaw_to_"):
        src, dst = name.split("_to_")
        u = sine(n, points_per_period) if src == "sine" else inverse_saw(n, points_per_period)
        if dst == "square":
            y = square(n, points_per_period)
        elif dst == "saw":
            y = saw(n, points_per_period)
        elif dst == "sine":
            y = sine(n, points_per_period)
        elif dst == "nonlinear":
            # the recursion is only stable for small inputs -> use I in [0, 0.5]
            y = second_order_hysteretic(normalise(u, 0.0, 0.5))
        else:
            raise ValueError(name)
    elif name.startswith("mackey_glass_"):
        h = int(name.rsplit("_", 1)[1])
        # ~30 samples per quasi-period of MG(tau=17) -> subsample by 2 of dt=1 integration
        x = mackey_glass(n + h, subsample=2, seed=seed)
        u, y = x[:n], x[h:h + n]
    elif name == "narma10":
        u = rng.uniform(0, 0.5, n)
        y = narma(u, 10)
    else:
        raise ValueError(f"unknown task {name!r}; choose from {list(TASKS)}")
    return normalise(u), np.asarray(y, dtype=float)


# ---------------------------------------------------------------------------
# input encoding: scalar -> field loop
# ---------------------------------------------------------------------------
def input_to_field(u: np.ndarray, proto: ProtocolParams) -> np.ndarray:
    """Linear map of normalised inputs (0..1) to loop amplitude (T)."""
    u = np.clip(np.asarray(u, dtype=float), 0, 1)
    return proto.b_min + (proto.b_max - proto.b_min) * u


def _ramp(b_from: float, b_to: float, step: float) -> np.ndarray:
    """Field magnitudes from b_from to b_to (inclusive of b_to) in |step| increments."""
    if abs(b_to - b_from) < 1e-15:
        return np.array([b_to])
    n = max(1, int(math.ceil(abs(b_to - b_from) / step - 1e-9)))
    return b_from + (b_to - b_from) * np.arange(1, n + 1) / n


def _adaptive_ramp(b_from: float, b_to: float, proto: ProtocolParams) -> np.ndarray:
    """Ramp with `loop_step` where |B| >= coarse_below and `coarse_step` below it.

    Far from the switching window nothing happens, so larger quasi-static
    increments there save energy minimisations without changing the
    trajectory near the switching fields.
    """
    if not proto.coarse_step or proto.coarse_below <= 0:
        return _ramp(b_from, b_to, proto.loop_step)
    thr = proto.coarse_below
    # split the interval at +-thr into fine / coarse pieces
    pts = sorted({b_from, b_to, *[x for x in (-thr, thr) if min(b_from, b_to) < x < max(b_from, b_to)]},
                 reverse=bool(b_to < b_from))
    out = []
    for a, b in zip(pts[:-1], pts[1:]):
        step = proto.coarse_step if max(abs(a), abs(b)) <= thr + 1e-12 else proto.loop_step
        out.append(_ramp(a, b, step))
    return np.concatenate(out) if out else np.array([b_to])


def measurement_field(b_loop: float, proto: ProtocolParams, sign: float = 1.0) -> float:
    """Signed field amplitude (along the loop axis) at which FMR is measured."""
    if proto.loop_shape == "alternating" and proto.measure_at == "loop_max":
        return sign * b_loop
    if proto.loop_shape == "leak":
        return proto.bias_field          # always read out at the bias field
    return b_loop if proto.measure_at == "loop_max" else proto.bias_field


def minor_loop(b_loop: float, proto: ProtocolParams, b_start: float | None = None,
               sign: float = 1.0) -> np.ndarray:
    """Signed field magnitudes (along the loop axis) of one quasi-static minor loop.

    The sequence starts *after* ``b_start`` (default: the previous
    measurement field, assumed equal to measurement_field of the same loop)
    and ends at the measurement field of this loop.
    """
    if b_start is None:
        b_start = measurement_field(b_loop, proto, sign)
    r = lambda a, b: _adaptive_ramp(a, b, proto)  # noqa: E731
    if proto.loop_shape == "alternating":
        seq = [r(b_start, sign * b_loop)]
    elif proto.loop_shape == "leak":
        seq = [r(b_start, +b_loop), r(+b_loop, -proto.leak_field)]
    elif proto.loop_shape == "bipolar":
        seq = [r(b_start, -b_loop), r(-b_loop, +b_loop)]
    elif proto.loop_shape == "unipolar":
        seq = [r(b_start, +b_loop)]
    elif proto.loop_shape == "return":
        seq = [r(b_start, -b_loop), r(-b_loop, +b_loop), r(+b_loop, 0.0)]
    else:
        raise ValueError(f"unknown loop_shape {proto.loop_shape!r}")
    seq.append(r(seq[-1][-1], measurement_field(b_loop, proto, sign)))
    out = np.concatenate(seq)
    # drop consecutive duplicates
    keep = np.ones(len(out), dtype=bool)
    keep[1:] = np.abs(np.diff(out)) > 1e-15
    return out[keep]


def field_vector(b: float, proto: ProtocolParams) -> tuple[float, float, float]:
    """Signed amplitude along the loop axis -> (Bx, By, Bz) in tesla."""
    cx, cy = proto.loop_direction
    return (b * cx, b * cy, 0.0)


def field_schedule(u: np.ndarray, proto: ProtocolParams) -> list[dict]:
    """Full quasi-static schedule for an input series.

    Returns one dict per input step with keys ``b_loop`` (T), ``ramp``
    (signed amplitudes to visit, ending at the measurement field) and
    ``b_meas``.
    """
    b_loops = input_to_field(u, proto)
    sched = []
    prev = None  # we start from negative saturation
    for k, b in enumerate(b_loops):
        sign = (-1.0) ** k if proto.loop_shape == "alternating" else 1.0
        if prev is None:
            # coarse descent from -B_sat towards the first loop, then the loop proper
            first = -float(b) if proto.loop_shape != "alternating" else -float(b)
            approach = _ramp(-proto.saturation_field, first, proto.approach_step)[:-1]
            ramp = np.concatenate([approach, minor_loop(float(b), proto, b_start=first, sign=sign)])
        else:
            ramp = minor_loop(float(b), proto, b_start=prev, sign=sign)
        b_meas = measurement_field(float(b), proto, sign)
        sched.append({"b_loop": float(b), "ramp": ramp, "b_meas": b_meas, "sign": sign})
        prev = b_meas
    return sched

"""Drive the multilayered ASVI with the mumax+ Python API.

Everything mumax+-specific is confined to this module; the geometry, field
schedule and spectral analysis are the shared numpy code.  ``mumaxplus`` is
imported lazily so that the rest of the package (and the tests) work on a
machine without CUDA.

Typical use (see mumaxplus/run_reservoir.py)::

    sim = ASVISimulation(params)
    sim.saturate(proto)
    for k, step in enumerate(field_schedule(u, proto)):
        sim.apply_ramp(step["ramp"], proto)
        freqs, power = sim.fmr(field_vector(step["b_meas"], proto))
"""
from __future__ import annotations

import time

import numpy as np

from .geometry import build_geometry, build_islands, region_masks, uniform_magnetization
from .params import ASVIParams, ProtocolParams
from .spectra import power_spectrum
from .tasks import field_vector


def _import_mumaxplus():
    try:
        import mumaxplus  # noqa: F401
        from mumaxplus import Ferromagnet, Grid, World
    except ImportError as e:  # pragma: no cover
        raise ImportError("mumaxplus is not installed; see https://mumax.github.io/plus/") from e
    return World, Grid, Ferromagnet


class ASVISimulation:
    """One mumax+ world holding the periodic supercell as a single Ferromagnet.

    Both magnetic layers live in the same Ferromagnet with the Al spacer as
    empty (non-magnetic) cells, so the inter-layer dipolar coupling is the
    ordinary demagnetising field and no exchange crosses the spacer.
    """

    def __init__(self, p: ASVIParams, islands=None, verbose: bool = True):
        World, Grid, Ferromagnet = _import_mumaxplus()
        self.p = p
        self.mask, self.regions, self.islands = build_geometry(p, islands or build_islands(p))
        nx, ny, nz = p.grid
        # periodic (in-plane) master grid = the supercell; z is open
        self.world = World(cellsize=(p.cell_xy, p.cell_xy, p.cell_z),
                           pbc_repetitions=tuple(p.pbc_repetitions),
                           mastergrid=Grid((nx, ny, 0)))
        self.magnet = Ferromagnet(self.world, Grid((nx, ny, nz)), name="asvi",
                                  geometry=self.mask, regions=self.regions)
        self.magnet.msat = p.msat
        self.magnet.aex = p.aex
        self.magnet.alpha = p.alpha
        self.region_ids = [isl.region for isl in self.islands]
        self._rmask = region_masks(self.regions, self.islands)
        self.verbose = verbose
        self._b_static = (0.0, 0.0, 0.0)

    # ------------------------------------------------------------------ state
    def log(self, *a):
        if self.verbose:
            print(time.strftime("[%H:%M:%S]"), *a, flush=True)

    def set_magnetization(self, m: np.ndarray):
        self.magnet.magnetization = np.ascontiguousarray(m, dtype=float)

    def get_magnetization(self) -> np.ndarray:
        return np.asarray(self.magnet.magnetization.eval())

    def region_averages(self, m: np.ndarray | None = None) -> np.ndarray:
        """(nregions, 3) spatial average of m over every layer-island."""
        m = self.get_magnetization() if m is None else m
        flat = m.reshape(3, -1)
        return np.stack([flat[:, self._rmask[r]].mean(axis=1) for r in self.region_ids])

    def set_field(self, b):
        """Static in-plane field (T); also clears any time-dependent term."""
        self._b_static = tuple(float(x) for x in b)
        self.magnet.bias_magnetic_field = self._b_static

    def minimize(self, robust: bool = False):
        if robust:
            self.world.relax()
        else:
            self.world.minimize()

    def saturate(self, proto: ProtocolParams | None = None, direction=None, amplitude=None):
        """Saturate along -direction (default: the loop axis) with the given amplitude."""
        cx, cy = proto.loop_direction if direction is None else direction
        amp = proto.saturation_field if amplitude is None else amplitude
        self.set_magnetization(uniform_magnetization(self.regions, (-cx, -cy, 0.0)))
        self.set_field((-amp * cx, -amp * cy, 0.0))
        self.minimize()
        self.log("saturated at", self._b_static)

    def apply_ramp(self, ramp, proto: ProtocolParams, robust: bool = False):
        """Quasi-static field ramp: minimise after every increment."""
        for b in ramp:
            self.set_field(field_vector(float(b), proto))
            self.minimize(robust)

    # ------------------------------------------------------------------- FMR
    def fmr(self, b_static, return_timeseries: bool = False):
        """Broadband sinc excitation along z at fixed in-plane field.

        Returns (freqs, power (nf, nregions)) - and the raw region-averaged
        magnetisation (nt, nregions, 3) if requested.
        """
        p = self.p
        self.set_field(b_static)
        self.minimize()
        m0 = self.get_magnetization()
        amp, fc, t0 = p.fmr_amp, p.fmr_fcut, p.fmr_t0
        # numpy sinc is normalised: sinc(x) = sin(pi x)/(pi x)
        self.magnet.bias_magnetic_field.add_time_term(
            lambda t: (0.0, 0.0, amp * float(np.sinc(2 * fc * (t - t0)))))
        solver = self.world.timesolver
        solver.time = 0.0
        timepoints = np.arange(p.fmr_nt) * p.fmr_dt
        out = solver.solve(timepoints, {"m": lambda: self.region_averages()})
        m_t = np.asarray(out["m"])                      # (nt, nregions, 3)
        self.set_field(b_static)                        # removes the time term
        self.set_magnetization(m0)                      # discard ring-down, keep the static state
        freqs, power = power_spectrum(m_t, p.fmr_dt)
        return (freqs, power, m_t) if return_timeseries else (freqs, power)

    def save_state(self, path):
        np.save(path, self.get_magnetization())

    def load_state(self, path):
        self.set_magnetization(np.load(path))

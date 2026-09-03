"""Minimal stand-in for the mumax+ API used by asvi_rc.mumaxplus_driver.

Only used by the test-suite to exercise the driver's control flow on a
machine without CUDA.  Mirrors the signatures of World, Grid, Ferromagnet,
Parameter, Variable and TimeSolver that the driver touches.
"""
import numpy as np


class Grid:
    def __init__(self, size, origin=(0, 0, 0)):
        self.size = tuple(size)
        self.origin = tuple(origin)

    @property
    def shape(self):
        return tuple(reversed(self.size))


class _Param:
    def __init__(self, ncomp=3):
        self.value = (0.0, 0.0, 0.0) if ncomp == 3 else 0.0
        self.time_terms = []

    def set(self, value):
        self.value = value
        self.time_terms = []

    def add_time_term(self, term, mask=None):
        self.time_terms.append((term, mask))

    def remove_time_terms(self):
        self.time_terms = []

    def __call__(self, t):
        v = np.array(self.value, dtype=float)
        for term, _ in self.time_terms:
            v = v + np.array(term(t), dtype=float)
        return v


class _Variable:
    def __init__(self, shape):
        self._m = np.zeros(shape)

    def set(self, value):
        value = np.asarray(value, dtype=float)
        if value.shape != self._m.shape:
            raise ValueError(f"magnetization shape {value.shape} != {self._m.shape}")
        self._m = value.copy()

    def eval(self):
        return self._m.copy()

    def average(self):
        return self._m.reshape(3, -1).mean(axis=1)


class Ferromagnet:
    def __init__(self, world, grid, name="", geometry=None, regions=None):
        self.world, self.grid, self.name = world, grid, name
        self.geometry = None if geometry is None else np.asarray(geometry, dtype=bool)
        self.regions = None if regions is None else np.asarray(regions, dtype=int)
        for arr in (self.geometry, self.regions):
            if arr is not None and arr.shape != grid.shape:
                raise ValueError("geometry/regions shape mismatch")
        self._m = _Variable((3,) + grid.shape)
        self._bias = _Param(3)
        self.msat = self.aex = self.alpha = None
        world.ferromagnets[name] = self

    @property
    def magnetization(self):
        return self._m

    @magnetization.setter
    def magnetization(self, value):
        self._m.set(value)

    @property
    def bias_magnetic_field(self):
        return self._bias

    @bias_magnetic_field.setter
    def bias_magnetic_field(self, value):
        self._bias.set(value)

    def minimize(self, tol=1e-6, nsamples=10):
        self.world.minimize()


class TimeSolver:
    def __init__(self, world):
        self.world = world
        self.time = 0.0
        self.calls = 0

    def run(self, duration):
        self.time += duration

    def solve(self, timepoints, quantity_dict, file_name=None, store_as_dict=True, tqdm=False):
        assert self.time <= timepoints[0]
        out = {k: [] for k in quantity_dict}
        out["time"] = []
        for tp in timepoints:
            self.run(tp - self.time)
            # fake dynamics: tiny precession of mz driven by the excitation field
            for fm in self.world.ferromagnets.values():
                b = fm.bias_magnetic_field(self.time)
                m = fm.magnetization.eval()
                m[2] += 1e-3 * np.sin(2 * np.pi * 7e9 * self.time) * (fm.geometry if fm.geometry is not None else 1) + 0 * b[2]
                fm.magnetization.set(m)
            for k, q in quantity_dict.items():
                out[k].append(q())
            out["time"].append(self.time)
        self.calls += 1
        return out


class World:
    def __init__(self, cellsize, pbc_repetitions=(0, 0, 0), mastergrid=None):
        self.cellsize = tuple(cellsize)
        self.pbc_repetitions = tuple(pbc_repetitions)
        self.mastergrid = mastergrid
        self.ferromagnets = {}
        self._solver = TimeSolver(self)
        self.n_minimize = 0

    @property
    def timesolver(self):
        return self._solver

    def minimize(self, tol=1e-6, nsamples=10):
        self.n_minimize += 1
        # fake relaxation: align every magnetic cell with the in-plane bias field sign along x
        for fm in self.ferromagnets.values():
            b = np.array(fm.bias_magnetic_field(0.0))
            if np.linalg.norm(b[:2]) > 25e-3:
                m = fm.magnetization.eval()
                inside = fm.geometry if fm.geometry is not None else np.ones(m.shape[1:], bool)
                d = b[:2] / np.linalg.norm(b[:2])
                m[0][inside], m[1][inside], m[2][inside] = d[0], d[1], 0.0
                fm.magnetization.set(m)

    def relax(self, tol=1e-9):
        self.minimize()

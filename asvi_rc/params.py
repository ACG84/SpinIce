"""Physical and protocol parameters.

Defaults follow the Methods of Dion et al. 2024 (geometry, NiFe parameters,
sinc excitation) and Gartside et al. 2022 (reservoir protocol).  Everything is
in SI units (metres, tesla, seconds).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
import json
import math


@dataclass
class ASVIParams:
    """Geometry, material and discretisation of the 3D multilayered ASVI.

    A single 3D nanoisland is a NiFe(30 nm) / Al(35 nm) / NiFe(20 nm) stack.
    The lateral footprint is 550 nm x 140 nm; the top layer is laterally
    displaced by 50 nm along y ("shadow deposition" offset) which breaks the
    chiral symmetry of the vortex states.  Islands sit on a square ASI lattice
    with 125 nm gap between island end and vertex centre, i.e. a lattice
    constant of 550 + 2*125 = 800 nm.
    """

    # --- single island footprint -------------------------------------------------
    length: float = 550e-9
    width: float = 140e-9
    rounded_ends: bool = True          # stadium shape (True) or plain rectangle
    # --- vertical stack (bottom -> top) -------------------------------------------
    t_bottom: float = 30e-9            # "hard" bottom NiFe layer
    t_spacer: float = 35e-9            # non-magnetic Al spacer
    t_top: float = 20e-9               # "soft" top NiFe layer
    top_offset: tuple[float, float] = (0.0, 50e-9)  # lateral shift of the top layer (x, y)
    # --- lattice ----------------------------------------------------------------------
    vertex_gap: float = 125e-9         # island end -> vertex centre
    n_cells: tuple[int, int] = (2, 2)  # supercell size in unit cells (each unit cell = 2 islands)
    # --- discretisation ---------------------------------------------------------------
    cell_xy: float = 5e-9
    cell_z: float = 5e-9
    pbc_repetitions: tuple[int, int, int] = (4, 4, 0)
    # --- material (Ni81Fe19) -----------------------------------------------------------
    msat: float = 800e3
    aex: float = 13e-12
    alpha: float = 0.001
    # --- quenched disorder (lithographic imperfection) ------------------------------------
    # relative std of island width and length, drawn per 3D island (both layers
    # share the same footprint because they share the same resist mask).
    disorder_sigma: float = 0.02
    seed: int = 0
    # --- broadband FMR excitation --------------------------------------------------------
    fmr_fcut: float = 15e9             # sinc cut-off frequency
    fmr_amp: float = 1e-3              # sinc amplitude (T), applied along z
    fmr_duration: float = 26e-9
    fmr_dt: float = 33e-12             # sampling period (Nyquist 15.15 GHz)
    fmr_t0: float = 0.5e-9             # centre of the sinc pulse

    # ------------------------------------------------------------------------------------
    @property
    def lattice_constant(self) -> float:
        return self.length + 2 * self.vertex_gap

    @property
    def box(self) -> tuple[float, float]:
        """Lateral size of the periodic simulation box (m)."""
        a = self.lattice_constant
        return (self.n_cells[0] * a, self.n_cells[1] * a)

    @property
    def thickness(self) -> float:
        return self.t_bottom + self.t_spacer + self.t_top

    @property
    def grid(self) -> tuple[int, int, int]:
        """Number of cells (nx, ny, nz).  Raises if the box is not commensurate."""
        bx, by = self.box
        nx = _commensurate(bx, self.cell_xy, "box x")
        ny = _commensurate(by, self.cell_xy, "box y")
        nz = _commensurate(self.thickness, self.cell_z, "stack thickness")
        for name, t in (("t_bottom", self.t_bottom), ("t_spacer", self.t_spacer), ("t_top", self.t_top)):
            _commensurate(t, self.cell_z, name)
        return nx, ny, nz

    @property
    def z_layers(self) -> tuple[tuple[int, int], tuple[int, int]]:
        """Cell index ranges [start, stop) of (bottom, top) magnetic layers."""
        kb = int(round(self.t_bottom / self.cell_z))
        ks = int(round(self.t_spacer / self.cell_z))
        kt = int(round(self.t_top / self.cell_z))
        return (0, kb), (kb + ks, kb + ks + kt)

    @property
    def n_islands(self) -> int:
        return 2 * self.n_cells[0] * self.n_cells[1]

    @property
    def fmr_nt(self) -> int:
        return int(round(self.fmr_duration / self.fmr_dt)) + 1

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ASVIParams":
        d = dict(d)
        for k in ("top_offset", "n_cells", "pbc_repetitions"):
            if k in d:
                d[k] = tuple(d[k])
        return cls(**d)

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path) -> "ASVIParams":
        with open(path) as f:
            return cls.from_dict(json.load(f))


@dataclass
class ProtocolParams:
    """Reservoir-computing input protocol (Gartside et al. 2022, adapted).

    Each scalar input u_k in [0, 1] is mapped linearly onto a field amplitude
    B_k in [b_min, b_max].  A quasi-static minor field loop with that maximum
    amplitude is then applied along ``loop_angle`` (degrees from +x).  After
    the loop the field is set to the measurement field and a broadband FMR
    spectrum is computed - that spectrum is the reservoir output for step k.

    Loop shapes
    -----------
    ``"bipolar"``  : B_meas -> -B_k -> +B_k   (symmetric +-B_k minor loop,
                     ends at +B_k; measurement at +B_k unless a bias is given)
    ``"unipolar"`` : B_meas -> +B_k
    ``"return"``   : B_meas -> -B_k -> +B_k -> B_meas
    ``"leak"``     : B_meas -> +B_k -> -B_leak -> B_meas, measured at the bias
                     field.  Unipolar write followed by a fixed weak negative
                     "leak" that resets only the softest elements: elements
                     with switching field between B_leak and B_k keep a record
                     of recent inputs that exceeded their threshold (fading
                     memory of amplitudes).  Set ``leak_field`` between the
                     softest and hardest switching fields of the array.
    ``"alternating"``: B_meas -> s_k B_k -> B_meas with s_k = (-1)^k, measured at
                     the bias field.  Each layer-island then stores the sign
                     of the last input that exceeded its switching field, so
                     the microstate carries a fading memory of the input
                     history (a symmetric loop ending at +B_k erases it).

    The default field range 20-30 mT straddles the antiparallel->parallel
    switching window (26-30 mT in the experiment) of the multilayer islands,
    so the fraction of reversed islands, and hence the spectrum, depends
    non-linearly and hysteretically on the input history.  Calibrate it for
    your own geometry with the field-sweep scripts (see README).
    """

    b_min: float = 20e-3
    b_max: float = 30e-3
    loop_angle_deg: float = 1.0          # 1 deg off the x axis, as in the experiments
    loop_shape: str = "bipolar"
    loop_step: float = 1e-3              # quasi-static field increment inside a loop
    approach_step: float = 10e-3         # coarser increment when coming down from saturation
    coarse_step: float | None = None     # optional larger increment used while |B| < coarse_below
    coarse_below: float = 0.0            # (T) fine `loop_step` is used at and above this amplitude
    leak_field: float = 30e-3            # (T) magnitude of the negative reset field for loop_shape="leak"
    measure_at: str = "loop_max"         # "loop_max" or "bias"
    bias_field: float = -1.2e-3          # used when measure_at == "bias"
    saturation_field: float = 0.2        # initial saturation (-x) amplitude
    # spectral feature extraction
    f_min: float = 2e9
    f_max: float = 14e9
    f_bin: float = 40e6

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ProtocolParams":
        return cls(**d)

    @property
    def loop_direction(self) -> tuple[float, float]:
        th = math.radians(self.loop_angle_deg)
        return (math.cos(th), math.sin(th))


def _commensurate(length: float, cell: float, name: str) -> int:
    n = length / cell
    if abs(n - round(n)) > 1e-6:
        raise ValueError(f"{name} = {length:.4g} m is not an integer number of cells of {cell:.4g} m")
    return int(round(n))

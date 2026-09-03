"""Geometry of the multilayered square artificial spin-vortex ice.

Builds, on a periodic lateral box, the boolean geometry mask and the integer
region map (one region per magnetic layer of each 3D island) that both the
mumax3 script generator and the mumax+ driver consume.

Coordinates: the box is [0, Lx) x [0, Ly) with cell centres at (i+0.5)*cell.
Islands are wrapped periodically, so an island may straddle the box edge.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math

import numpy as np

from .params import ASVIParams

LAYERS = ("bottom", "top")


@dataclass
class Island:
    """One magnetic layer of one 3D nanoisland."""
    index: int                 # 3D island index (shared by both layers)
    layer: int                 # 0 = bottom, 1 = top
    cx: float                  # centre (m), already including the top-layer offset
    cy: float
    angle_deg: float           # 0 -> long axis along x, 90 -> along y
    length: float
    width: float

    @property
    def axis(self) -> tuple[float, float]:
        th = math.radians(self.angle_deg)
        return (math.cos(th), math.sin(th))

    @property
    def region(self) -> int:
        return region_id(self.index, self.layer)

    def to_dict(self) -> dict:
        return asdict(self)


def region_id(island_index: int, layer: int) -> int:
    """Region id used in mumax3/mumax+ (0 is reserved for 'no region')."""
    return 1 + 2 * island_index + layer


def build_islands(p: ASVIParams, rng: np.random.Generator | None = None) -> list[Island]:
    """Lay out the supercell.  Returns 2 layers x 2 islands x n_cells islands.

    Unit cell (lattice constant a): a horizontal island centred at (a/2, 0)
    and a vertical one centred at (0, a/2).  The whole pattern is shifted by
    (a/4, a/4) so that fewer islands straddle the box edge (cosmetic only:
    wrapping is handled everywhere).
    """
    if rng is None:
        rng = np.random.default_rng(p.seed)
    a = p.lattice_constant
    islands: list[Island] = []
    idx = 0
    for i in range(p.n_cells[0]):
        for j in range(p.n_cells[1]):
            for (ox, oy, ang) in ((0.5, 0.0, 0.0), (0.0, 0.5, 90.0)):
                cx = (i + ox + 0.25) * a
                cy = (j + oy + 0.25) * a
                # quenched disorder: both layers share the footprint
                L = p.length * (1 + p.disorder_sigma * rng.standard_normal())
                W = p.width * (1 + p.disorder_sigma * rng.standard_normal())
                for layer in (0, 1):
                    dx, dy = (0.0, 0.0) if layer == 0 else p.top_offset
                    islands.append(Island(idx, layer, cx + dx, cy + dy, ang, L, W))
                idx += 1
    return islands


def _local_coords(xx, yy, isl: Island, box):
    """Minimum-image coordinates (u along axis, v across) relative to island centre."""
    Lx, Ly = box
    dx = (xx - isl.cx + Lx / 2) % Lx - Lx / 2
    dy = (yy - isl.cy + Ly / 2) % Ly - Ly / 2
    c, s = isl.axis
    u = c * dx + s * dy
    v = -s * dx + c * dy
    return u, v


def island_mask_2d(xx, yy, isl: Island, box, rounded: bool = True):
    """Boolean lateral mask of one island on the meshgrid (xx, yy)."""
    u, v = _local_coords(xx, yy, isl, box)
    if rounded and isl.width < isl.length:
        half = (isl.length - isl.width) / 2
        r = isl.width / 2
        body = (np.abs(u) <= half) & (np.abs(v) <= r)
        caps = ((np.abs(u) - half) ** 2 + v ** 2 <= r ** 2) & (np.abs(u) > half)
        return body | caps
    return (np.abs(u) <= isl.length / 2) & (np.abs(v) <= isl.width / 2)


def lateral_meshgrid(p: ASVIParams):
    nx, ny, _ = p.grid
    x = (np.arange(nx) + 0.5) * p.cell_xy
    y = (np.arange(ny) + 0.5) * p.cell_xy
    return np.meshgrid(x, y, indexing="xy")   # shapes (ny, nx)


def build_geometry(p: ASVIParams, islands: list[Island] | None = None):
    """Return (mask, regions, islands).

    mask    : bool array (nz, ny, nx) - True inside magnetic material
    regions : int  array (nz, ny, nx) - region id per cell (0 outside)
    """
    if islands is None:
        islands = build_islands(p)
    nx, ny, nz = p.grid
    xx, yy = lateral_meshgrid(p)
    (b0, b1), (t0, t1) = p.z_layers
    mask = np.zeros((nz, ny, nx), dtype=bool)
    regions = np.zeros((nz, ny, nx), dtype=np.int32)
    for isl in islands:
        m2 = island_mask_2d(xx, yy, isl, p.box, p.rounded_ends)
        k0, k1 = (b0, b1) if isl.layer == 0 else (t0, t1)
        if np.any(mask[k0:k1] & m2[None]):
            raise ValueError(f"island {isl.index} layer {isl.layer} overlaps another island")
        mask[k0:k1] |= m2[None]
        regions[k0:k1][:, m2] = isl.region
    return mask, regions, islands


def region_masks(regions: np.ndarray, islands: list[Island]) -> dict[int, np.ndarray]:
    """Flat cell indices per region id (for fast spatial averaging)."""
    flat = regions.ravel()
    return {isl.region: np.flatnonzero(flat == isl.region) for isl in islands}


def macrospin_magnetization(p: ASVIParams, regions: np.ndarray, islands: list[Island],
                            signs: dict[int, float] | float = 1.0,
                            tilt: float = 0.0) -> np.ndarray:
    """Magnetisation array (3, nz, ny, nx) with each layer-island a macrospin.

    signs : +-1 per region id (dict) or a single value for all islands.  The
            sign multiplies the island axis, i.e. +1 = "along +axis".
    tilt  : small transverse component (breaks symmetry before relaxation).
    """
    m = np.zeros((3,) + regions.shape)
    for isl in islands:
        s = signs if isinstance(signs, (int, float)) else signs.get(isl.region, 1.0)
        c, sn = isl.axis
        sel = regions == isl.region
        m[0][sel] = s * c - tilt * sn
        m[1][sel] = s * sn + tilt * c
    return m


def uniform_magnetization(regions: np.ndarray, direction=(1.0, 0.0, 0.0)) -> np.ndarray:
    d = np.asarray(direction, dtype=float)
    d /= np.linalg.norm(d)
    m = np.zeros((3,) + regions.shape)
    inside = regions > 0
    for c in range(3):
        m[c][inside] = d[c]
    return m


def vortex_magnetization(p: ASVIParams, regions: np.ndarray, islands: list[Island],
                         chirality: dict[int, int] | int = 1, polarity: int = 1,
                         core_radius: float = 10e-9, base: np.ndarray | None = None) -> np.ndarray:
    """Seed a vortex (circulation +-1, core polarity +-1) in the chosen islands.

    chirality : dict region -> +-1 (islands not listed keep ``base``), or a
                single +-1 applied to every island.
    """
    xx, yy = lateral_meshgrid(p)
    m = np.array(base, copy=True) if base is not None else macrospin_magnetization(p, regions, islands)
    for isl in islands:
        ch = chirality if isinstance(chirality, int) else chirality.get(isl.region)
        if ch is None:
            continue
        u, v = _local_coords(xx, yy, isl, p.box)
        r = np.hypot(u, v) + 1e-30
        c, s = isl.axis
        # in-plane circulating field in local (u, v) coords, rotated back to (x, y)
        mu, mv = -ch * v / r, ch * u / r
        mx = c * mu - s * mv
        my = s * mu + c * mv
        mz = polarity * np.exp(-(r / core_radius) ** 2)
        scale = np.sqrt(1 - mz ** 2)
        sel = regions == isl.region
        for k in range(regions.shape[0]):
            sk = sel[k]
            m[0, k][sk] = (scale * mx)[sk]
            m[1, k][sk] = (scale * my)[sk]
            m[2, k][sk] = mz[sk]
    return m


def describe(p: ASVIParams, islands: list[Island]) -> str:
    nx, ny, nz = p.grid
    lines = [
        f"box {p.box[0]*1e9:.0f} x {p.box[1]*1e9:.0f} nm, grid {nx} x {ny} x {nz}"
        f" (cell {p.cell_xy*1e9:g} x {p.cell_xy*1e9:g} x {p.cell_z*1e9:g} nm)",
        f"{p.n_islands} 3D islands, {len(islands)} magnetic layer-islands, regions 1..{2*p.n_islands}",
    ]
    for isl in islands:
        lines.append(f"  region {isl.region:2d}: island {isl.index} {LAYERS[isl.layer]:6s} "
                     f"centre ({isl.cx*1e9:6.1f},{isl.cy*1e9:6.1f}) nm angle {isl.angle_deg:3.0f} "
                     f"L {isl.length*1e9:5.1f} W {isl.width*1e9:5.1f} nm")
    return "\n".join(lines)

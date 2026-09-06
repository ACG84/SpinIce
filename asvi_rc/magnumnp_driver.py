"""magnum.np (PyTorch) driver for the ASVI: runs on CPU or GPU and is differentiable.

Same interface as :class:`asvi_rc.mumaxplus_driver.ASVISimulation` (set/get
magnetization, set_field, minimize, total_energy, region_averages) so the
catalogue and transition-table scripts can run without a CUDA GPU, plus a
differentiable *soft geometry* in which island length/width, layer
thicknesses, offsets and the material constants are torch parameters.  The
saturation magnetisation of every cell is then a smooth function of those
parameters (an anti-aliased level-set mask: boundary cells carry the
fraction of their volume inside the island, everything else is 0 or 1), so
``torch.autograd`` gives d(energy)/d(design parameter).

Energies of relaxed states are stationary in m, so by the envelope theorem
dE_state/dp = (dE/dp)|_{m fixed}: one backward pass per state, no need to
differentiate through the minimiser.  ``energy_grad`` does exactly that.
"""
from __future__ import annotations

import math
import time

import numpy as np

from .geometry import build_geometry, build_islands, region_masks, uniform_magnetization
from .params import ASVIParams, ProtocolParams
from .tasks import field_vector

MU0 = 4e-7 * math.pi


def _import_magnumnp():
    try:
        import torch
        import magnumnp as mnp
    except ImportError as e:                                    # pragma: no cover
        raise ImportError("pip install torch magnumnp (CPU wheels are fine)") from e
    mnp.set_log_level(30)                                       # WARNING: silence per-step chatter
    return torch, mnp


# ------------------------------------------------------------------ soft geometry
def _ramp(s, w):
    """Kink-free occupation fraction as a function of the signed distance s (>0 outside).

    Exact area fraction of a cell of width w cut by a straight edge, averaged
    once more over the cell: a C1 quadratic spline going from 1 at s <= -w to 0
    at s >= w.  Being C1 it has no kinks where edges sit exactly on cell
    boundaries (torch.clamp would give a zero gradient there)."""
    import torch
    x = torch.clamp(s / w, -1.0, 1.0)
    return torch.where(x <= 0, 1 - (x + 1) ** 2 / 2, (1 - x) ** 2 / 2)


DESIGN_KEYS = ("length", "width", "t_bottom", "t_spacer", "t_top", "offset_x", "offset_y",
               "msat_bottom", "msat_top", "aex")


class SoftGeometry:
    """Differentiable cell-wise Ms and A for a set of islands.

    design : dict of torch scalars (any subset of DESIGN_KEYS); missing keys
             are taken from the ASVIParams.  Cells within 2*eps (lateral,
             default one cell) or 2*eps_z (vertical, default one cell) of an
             island boundary get a fractional occupation from the C1 spline
             ``_ramp``; all other cells are exactly 0 or 1, so the soft mask
             equals the hard mask of build_geometry away from boundaries.
    """

    def __init__(self, p: ASVIParams, islands, design: dict, eps: float | None = None,
                 eps_z: float | None = None, device="cpu"):
        torch, _ = _import_magnumnp()
        self.p, self.islands, self.design = p, islands, design
        self.eps = p.cell_xy / 2 if eps is None else eps
        self.eps_z = p.cell_z / 2 if eps_z is None else eps_z
        nx, ny, nz = p.grid
        x = (torch.arange(nx, dtype=torch.float64, device=device) + 0.5) * p.cell_xy
        y = (torch.arange(ny, dtype=torch.float64, device=device) + 0.5) * p.cell_xy
        z = (torch.arange(nz, dtype=torch.float64, device=device) + 0.5) * p.cell_z
        self.xx, self.yy = torch.meshgrid(x, y, indexing="ij")     # (nx, ny)
        self.zz = z
        self.device = device

    def par(self, key):
        if key in self.design:
            return self.design[key]
        p = self.p
        defaults = {"length": p.length, "width": p.width, "t_bottom": p.t_bottom,
                    "t_spacer": p.t_spacer, "t_top": p.t_top,
                    "offset_x": p.top_offset[0], "offset_y": p.top_offset[1],
                    "msat_bottom": p.msat, "msat_top": p.msat, "aex": p.aex}
        import torch
        return torch.tensor(defaults[key], dtype=torch.float64, device=self.device)

    def lateral(self, isl):
        """Soft stadium mask (nx, ny) of one layer-island."""
        import torch
        Lx, Ly = self.p.box
        cx, cy = isl.cx, isl.cy
        if isl.layer == 1:                                   # the top-layer offset is a design parameter
            cx = cx - self.p.top_offset[0] + self.par("offset_x")
            cy = cy - self.p.top_offset[1] + self.par("offset_y")
        dx = torch.remainder(self.xx - cx + Lx / 2, Lx) - Lx / 2
        dy = torch.remainder(self.yy - cy + Ly / 2, Ly) - Ly / 2
        c, s = isl.axis
        u = c * dx + s * dy
        v = -s * dx + c * dy
        L = self.par("length") * (isl.length / self.p.length)      # keep per-island disorder ratios
        W = self.par("width") * (isl.width / self.p.width)
        r = W / 2
        if self.p.rounded_ends:
            half = torch.clamp(L / 2 - r, min=0.0)
            du = torch.clamp(u.abs() - half, min=0.0)
            sdf = torch.sqrt(du ** 2 + v ** 2 + 1e-30) - r        # signed distance to the stadium
        else:
            sdf = torch.maximum(u.abs() - L / 2, v.abs() - r)
        return _ramp(sdf, 2 * self.eps)

    def vertical(self, layer):
        """Soft z-profile (nz,) of the bottom (0) or top (1) layer: linear ramps of width 2*eps_z
        (default 2 cells) across each interface, so the gradient w.r.t. a thickness is non-zero
        even when the interface sits exactly on a cell boundary."""
        import torch
        tb, ts, tt = self.par("t_bottom"), self.par("t_spacer"), self.par("t_top")
        bounds = [(0.0 * tb, tb), (tb + ts, tb + ts + tt)]
        z = tb + ts + tt
        for sp, t in self.p.extra_layers:                    # extra layers ride on top of the design stack
            bounds.append((z + sp, z + sp + t))
            z = z + sp + t
        z0, z1 = bounds[layer]
        w = 2 * self.eps_z
        return _ramp(z0 - self.zz, w) * _ramp(self.zz - z1, w)

    def densities(self):
        """List of per-island density fields rho_i (nx, ny, nz) in [0, 1]."""
        return [self.lateral(isl)[:, :, None] * self.vertical(isl.layer)[None, None, :] for isl in self.islands]

    def fields(self, rhos=None):
        """(Ms, A) tensors of shape (nx, ny, nz, 1) built from the design parameters."""
        import torch
        rhos = self.densities() if rhos is None else rhos
        nx, ny, nz = self.p.grid
        Ms = torch.zeros(nx, ny, nz, dtype=torch.float64, device=self.device)
        A = torch.zeros_like(Ms)
        aex = self.par("aex")
        for isl, rho in zip(self.islands, rhos):
            ms = self.par("msat_bottom" if isl.layer == 0 else "msat_top")
            Ms = Ms + ms * rho
            A = A + aex * rho
        return Ms.unsqueeze(-1), A.unsqueeze(-1)

    def regions(self, rhos=None, threshold: float = 0.5):
        """Hard (nz, ny, nx) region map and (nx, ny, nz, 3) axis-fill directions from the soft mask."""
        import torch
        rhos = self.densities() if rhos is None else rhos
        stack = torch.stack([r.detach() for r in rhos])                       # (n_isl, nx, ny, nz)
        best = stack.argmax(dim=0)
        inside = stack.max(dim=0).values > threshold
        ids = torch.tensor([isl.region for isl in self.islands], device=self.device)
        regions = torch.where(inside, ids[best], torch.zeros_like(best))
        axes = torch.tensor([[*isl.axis, 0.0] for isl in self.islands], dtype=torch.float64, device=self.device)
        fill = axes[best]                                                      # (nx, ny, nz, 3)
        return regions.cpu().numpy().transpose(2, 1, 0).astype(np.int32), fill


# ------------------------------------------------------------------ simulation
class ASVISimulationNP:
    """One magnum.np State holding the (optionally periodic) supercell."""

    def __init__(self, p: ASVIParams, islands=None, verbose: bool = True, device: str = "cpu",
                 design: dict | None = None, eps: float | None = None, dm_tol: float = 1e-2):
        torch, mnp = _import_magnumnp()
        self.torch, self.mnp = torch, mnp
        self.p = p
        self.mask, self.regions, self.islands = build_geometry(p, islands or build_islands(p))
        nx, ny, nz = p.grid
        self.mesh = mnp.Mesh((nx, ny, nz), (p.cell_xy, p.cell_xy, p.cell_z), pbc=tuple(p.pbc_repetitions))
        self.state = mnp.State(self.mesh)
        self.device = device
        self.dm_tol = dm_tol
        self.soft = None
        self._fill = None
        if design is not None:
            self.soft = SoftGeometry(p, self.islands, design, eps=eps, device=device)
            self.set_material()
        else:
            mask_t = torch.from_numpy(self.mask.transpose(2, 1, 0).astype(np.float64)).to(device)
            self.Ms = (p.msat * mask_t).unsqueeze(-1)
            self.A = (p.aex * mask_t).unsqueeze(-1)
            self.state.material = {"Ms": self.Ms, "A": self.A}
        self.demag = mnp.DemagField()
        self.exchange = mnp.ExchangeField()
        self.external = mnp.ExternalField([0.0, 0.0, 0.0])
        self.terms = [self.demag, self.exchange, self.external]
        self.region_ids = [isl.region for isl in self.islands]
        self._rmask = region_masks(self.regions, self.islands)
        self.verbose = verbose
        self._b_static = (0.0, 0.0, 0.0)
        self.state.m = self.state.Constant([0.0, 0.0, 0.0])
        self.set_magnetization(uniform_magnetization(self.regions))

    # ------------------------------------------------------------------ helpers
    def log(self, *a):
        if self.verbose:
            print(time.strftime("[%H:%M:%S]"), *a, flush=True)

    def set_material(self):
        """(Re)build Ms, A, the seed/classification regions and the fill directions from the design."""
        rhos = self.soft.densities()
        self.Ms, self.A = self.soft.fields(rhos)
        self.state.material = {"Ms": self.Ms, "A": self.A}
        self.regions, self._fill = self.soft.regions(rhos)
        self.mask = self.regions > 0
        self._rmask = region_masks(self.regions, self.islands)

    def _inside(self):
        return (self.Ms[..., 0] > 0).detach()

    # ------------------------------------------------------------------ state
    def set_magnetization(self, m: np.ndarray):
        torch = self.torch
        t = torch.from_numpy(np.ascontiguousarray(m.transpose(3, 2, 1, 0), dtype=np.float64)).to(self.device)
        n = torch.linalg.norm(t, dim=-1, keepdim=True)
        t = torch.where(n > 0, t / n.clamp(min=1e-30), torch.zeros_like(t))
        inside = self._inside()
        if self._fill is not None:                       # soft mask wider than the seed regions
            t = self._dilate(t, inside)
        t[~inside] = 0.0
        self.state.m = t

    def _dilate(self, t, inside, n_iter: int = 4):
        """Give magnetic cells that carry no seed (partial boundary cells of the soft mask) the
        average direction of their seeded neighbours, so a vortex or a macrospin extends smoothly
        into the fractional cells; cells still empty afterwards fall back to the island axis."""
        torch = self.torch
        for _ in range(n_iter):
            has = (torch.linalg.norm(t, dim=-1) > 0)
            empty = inside & ~has
            if not bool(empty.any()):
                break
            acc = torch.zeros_like(t)
            for ax in range(3):
                for sh in (1, -1):
                    acc = acc + torch.roll(t, sh, dims=ax)
            nrm = torch.linalg.norm(acc, dim=-1, keepdim=True)
            new = torch.where(nrm > 0, acc / nrm.clamp(min=1e-30), torch.zeros_like(acc))
            t = torch.where(empty[..., None], new, t)
        empty = inside & (torch.linalg.norm(t, dim=-1) == 0)
        t[empty] = self._fill[empty]
        return t

    def get_magnetization(self) -> np.ndarray:
        return self.state.m.detach().cpu().numpy().transpose(3, 2, 1, 0).copy()

    def region_averages(self, m: np.ndarray | None = None) -> np.ndarray:
        m = self.get_magnetization() if m is None else m
        flat = m.reshape(3, -1)
        return np.stack([flat[:, self._rmask[r]].mean(axis=1) for r in self.region_ids])

    def set_field(self, b):
        self._b_static = tuple(float(x) for x in b)
        self.external.h = [x / MU0 for x in self._b_static]

    # ------------------------------------------------------------------ energy
    def total_energy(self) -> float:
        with self.torch.no_grad():
            return float(sum(t.E(self.state) for t in self.terms))

    def exchange_energy(self):
        """Exchange energy sum_bonds A_bond |m_i - m_j|^2 V / d^2 with the harmonic-mean bond
        stiffness, written without any division by Ms or A so that autograd stays finite in
        empty cells (magnum.np's ExchangeField.h divides by Ms, which is NaN-safe only forward)."""
        torch = self.torch
        A, m = self.A[..., 0], self.state.m
        V = self.mesh.cell_volumes
        E = 0.0
        for ax, d in enumerate((self.p.cell_xy, self.p.cell_xy, self.p.cell_z)):
            if self.mesh.n[ax] == 1:
                continue
            if self.mesh.pbc[ax]:
                A2, m2 = torch.roll(A, -1, dims=ax), torch.roll(m, -1, dims=ax)
                A1, m1 = A, m
            else:
                sl1 = [slice(None)] * 3; sl2 = [slice(None)] * 3
                sl1[ax], sl2[ax] = slice(None, -1), slice(1, None)
                A1, A2 = A[tuple(sl1)], A[tuple(sl2)]
                m1, m2 = m[tuple(sl1)], m[tuple(sl2)]
            A_bond = 2 * A1 * A2 / (A1 + A2 + 1e-30)
            E = E + (A_bond * ((m1 - m2) ** 2).sum(-1)).sum() * V / d ** 2
        return E

    def energy_tensor(self):
        """Total energy as a torch scalar attached to the design-parameter graph."""
        if self.soft is None:
            return sum(t.E(self.state) for t in self.terms)
        return self.demag.E(self.state) + self.external.E(self.state) + self.exchange_energy()

    def energy_grad(self, keys=None) -> dict:
        """dE/d(design parameter) at the current (relaxed) magnetisation."""
        torch = self.torch
        assert self.soft is not None, "energy_grad needs a soft geometry (design=...)"
        keys = list(self.soft.design) if keys is None else list(keys)
        pars = [self.soft.design[k] for k in keys]
        self.set_material()
        E = self.energy_tensor()
        grads = torch.autograd.grad(E, pars, allow_unused=True)
        return {k: (0.0 if g is None else float(g)) for k, g in zip(keys, grads)}

    def moment(self):
        """Total magnetic moment vector (A m^2) as a torch tensor (differentiable)."""
        return (self.Ms * self.state.m).sum(dim=(0, 1, 2)) * self.mesh.cell_volumes

    # ------------------------------------------------------------------ relaxation
    def minimize(self, robust: bool = False, maxiter: int | None = None):
        torch, mnp = self.torch, self.mnp
        mn = mnp.MinimizerBB(self.terms)
        with torch.no_grad():
            m = self.state.m.detach().clone()
            m[~self._inside()] = 0.0
            self.state.m = m
            ok = mn.minimize(self.state, maxiter=maxiter or (6000 if robust else 3000),
                             dm_tol=self.dm_tol, tau_max=1e-5 if not robust else 3e-6)
            if robust and not ok:
                mn.minimize(self.state, maxiter=6000, dm_tol=self.dm_tol, tau_max=1e-6)
        return ok

    def saturate(self, proto: ProtocolParams | None = None, direction=None, amplitude=None):
        cx, cy = proto.loop_direction if direction is None else direction
        amp = proto.saturation_field if amplitude is None else amplitude
        self.set_magnetization(uniform_magnetization(self.regions, (-cx, -cy, 0.0)))
        self.set_field((-amp * cx, -amp * cy, 0.0))
        self.minimize()
        self.log("saturated at", self._b_static)

    def apply_ramp(self, ramp, proto: ProtocolParams, robust: bool = False):
        for b in ramp:
            self.set_field(field_vector(float(b), proto))
            self.minimize(robust)

    def save_state(self, path):
        np.save(path, self.get_magnetization())

    def load_state(self, path):
        self.set_magnetization(np.load(path))

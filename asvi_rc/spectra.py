"""Turn time-resolved magnetisation into FMR power spectra and feature vectors.

Both simulators produce, for every reservoir step, the spatially averaged
magnetisation of every region (one region per magnetic layer of every 3D
island) sampled every ``fmr_dt`` during the sinc-excited run.  Following the
Methods of Dion et al. the static (t=0) magnetisation is subtracted, the
result is Fourier transformed along time and the power |FFT|^2 is summed
over components.  Summing the power over regions gives the "global" FMR
spectrum that a flip-chip CPW measurement sees; keeping regions separate
gives a spatially resolved fingerprint (more reservoir outputs).
"""
from __future__ import annotations

import re

import numpy as np


def power_spectrum(m_t: np.ndarray, dt: float, subtract_static: bool = True,
                   window: str | None = "hann", components=(0, 1, 2)):
    """Power spectra of region-averaged magnetisation.

    m_t : array (nt, nregions, 3)
    Returns (freqs (nf,), power (nf, nregions)).
    """
    m = np.asarray(m_t, dtype=float)
    if m.ndim == 2:            # single region (nt, 3)
        m = m[:, None, :]
    if subtract_static:
        m = m - m[:1]
    nt = m.shape[0]
    if window == "hann":
        w = np.hanning(nt)[:, None, None]
    elif window is None:
        w = 1.0
    else:
        raise ValueError(window)
    spec = np.fft.rfft(m * w, axis=0)
    power = np.sum(np.abs(spec[:, :, list(components)]) ** 2, axis=2)
    freqs = np.fft.rfftfreq(nt, dt)
    return freqs, power


def bin_spectrum(freqs: np.ndarray, power: np.ndarray, f_min: float, f_max: float,
                 f_bin: float) -> tuple[np.ndarray, np.ndarray]:
    """Average ``power`` (nf, ...) into uniform frequency bins on [f_min, f_max].

    Returns (bin_centres (nb,), binned (nb, ...)).  Bins that contain no FFT
    sample are linearly interpolated from the raw spectrum.
    """
    edges = np.arange(f_min, f_max + 0.5 * f_bin, f_bin)
    centres = 0.5 * (edges[1:] + edges[:-1])
    idx = np.digitize(freqs, edges) - 1
    out = np.zeros((len(centres),) + power.shape[1:])
    for b in range(len(centres)):
        sel = idx == b
        if np.any(sel):
            out[b] = power[sel].mean(axis=0)
        else:
            out[b] = np.array([np.interp(centres[b], freqs, power[:, j])
                               for j in range(power.reshape(len(freqs), -1).shape[1])]
                              ).reshape(power.shape[1:])
    return centres, out


def features(freqs: np.ndarray, power: np.ndarray, f_min: float, f_max: float, f_bin: float,
             per_region: bool = False, log: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Feature vector for one reservoir step.

    power : (nf, nregions).  If ``per_region`` the binned spectra of all
    regions are concatenated, otherwise the region-summed spectrum is used.
    ``log`` applies log10(1 + P/max) which mimics the dynamic range of a
    lock-in FMR measurement and stabilises the ridge regression.
    """
    centres, binned = bin_spectrum(freqs, power, f_min, f_max, f_bin)
    if per_region:
        vec = binned.reshape(len(centres), -1).T.ravel()
    else:
        vec = binned.sum(axis=1)
    if log:
        vec = np.log10(1 + vec / (np.max(vec) + 1e-300) * 1e4)
    return centres, vec


def reservoir_features(freqs, power_steps, proto, per_region=False, log=True) -> np.ndarray:
    """Stack features of all steps: power_steps (nsteps, nf, nregions) -> (nsteps, nfeat)."""
    X = [features(freqs, P, proto.f_min, proto.f_max, proto.f_bin, per_region, log)[1]
         for P in power_steps]
    return np.vstack(X)


# ---------------------------------------------------------------------------
# mumax3 table parsing
# ---------------------------------------------------------------------------
def read_mumax3_table(path) -> dict[str, np.ndarray]:
    """Parse a mumax3 table.txt into a dict column-name -> array."""
    with open(path) as f:
        header = f.readline()
    if not header.startswith("#"):
        raise ValueError("not a mumax3 table (missing header line)")
    names = [re.sub(r"\s*\(.*?\)\s*$", "", c.strip()) for c in header[1:].strip().split("\t")]
    data = np.loadtxt(path, comments="#", ndmin=2)
    if data.shape[1] != len(names):
        raise ValueError(f"header has {len(names)} columns, data {data.shape[1]}")
    return {n: data[:, i] for i, n in enumerate(names)}


def spectra_from_mumax3_table(path, region_ids, dt: float, nt: int | None = None,
                              step_col: str = "step"):
    """Split a reservoir table into steps and compute per-step spectra.

    The generated .mx3 scripts write ``m.regionN{x,y,z}`` columns plus a
    ``step`` counter and ``B_loop``.  Returns (freqs, power (nsteps, nf, nreg),
    steps, b_loop).
    """
    tab = read_mumax3_table(path)
    steps = tab[step_col]
    uniq = np.unique(steps)
    power_all, b_loop = [], []
    freqs = None
    for s in uniq:
        sel = steps == s
        m_t = np.stack([np.stack([tab[f"m.region{r}{c}"][sel] for c in "xyz"], axis=-1)
                        for r in region_ids], axis=1)         # (nt, nreg, 3)
        if nt is not None:
            m_t = m_t[:nt]
        freqs, P = power_spectrum(m_t, dt)
        power_all.append(P)
        b_loop.append(tab["B_loop"][sel][0] if "B_loop" in tab else np.nan)
    return freqs, np.stack(power_all), uniq, np.array(b_loop)


def save_spectra(path, freqs, power, b_loop, u=None, y=None, **meta):
    np.savez_compressed(path, freqs=freqs, power=power, b_loop=b_loop,
                        u=np.array([]) if u is None else u,
                        y=np.array([]) if y is None else y, **meta)


def load_spectra(path) -> dict:
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}

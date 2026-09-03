"""asvi_rc - reservoir computing with a 3D multilayered artificial spin-vortex ice.

Simulation-side tooling for the system of Dion et al., Nat. Commun. 15, 4077 (2024)
(doi:10.1038/s41467-024-48080-z) driven with the spin-wave-fingerprint reservoir
computing protocol of Gartside et al., Nat. Nanotechnol. 17, 460 (2022).

The package is split into GPU-free parts (geometry, input protocol, spectra,
readout training) that are shared by the mumax3 and mumax+ front-ends, and the
front-ends themselves (``mumax3_writer`` emits .mx3 scripts, ``mumaxplus_driver``
drives the mumax+ Python API).
"""

from .params import ASVIParams, ProtocolParams
from .geometry import Island, build_islands, build_geometry, region_id
from . import tasks, spectra, readout

__all__ = [
    "ASVIParams", "ProtocolParams", "Island", "build_islands", "build_geometry",
    "region_id", "tasks", "spectra", "readout",
]
__version__ = "0.1.0"

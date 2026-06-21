"""JET/DIII-D EFIT equilibrium loader via OMAS.

OMAS and MDSplus are optional dependencies (``pip install plasma-gsfno[gw]``).
This module is importable even when OMAS is absent; :class:`EFITLoader` will
raise ``ImportError`` at instantiation time if OMAS is not installed.

.. warning::
    This loader predates the forward-operator redesign (branch
    ``redesign-forward-operator``) and is NOT compatible with the new
    5-channel contract ``[psi_vac, R_norm, Z_norm, pprime_lift,
    ffprime_lift]`` with global dimensionless scaling and no mask channel.
    Calling :meth:`EFITLoader.load_shot` or
    :meth:`EFITLoader._process_time_slice` will raise
    ``NotImplementedError`` until this loader is migrated.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional OMAS import
# ---------------------------------------------------------------------------
try:
    import omas  # type: ignore

    OMAS_AVAILABLE = True
except ImportError:
    OMAS_AVAILABLE = False


class EFITLoader:
    """Load real EFIT equilibrium reconstructions from JET or DIII-D via OMAS.

    Produces samples in the same format as PlasmaConfigGenerator so they can be
    used directly with GradShafranovDataset / write_hdf5.

    Parameters
    ----------
    machine:
        One of ``"jet"`` or ``"diiid"``.
    NR, NZ:
        Target grid resolution (default 65×65).
    R_min, R_max, Z_min, Z_max:
        Physical domain bounds of the output grid in metres.
    """

    SUPPORTED_MACHINES = ("jet", "diiid")

    def __init__(
        self,
        machine: str = "jet",
        NR: int = 65,
        NZ: int = 65,
        R_min: float = 0.5,
        R_max: float = 2.5,
        Z_min: float = -1.5,
        Z_max: float = 1.5,
    ) -> None:
        if machine not in self.SUPPORTED_MACHINES:
            raise ValueError(f"machine must be one of {self.SUPPORTED_MACHINES}")
        if not OMAS_AVAILABLE:
            raise ImportError(
                "OMAS is required for real data loading. "
                "Install with: pip install omas"
            )
        self.machine = machine
        self.NR = NR
        self.NZ = NZ
        self.R_min = R_min
        self.R_max = R_max
        self.Z_min = Z_min
        self.Z_max = Z_max
        self._R_out = np.linspace(R_min, R_max, NR)
        self._Z_out = np.linspace(Z_min, Z_max, NZ)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    _REDESIGN_MSG = (
        "EFITLoader predates the forward-operator redesign and must be migrated "
        "to the new 5-channel contract [psi_vac, R_norm, Z_norm, pprime_lift, "
        "ffprime_lift] with global dimensionless scaling and no boundary_mask "
        "channel before use.  Real-machine support is out of scope for the "
        "redesign-forward-operator branch."
    )

    def load_shot(self, shot: int, time_indices: Optional[list[int]] = None) -> list[dict]:
        """Load equilibrium time slices from a single shot.

        .. deprecated::
            Raises ``NotImplementedError`` — see module docstring.
        """
        raise NotImplementedError(self._REDESIGN_MSG)

    # ------------------------------------------------------------------
    # ODS loading
    # ------------------------------------------------------------------

    def _load_ods(self, shot: int) -> "omas.ODS":  # type: ignore[name-defined]
        """Load ODS from machine MDS+ server."""
        if self.machine == "jet":
            return omas.load_omas_mds(machine="jet", pulse=shot, imas_version="3")
        elif self.machine == "diiid":
            return omas.load_omas_mds(machine="d3d", pulse=shot, imas_version="3")

    # ------------------------------------------------------------------
    # Time-slice processing
    # ------------------------------------------------------------------

    def _process_time_slice(self, ods, t_idx: int) -> Optional[dict]:
        """Extract and normalise one time slice from ODS.

        .. deprecated::
            Raises ``NotImplementedError`` — see module docstring.
        """
        raise NotImplementedError(self._REDESIGN_MSG)

    # ------------------------------------------------------------------
    # Interpolation helper
    # ------------------------------------------------------------------

    def _interpolate_to_grid(
        self, psi: np.ndarray, R_src: np.ndarray, Z_src: np.ndarray
    ) -> np.ndarray:
        """Interpolate psi from source grid to standard (NR, NZ) output grid.

        Parameters
        ----------
        psi:
            Source psi array, shape ``(len(R_src), len(Z_src))``.
        R_src:
            R coordinates of source grid, shape ``(NR_src,)``.
        Z_src:
            Z coordinates of source grid, shape ``(NZ_src,)``.

        Returns
        -------
        Interpolated psi on the output grid, shape ``(NR, NZ)``.
        """
        from scipy.interpolate import RegularGridInterpolator

        interp = RegularGridInterpolator(
            (R_src, Z_src),
            psi,
            method="linear",
            bounds_error=False,
            fill_value=float(psi.max()),
        )
        R_out_2d, Z_out_2d = np.meshgrid(self._R_out, self._Z_out, indexing="ij")
        pts = np.stack([R_out_2d.ravel(), Z_out_2d.ravel()], axis=1)
        return interp(pts).reshape(self.NR, self.NZ)

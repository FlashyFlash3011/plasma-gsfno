"""JET/DIII-D EFIT equilibrium loader via OMAS.

OMAS and MDSplus are optional dependencies (``pip install plasma-gsfno[gw]``).
This module is importable even when OMAS is absent; :class:`EFITLoader` will
raise ``ImportError`` at instantiation time if OMAS is not installed.
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

    def load_shot(self, shot: int, time_indices: Optional[list[int]] = None) -> list[dict]:
        """Load equilibrium time slices from a single shot.

        Parameters
        ----------
        shot:
            Shot number (e.g. 92213 for JET, 145098 for DIII-D).
        time_indices:
            List of time slice indices to load. If *None*, loads all available.

        Returns
        -------
        List of sample dicts (same format as ``PlasmaConfigGenerator.generate_one``):
        ``{'inputs': (5, NR, NZ) float32, 'psi': (1, NR, NZ) float32, 'params': dict}``.
        Samples that fail to load are skipped with a warning.
        """
        ods = self._load_ods(shot)

        all_slices = ods["equilibrium.time_slice"]
        if time_indices is None:
            time_indices = list(all_slices.keys())

        samples: list[dict] = []
        for t_idx in time_indices:
            result = self._process_time_slice(ods, t_idx)
            if result is not None:
                samples.append(result)

        if not samples:
            logger.warning("No valid time slices loaded for shot %d.", shot)

        return samples

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

        Parameters
        ----------
        ods:
            OMAS ODS (or compatible dict-like) object.
        t_idx:
            Time slice index.

        Returns
        -------
        Sample dict or *None* on failure.
        """
        try:
            ts = ods["equilibrium.time_slice"][t_idx]
            psi_2d = np.array(ts["profiles_2d"][0]["psi"])           # (NR_src, NZ_src)
            R_src = np.array(ts["profiles_2d"][0]["grid"]["dim1"])   # (NR_src,)
            Z_src = np.array(ts["profiles_2d"][0]["grid"]["dim2"])   # (NZ_src,)
            Ip = float(ts["global_quantities"]["ip"])
        except (KeyError, IndexError) as exc:
            logger.warning("Time slice %d missing data: %s", t_idx, exc)
            return None

        # Interpolate to standard grid
        psi_interp = self._interpolate_to_grid(psi_2d, R_src, Z_src)

        # Normalise psi to [0, 1]
        psi_min, psi_max = psi_interp.min(), psi_interp.max()
        eps = 1e-10
        psi_norm = (psi_interp - psi_min) / (psi_max - psi_min + eps)

        # Build normalised coordinate grids
        R_norm = (self._R_out - self.R_min) / (self.R_max - self.R_min)
        Z_norm = (self._Z_out - self.Z_min) / (self.Z_max - self.Z_min)
        R_grid_2d, Z_grid_2d = np.meshgrid(R_norm, Z_norm, indexing="ij")

        # Boundary mask: approximate as psi < 0.99 (LCFS at normalised psi = 1)
        boundary_mask = (psi_norm < 0.99).astype(np.float32)

        inputs = np.stack(
            [
                boundary_mask,
                R_grid_2d.astype(np.float32),
                Z_grid_2d.astype(np.float32),
                np.zeros((self.NR, self.NZ), dtype=np.float32),  # p'  unknown for EFIT
                np.zeros((self.NR, self.NZ), dtype=np.float32),  # ff' unknown for EFIT
            ]
        ).astype(np.float32)  # (5, NR, NZ)

        return {
            "inputs": inputs,
            "psi": psi_norm[np.newaxis].astype(np.float32),  # (1, NR, NZ)
            "params": {
                "machine": self.machine,
                "shot": float(t_idx),
                "Ip": float(Ip),
                "source": "efit",
            },
        }

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

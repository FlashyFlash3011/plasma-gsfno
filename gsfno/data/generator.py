"""FreeGS wrapper for generating synthetic Grad-Shafranov equilibria as training data.

FreeGS is an optional dependency (``pip install plasma-gsfno[gw]``).
The class :class:`PlasmaConfigGenerator` is importable even when FreeGS is absent;
:meth:`generate_one` / :meth:`generate_batch` will raise ``ImportError`` at call time.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional FreeGS import
# ---------------------------------------------------------------------------
try:
    import freegs  # type: ignore

    FREEGS_AVAILABLE = True
except ImportError:
    FREEGS_AVAILABLE = False

# ---------------------------------------------------------------------------
# Grid constants
# ---------------------------------------------------------------------------
_R_MIN_DEFAULT = 0.5
_R_MAX_DEFAULT = 2.5
_Z_MIN_DEFAULT = -1.5
_Z_MAX_DEFAULT = 1.5
_NR_DEFAULT = 65
_NZ_DEFAULT = 65

# Floating-point guard for normalization
_EPS = 1e-8


class PlasmaConfigGenerator:
    """Generate synthetic Grad-Shafranov equilibria via FreeGS.

    Parameters
    ----------
    NR, NZ:
        Grid resolution (default 65×65).
    R_min, R_max, Z_min, Z_max:
        Physical domain bounds in metres.
    seed:
        Optional RNG seed for reproducibility.
    """

    def __init__(
        self,
        NR: int = _NR_DEFAULT,
        NZ: int = _NZ_DEFAULT,
        R_min: float = _R_MIN_DEFAULT,
        R_max: float = _R_MAX_DEFAULT,
        Z_min: float = _Z_MIN_DEFAULT,
        Z_max: float = _Z_MAX_DEFAULT,
        seed: Optional[int] = None,
    ) -> None:
        self.NR = NR
        self.NZ = NZ
        self.R_min = R_min
        self.R_max = R_max
        self.Z_min = Z_min
        self.Z_max = Z_max

        self.rng = np.random.default_rng(seed)

        # Pre-compute coordinate grids (shape: NR, NZ)
        R_vals = np.linspace(R_min, R_max, NR, dtype=np.float32)
        Z_vals = np.linspace(Z_min, Z_max, NZ, dtype=np.float32)
        self.R_grid, self.Z_grid = np.meshgrid(R_vals, Z_vals, indexing="ij")

        # Normalised coordinate grids [0, 1]
        self._R_norm = (self.R_grid - R_min) / (R_max - R_min)
        self._Z_norm = (self.Z_grid - Z_min) / (Z_max - Z_min)

    # ------------------------------------------------------------------
    # Parameter sampling
    # ------------------------------------------------------------------

    def _sample_params(self) -> dict:
        """Sample one random plasma configuration.

        Returns a dict with keys: ``R0``, ``a``, ``kappa``, ``delta``,
        ``Ip``, ``p_scale``, ``ff_scale``.
        """
        rng = self.rng

        R0 = float(rng.uniform(1.0, 2.0))
        # minor radius must satisfy a < R0/2 and be in [0.3, 0.8]
        a_max = min(0.8, R0 / 2.0 - 1e-3)
        a_min = 0.3
        if a_min >= a_max:
            a_min = a_max * 0.5  # fallback for very small R0
        a = float(rng.uniform(a_min, a_max))

        kappa = float(rng.uniform(1.0, 2.5))
        delta = float(rng.uniform(0.0, 0.6))
        Ip = float(rng.uniform(1e5, 5e6))
        p_scale = float(rng.uniform(1e2, 1e4))
        ff_scale = float(rng.uniform(0.1, 1.0))

        return {
            "R0": R0,
            "a": a,
            "kappa": kappa,
            "delta": delta,
            "Ip": Ip,
            "p_scale": p_scale,
            "ff_scale": ff_scale,
        }

    # ------------------------------------------------------------------
    # Profile evaluation
    # ------------------------------------------------------------------

    def _make_profiles(
        self, params: dict, R_grid: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate p'(R) and ff'(R) on *R_grid*.

        Parameters
        ----------
        params:
            Parameter dict from :meth:`_sample_params`.
        R_grid:
            Array of R coordinates (any shape; profiles evaluated element-wise).

        Returns
        -------
        p_prime, ff_prime:
            Arrays of the same shape as *R_grid*, dtype float32, values ≥ 0.
        """
        R0 = params["R0"]
        a = params["a"]
        p_scale = params["p_scale"]
        ff_scale = params["ff_scale"]

        xi = ((R_grid - R0) / a) ** 2  # dimensionless radial variable

        # Smooth bump: max(0, 1 - xi)^2
        inner = np.maximum(0.0, 1.0 - xi)
        p_prime = (p_scale * inner**2).astype(np.float32)

        # Triangular bump: max(0, 1 - xi)
        ff_prime = (ff_scale * inner).astype(np.float32)

        return p_prime, ff_prime

    # ------------------------------------------------------------------
    # Boundary mask
    # ------------------------------------------------------------------

    def _boundary_mask(self, psi: np.ndarray, eq) -> np.ndarray:  # type: ignore[type-arg]
        """Compute binary mask: 1 inside separatrix, 0 outside.

        Uses the O-point / X-point psi values from the equilibrium object.
        Falls back to a simple threshold if FreeGS psi_bndry is unavailable.
        """
        try:
            psi_axis = float(eq.psi_axis)
            psi_bndry = float(eq.psi_bndry)
        except AttributeError:
            # Fallback: use median as a crude boundary threshold
            psi_bndry = float(np.median(psi))
            psi_axis = float(psi.min())

        # Inside separatrix: psi between axis and boundary (handle both signs)
        if psi_axis < psi_bndry:
            mask = (psi >= psi_axis) & (psi <= psi_bndry)
        else:
            mask = (psi <= psi_axis) & (psi >= psi_bndry)

        return mask.astype(np.float32)

    # ------------------------------------------------------------------
    # Single-sample generation
    # ------------------------------------------------------------------

    def generate_one(self, params: Optional[dict] = None) -> Optional[dict]:
        """Generate one equilibrium sample using FreeGS.

        Parameters
        ----------
        params:
            Pre-sampled parameter dict. If *None*, :meth:`_sample_params`
            is called internally.

        Returns
        -------
        dict with keys ``inputs`` (5, NR, NZ), ``psi`` (1, NR, NZ), ``params``.
        Returns *None* if the FreeGS solve fails for this parameter set.

        Raises
        ------
        ImportError
            If FreeGS is not installed.
        """
        if not FREEGS_AVAILABLE:
            raise ImportError(
                "FreeGS is not installed. Install it with: pip install plasma-gsfno[gw]"
            )

        if params is None:
            params = self._sample_params()

        R0 = params["R0"]
        a = params["a"]
        kappa = params["kappa"]
        delta = params["delta"]
        Ip = params["Ip"]
        p_scale = params["p_scale"]

        try:
            # Build shaped boundary (D-shaped cross-section)
            theta = np.linspace(0, 2 * np.pi, 129)[:-1]  # 128 points
            R_boundary = R0 + a * np.cos(theta + delta * np.sin(theta))
            Z_boundary = kappa * a * np.sin(theta)

            boundary = freegs.machine.Wall(R_boundary, Z_boundary)
            tokamak = freegs.machine.Machine(coils=[], wall=boundary)

            # Create equilibrium grid
            eq = freegs.Equilibrium(
                tokamak=tokamak,
                Rmin=self.R_min,
                Rmax=self.R_max,
                Zmin=self.Z_min,
                Zmax=self.Z_max,
                nx=self.NR,
                ny=self.NZ,
                boundary=freegs.boundary.freeBoundaryHagenow,
            )

            # Profiles: pressure on axis and vacuum f = R*Btor
            paxis = float(p_scale)  # pa ~ p_scale as axis pressure proxy
            fvac = R0 * 2.0  # vacuum toroidal field factor (simplified)

            profiles = freegs.jtor.ConstrainPaxisIp(paxis, Ip, fvac)

            # Solve
            freegs.solve(eq, profiles, constrain=None, maxits=25, atol=1e-4)

            # Extract raw psi (NR, NZ)
            psi_raw = eq.psi()  # numpy array from FreeGS
            if psi_raw.shape != (self.NR, self.NZ):
                psi_raw = psi_raw.T  # FreeGS may return (NZ, NR) in some versions

        except Exception as exc:
            logger.warning("FreeGS solve failed (params=%s): %s", params, exc)
            return None

        # ------------------------------------------------------------------
        # Build input channels on (NR, NZ) grid
        # ------------------------------------------------------------------

        # Boundary mask (1 inside separatrix)
        mask = self._boundary_mask(psi_raw, eq)

        # Profile channels evaluated on R_grid (NR, NZ)
        p_prime, ff_prime = self._make_profiles(params, self.R_grid)

        # Normalise profiles per-sample by max absolute value
        p_max = float(np.max(np.abs(p_prime))) + _EPS
        ff_max = float(np.max(np.abs(ff_prime))) + _EPS
        p_prime_norm = (p_prime / p_max).astype(np.float32)
        ff_prime_norm = (ff_prime / ff_max).astype(np.float32)

        # Stack input channels: (5, NR, NZ)
        inputs = np.stack(
            [
                mask,               # ch0: boundary mask
                self._R_norm,       # ch1: R normalised [0,1]
                self._Z_norm,       # ch2: Z normalised [0,1]
                p_prime_norm,       # ch3: p'(R) normalised
                ff_prime_norm,      # ch4: ff'(R) normalised
            ],
            axis=0,
        ).astype(np.float32)

        # Normalise psi to [0, 1] per-sample
        psi_min = float(psi_raw.min())
        psi_max = float(psi_raw.max())
        psi_norm = ((psi_raw - psi_min) / (psi_max - psi_min + _EPS)).astype(np.float32)
        psi_out = psi_norm[np.newaxis, :, :]  # (1, NR, NZ)

        return {
            "inputs": inputs,
            "psi": psi_out,
            "params": dict(params),
        }

    # ------------------------------------------------------------------
    # Batch generation
    # ------------------------------------------------------------------

    def generate_batch(self, n: int, max_attempts: Optional[int] = None) -> list[dict]:
        """Generate *n* valid samples, retrying on FreeGS failures.

        Parameters
        ----------
        n:
            Number of valid samples to collect.
        max_attempts:
            Maximum total solver calls before giving up. Defaults to ``3*n``.

        Returns
        -------
        List of sample dicts (length ≤ n; may be less if max_attempts reached).

        Raises
        ------
        ImportError
            If FreeGS is not installed (raised immediately, before any attempts).
        """
        if not FREEGS_AVAILABLE:
            raise ImportError(
                "FreeGS is not installed. Install it with: pip install plasma-gsfno[gw]"
            )

        if max_attempts is None:
            max_attempts = 3 * n

        samples: list[dict] = []
        attempts = 0

        while len(samples) < n and attempts < max_attempts:
            result = self.generate_one()
            attempts += 1
            if result is not None:
                samples.append(result)
            else:
                logger.debug(
                    "Attempt %d/%d failed, have %d/%d samples.",
                    attempts, max_attempts, len(samples), n,
                )

        if len(samples) < n:
            logger.warning(
                "Only collected %d/%d samples after %d attempts.",
                len(samples), n, attempts,
            )

        return samples

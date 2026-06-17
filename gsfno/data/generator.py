"""FreeGS free-boundary FORWARD solver wrapper for GS training data.

Each sample varies profile parameters (paxis, Ip, fvac, alpha_m, alpha_n),
runs a free-boundary Picard solve with fixed x-point constraints, and records:

  psi_total      : total poloidal flux on the grid (Wb) = plasma + coil flux
  psi_vac        : coil-only (vacuum) flux on the grid (Wb) -- computed from
                   the Green's-function field without the plasma, so it leaks
                   no answer to the network
  pprime_curve   : p'(psi_N) sampled at n_psi points in [0, 1]
  ffprime_curve  : ff'(psi_N) sampled at n_psi points in [0, 1]
  params         : dict with paxis, Ip, fvac, alpha_m, alpha_n and the
                   resulting coil currents (recorded after the constrained solve)

Validation gate
---------------
``validate(sample)`` checks that the stored psi_total satisfies the
Grad-Shafranov equation by computing the physical residual:

    r = || Delta*(plasma_psi) + mu0 * R * Jtor ||
        ----------------------------------------
             || Delta*(plasma_psi) ||

where plasma_psi = psi_total - psi_vac.  For a converged FreeGS solve this
is typically 3-5 %; the gate rejects samples with r > tol (default 0.10).
A deliberately corrupted psi_total yields r close to 1.0.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch

from gsfno.physics import star_laplacian

logger = logging.getLogger(__name__)

try:
    import freegs  # type: ignore
    FREEGS_AVAILABLE = True
except ImportError:
    FREEGS_AVAILABLE = False

_MU0 = 1.2566370614e-6  # vacuum permeability [H/m]
_EPS = 1e-12

# Fixed x-point and isoflux constraints for TestTokamak.
# These match the FreeGS test suite and allow the free-boundary Picard
# iteration to converge reliably across a wide range of profile parameters.
_XPOINTS = [(1.1, -0.6), (1.1, 0.8)]
_ISOFLUX = [(1.1, -0.6, 1.1, 0.6)]


class PlasmaConfigGenerator:
    """Generate free-boundary Grad-Shafranov equilibria via FreeGS.

    Parameters
    ----------
    machine_name:
        Name of a ``freegs.machine`` factory function (default ``"TestTokamak"``).
    NR, NZ:
        Grid resolution.
    n_psi:
        Number of normalised-psi sample points for profile curves.
    seed:
        Optional RNG seed for reproducibility.
    """

    def __init__(
        self,
        machine_name: str = "TestTokamak",
        NR: int = 65,
        NZ: int = 65,
        n_psi: int = 64,
        seed: Optional[int] = None,
    ) -> None:
        self.machine_name = machine_name
        self.NR = NR
        self.NZ = NZ
        self.n_psi = n_psi
        self.rng = np.random.default_rng(seed)

        # Standard domain for TestTokamak (matches FreeGS examples)
        self.Rmin, self.Rmax = 0.1, 2.0
        self.Zmin, self.Zmax = -1.0, 1.0

        if FREEGS_AVAILABLE:
            _tok = getattr(freegs.machine, machine_name)()
            self._coil_names: list[str] = [c[0] for c in _tok.coils]
        else:
            self._coil_names = []

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def grid(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (R_vals, Z_vals) 1-D arrays in metres."""
        R = np.linspace(self.Rmin, self.Rmax, self.NR, dtype=np.float64)
        Z = np.linspace(self.Zmin, self.Zmax, self.NZ, dtype=np.float64)
        return R, Z

    def psiN_grid(self) -> np.ndarray:
        """Return (n_psi,) normalised-psi sample points in [0, 1]."""
        return np.linspace(0.0, 1.0, self.n_psi, dtype=np.float64)

    # ------------------------------------------------------------------
    # Parameter sampling
    # ------------------------------------------------------------------

    def _sample_params(self) -> dict:
        """Sample one random plasma configuration.

        Returns a dict with profile keys only (coil currents are set by the
        constrained solver and recorded after the solve).

        Ranges chosen to keep Picard iteration convergent:
          paxis  : axis pressure [Pa]         1e3 – 5e4
          Ip     : plasma current [A]         2e5 – 1.2e6
          fvac   : vacuum f = R*Btor [T·m]   1.0 – 3.0
          alpha_m: ConstrainPaxisIp exponent  1.0 – 2.0
          alpha_n: ConstrainPaxisIp exponent  1.0 – 2.5
        """
        rng = self.rng
        return {
            "paxis": float(rng.uniform(1e3, 5e4)),
            "Ip": float(rng.uniform(2e5, 1.2e6)),
            "fvac": float(rng.uniform(1.0, 3.0)),
            "alpha_m": float(rng.uniform(1.0, 2.0)),
            "alpha_n": float(rng.uniform(1.0, 2.5)),
        }

    # ------------------------------------------------------------------
    # Single-sample generation
    # ------------------------------------------------------------------

    def generate_one(self) -> Optional[dict]:
        """Run one free-boundary forward solve and return a sample dict.

        Returns ``None`` if the FreeGS Picard iteration does not converge
        or if the result is non-finite.

        Raises
        ------
        ImportError
            If FreeGS is not installed.
        """
        if not FREEGS_AVAILABLE:
            raise ImportError(
                "FreeGS is not installed. Install it with: pip install freegs"
            )

        params = self._sample_params()

        try:
            tok = getattr(freegs.machine, self.machine_name)()

            eq = freegs.Equilibrium(
                tokamak=tok,
                Rmin=self.Rmin, Rmax=self.Rmax,
                Zmin=self.Zmin, Zmax=self.Zmax,
                nx=self.NR, ny=self.NZ,
                boundary=freegs.boundary.freeBoundaryHagenow,
            )

            profiles = freegs.jtor.ConstrainPaxisIp(
                eq,
                params["paxis"],
                params["Ip"],
                params["fvac"],
                alpha_m=params["alpha_m"],
                alpha_n=params["alpha_n"],
            )

            constrain = freegs.control.constrain(
                xpoints=_XPOINTS,
                isoflux=_ISOFLUX,
            )

            freegs.solve(
                eq, profiles, constrain,
                maxits=60, atol=1e-3, rtol=1e-2,
            )

            # ---- Extract fields ------------------------------------------------

            psi_total = np.asarray(eq.psi(), dtype=np.float64)
            if psi_total.shape != (self.NR, self.NZ):
                psi_total = psi_total.T

            R, Z = self.grid()
            RR, ZZ = np.meshgrid(R, Z, indexing="ij")

            # Vacuum (coil-only) flux via the machine's Green's function
            psi_vac = np.asarray(tok.psi(RR, ZZ), dtype=np.float64)
            if psi_vac.shape != (self.NR, self.NZ):
                psi_vac = psi_vac.T

            # Record coil currents resulting from the constrained solve
            coil_currents = {
                f"coil_{name}": float(tok[name].current)
                for name in self._coil_names
            }

            # Profile curves vs psi_N
            psiN = self.psiN_grid()
            pprime_curve = np.asarray(profiles.pprime(psiN), dtype=np.float64)
            ffprime_curve = np.asarray(profiles.ffprime(psiN), dtype=np.float64)

            # ---- Sanity checks -------------------------------------------------
            if not (np.all(np.isfinite(psi_total)) and np.all(np.isfinite(psi_vac))):
                return None
            if not (np.all(np.isfinite(pprime_curve)) and np.all(np.isfinite(ffprime_curve))):
                return None

            R0 = float(0.5 * (self.Rmin + self.Rmax))

            full_params = {**params, **coil_currents}

            sample = {
                "psi_total": psi_total.astype(np.float32),
                "psi_vac": psi_vac.astype(np.float32),
                "pprime_curve": pprime_curve.astype(np.float32),
                "ffprime_curve": ffprime_curve.astype(np.float32),
                "params": full_params,
                "R0": R0,
                "Ip": float(params["Ip"]),
                # Stash solve artefacts needed by validate()
                "_eq": eq,
                "_profiles": profiles,
                "_RR": RR,
                "_ZZ": ZZ,
            }
            return sample

        except Exception as exc:
            logger.debug("FreeGS solve failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Validation gate
    # ------------------------------------------------------------------

    def validate(self, sample: dict, tol: float = 0.10) -> bool:
        """Residual gate: reject non-equilibria.

        The gate computes the relative Grad-Shafranov residual using the
        physical form

            Delta*(plasma_psi) + mu0 * R * Jtor  ~=  0

        where plasma_psi = psi_total - psi_vac and Jtor is evaluated from
        the FreeGS profiles object stored in the sample at generation time.

        For a converged FreeGS solve the relative residual is typically 3-5 %.
        The default tolerance is 10 %, giving comfortable headroom.
        A corrupted psi_total yields a relative residual close to 1.0.

        Parameters
        ----------
        sample:
            Dict produced by ``generate_one()``.
        tol:
            Maximum allowed relative residual (default 0.10).

        Returns
        -------
        bool: True if sample passes, False if it should be rejected.
        """
        eq = sample.get("_eq")
        profiles = sample.get("_profiles")
        RR = sample.get("_RR")

        psi_total = sample["psi_total"].astype(np.float64)
        psi_vac = sample["psi_vac"].astype(np.float64)
        plasma_psi = psi_total - psi_vac

        R, Z = self.grid()
        dR = float(R[1] - R[0])
        dZ = float(Z[1] - Z[0])

        # Compute Jtor using the actual FreeGS profiles
        if eq is not None and profiles is not None and RR is not None:
            try:
                ZZ = sample.get("_ZZ")
                psi_bndry = eq.psi_bndry
                Jtor = profiles.Jtor(RR, ZZ, psi_total, psi_bndry)
                mu0_R_Jtor = _MU0 * RR * Jtor
            except Exception:
                # If Jtor can't be evaluated (e.g. corrupted psi lost O-point),
                # fall back to zero and let the Laplacian check handle it
                mu0_R_Jtor = np.zeros_like(plasma_psi)
        else:
            # No FreeGS objects: fall back to Laplacian-only finiteness check
            mu0_R_Jtor = np.zeros_like(plasma_psi)

        # Compute star-Laplacian of plasma_psi via PyTorch FD stencil
        psi_t = torch.from_numpy(plasma_psi).view(1, 1, self.NR, self.NZ)
        R_t = torch.from_numpy(R)
        lap = star_laplacian(psi_t, R_t, dR, dZ)

        mu0_RJ_t = torch.from_numpy(mu0_R_Jtor).view(1, 1, self.NR, self.NZ)

        # Relative residual over interior (skip 2-cell border)
        resid = lap + mu0_RJ_t
        lap_norm = lap[:, :, 2:-2, 2:-2].abs().mean().item()
        resid_norm = resid[:, :, 2:-2, 2:-2].abs().mean().item()
        rel = resid_norm / (lap_norm + _EPS)

        if not np.isfinite(rel):
            return False

        return rel < tol

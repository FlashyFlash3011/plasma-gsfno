"""Global dimensionless scaling for Grad-Shafranov fields.

A single (R0, psi_ref) pair is shared across the WHOLE dataset, so relative
magnitudes between samples are preserved. This deliberately replaces the old
per-sample min-max normalization, which deleted the magnitude information that
determines the equilibrium.
"""

from __future__ import annotations

from dataclasses import dataclass

_MU0 = 1.2566370614e-6  # vacuum permeability [H/m]


def reference_psi(R0: float, Ip: float, mu0: float = _MU0) -> float:
    """A fixed flux scale with units of Weber: mu0 * Ip * R0."""
    return mu0 * Ip * R0


@dataclass(frozen=True)
class Normalization:
    R0: float       # length scale [m]
    psi_ref: float  # flux scale [Wb]

    def length_to_dimensionless(self, x_m):
        return x_m / self.R0

    def length_to_physical(self, x_hat):
        return x_hat * self.R0

    def flux_to_dimensionless(self, psi_wb):
        return psi_wb / self.psi_ref

    def flux_to_physical(self, psi_hat):
        return psi_hat * self.psi_ref

    def to_attrs(self) -> dict[str, float]:
        return {"R0": float(self.R0), "psi_ref": float(self.psi_ref)}

    @classmethod
    def from_attrs(cls, d: dict) -> "Normalization":
        return cls(R0=float(d["R0"]), psi_ref=float(d["psi_ref"]))

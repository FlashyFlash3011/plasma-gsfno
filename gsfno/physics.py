"""Physics utilities for the Grad-Shafranov equation.

Five standalone functions — pure PyTorch, fully differentiable.

GS equation (cylindrical coordinates, axisymmetric):
    Δ*ψ = -μ₀ R p'(ψ) - ff'(ψ)

where the star-Laplacian is:
    Δ*ψ = R ∂/∂R(1/R ∂ψ/∂R) + ∂²ψ/∂Z²
"""

import torch
from torch import Tensor

_MU0 = 1.2566370614e-6  # vacuum permeability [H/m]


def star_laplacian(psi: Tensor, R: Tensor, dR: float, dZ: float) -> Tensor:
    """Compute the GS star-Laplacian Δ*ψ on interior grid points.

    Δ*ψ = R ∂/∂R(1/R ∂ψ/∂R) + ∂²ψ/∂Z²

    Expanding the first term:
        R ∂/∂R(1/R ∂ψ/∂R) = ∂²ψ/∂R² - (1/R) ∂ψ/∂R

    Uses second-order central finite differences. Boundary values are set to zero.

    Args:
        psi:  (B, 1, NR, NZ) flux field tensor.
        R:    (NR,) radial coordinate values.
        dR:   Grid spacing in R direction.
        dZ:   Grid spacing in Z direction.

    Returns:
        (B, 1, NR, NZ) tensor — Δ*ψ, zero on boundaries.
    """
    # psi: (B, 1, NR, NZ)
    # R:   (NR,)

    # Reshape R for broadcasting: (1, 1, NR, 1)
    R4 = R.view(1, 1, -1, 1)

    # Second-order central differences for interior points
    # ∂²ψ/∂R²  via (psi[i+1] - 2*psi[i] + psi[i-1]) / dR²
    d2psi_dR2 = (psi[:, :, 2:, :] - 2.0 * psi[:, :, 1:-1, :] + psi[:, :, :-2, :]) / (dR ** 2)

    # ∂ψ/∂R via (psi[i+1] - psi[i-1]) / (2*dR)
    dpsi_dR = (psi[:, :, 2:, :] - psi[:, :, :-2, :]) / (2.0 * dR)

    # ∂²ψ/∂Z²  via (psi[j+1] - 2*psi[j] + psi[j-1]) / dZ²
    d2psi_dZ2 = (psi[:, :, :, 2:] - 2.0 * psi[:, :, :, 1:-1] + psi[:, :, :, :-2]) / (dZ ** 2)

    # Interior R values: indices 1..NR-2  -> shape (NR-2,)
    R_int = R4[:, :, 1:-1, :]  # (1, 1, NR-2, 1)

    # Star-Laplacian on R-interior, Z-interior patch
    # Shape of d2psi_dR2:  (B, 1, NR-2, NZ)
    # Shape of d2psi_dZ2:  (B, 1, NR,   NZ-2)
    # We need the intersection: (B, 1, NR-2, NZ-2)
    lap_star_int = (
        d2psi_dR2[:, :, :, 1:-1]          # ∂²ψ/∂R² on interior Z
        - (1.0 / R_int) * dpsi_dR[:, :, :, 1:-1]  # -(1/R) ∂ψ/∂R on interior Z
        + d2psi_dZ2[:, :, 1:-1, :]        # ∂²ψ/∂Z² on interior R
    )

    # Build output, zero everywhere (Dirichlet boundaries)
    result = torch.zeros_like(psi)
    result[:, :, 1:-1, 1:-1] = lap_star_int

    return result


def gs_residual(
    psi: Tensor,
    p_prime: Tensor,
    ff_prime: Tensor,
    R_grid: Tensor,
    dR: float,
    dZ: float,
    mu0: float = _MU0,
) -> Tensor:
    """Compute the Grad-Shafranov equation residual.

    residual = Δ*ψ + μ₀ R p'(ψ) + ff'(ψ)

    In equilibrium this is zero everywhere inside the plasma.

    Args:
        psi:      (B, 1, NR, NZ) predicted flux field.
        p_prime:  (B, 1, NR, NZ) pressure profile channel (grid-evaluated).
        ff_prime: (B, 1, NR, NZ) current profile channel (grid-evaluated).
        R_grid:   (B, 1, NR, NZ) or broadcastable — R coordinate on the grid.
                  Must be uniform across batch and Z (axisymmetric: R depends
                  only on the R axis). Typically shape (1, 1, NR, 1).
        dR:       Grid spacing in R.
        dZ:       Grid spacing in Z.
        mu0:      Vacuum permeability (default 4π×10⁻⁷ H/m).

    Returns:
        (B, 1, NR, NZ) residual tensor.
    """
    # Extract 1-D R vector for star_laplacian from whatever shape R_grid has
    # R_grid is (B, 1, NR, NZ) or (1, 1, NR, 1) — grab the NR axis
    # R is axisymmetric: same for every batch element and every Z position.
    # Take a 1-D slice from batch 0, Z column 0.
    R_1d = R_grid.reshape(-1, R_grid.shape[-2], R_grid.shape[-1])[0, :, 0]  # (NR,)

    lap = star_laplacian(psi, R_1d, dR, dZ)
    return lap + mu0 * R_grid * p_prime + ff_prime


def gs_residual_loss(
    psi: Tensor,
    p_prime: Tensor,
    ff_prime: Tensor,
    R_grid: Tensor,
    dR: float,
    dZ: float,
) -> Tensor:
    """Mean-squared Grad-Shafranov residual — scalar physics loss term.

    Args:
        psi:      (B, 1, NR, NZ) predicted flux field.
        p_prime:  (B, 1, NR, NZ) pressure profile channel.
        ff_prime: (B, 1, NR, NZ) current profile channel.
        R_grid:   (B, 1, NR, NZ) or broadcastable — R coordinate on the grid.
        dR:       Grid spacing in R.
        dZ:       Grid spacing in Z.

    Returns:
        Scalar tensor — mean squared GS residual.
    """
    return gs_residual(psi, p_prime, ff_prime, R_grid, dR, dZ).pow(2).mean()


def compute_magnetic_field(
    psi: Tensor,
    R_grid: Tensor,
    dR: float,
    dZ: float,
) -> tuple[Tensor, Tensor]:
    """Compute poloidal magnetic field components from the flux function ψ.

    In axisymmetric cylindrical coordinates (R, Z):
        B_R = -(1/R) ∂ψ/∂Z
        B_Z =  (1/R) ∂ψ/∂R

    Central differences are used for interior points; boundary values are zero.

    Args:
        psi:     (B, 1, NR, NZ) flux field.
        R_grid:  (B, 1, NR, NZ) or broadcastable — R coordinate on grid.
        dR:      Grid spacing in R.
        dZ:      Grid spacing in Z.

    Returns:
        (B_R, B_Z) — each (B, 1, NR, NZ), zero on boundaries.
    """
    # ∂ψ/∂Z — central difference along Z axis (axis 3)
    dpsi_dZ = torch.zeros_like(psi)
    dpsi_dZ[:, :, :, 1:-1] = (psi[:, :, :, 2:] - psi[:, :, :, :-2]) / (2.0 * dZ)

    # ∂ψ/∂R — central difference along R axis (axis 2)
    dpsi_dR = torch.zeros_like(psi)
    dpsi_dR[:, :, 1:-1, :] = (psi[:, :, 2:, :] - psi[:, :, :-2, :]) / (2.0 * dR)

    B_R = -(1.0 / R_grid) * dpsi_dZ
    B_Z = (1.0 / R_grid) * dpsi_dR

    return B_R, B_Z


def divergence_free_error(
    psi: Tensor,
    R_grid: Tensor,
    dR: float,
    dZ: float,
) -> Tensor:
    """Compute the mean absolute divergence of B derived from ψ.

    In cylindrical coordinates:
        ∇·B = (1/R) ∂(R B_R)/∂R + ∂B_Z/∂Z

    For B = (-∂ψ/∂Z / R,  ∂ψ/∂R / R) this is analytically zero; numerical
    FD errors introduce a small residual that should be ~machine epsilon for
    a smooth ψ on a fine grid.

    Args:
        psi:     (B, 1, NR, NZ) flux field.
        R_grid:  (B, 1, NR, NZ) or broadcastable — R coordinate on grid.
        dR:      Grid spacing in R.
        dZ:      Grid spacing in Z.

    Returns:
        Scalar tensor — mean |∇·B| over the interior.
    """
    B_R, B_Z = compute_magnetic_field(psi, R_grid, dR, dZ)

    # R * B_R to form the flux for the divergence term
    RBR = R_grid * B_R  # (B, 1, NR, NZ)

    # ∂(R B_R)/∂R  — central diff along R, interior only
    d_RBR_dR = torch.zeros_like(psi)
    d_RBR_dR[:, :, 1:-1, :] = (RBR[:, :, 2:, :] - RBR[:, :, :-2, :]) / (2.0 * dR)

    # ∂B_Z/∂Z — central diff along Z, interior only
    d_BZ_dZ = torch.zeros_like(psi)
    d_BZ_dZ[:, :, :, 1:-1] = (B_Z[:, :, :, 2:] - B_Z[:, :, :, :-2]) / (2.0 * dZ)

    # R_grid at interior points for (1/R) prefactor
    div_B = (1.0 / R_grid) * d_RBR_dR + d_BZ_dZ

    # Mean absolute divergence on interior (exclude 2-cell border where FD is zero)
    interior = div_B[:, :, 2:-2, 2:-2]
    return interior.abs().mean()

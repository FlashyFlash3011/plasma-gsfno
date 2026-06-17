"""Tests for gsfno/physics.py — five physics utility functions."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gsfno.physics import (
    compute_magnetic_field,
    divergence_free_error,
    gs_residual,
    gs_residual_dimensionless,
    gs_residual_loss,
    star_laplacian,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_grid(NR: int = 65, NZ: int = 65, R0: float = 0.5, R1: float = 1.5,
               Z0: float = -0.5, Z1: float = 0.5):
    """Return R (1-D), Z (1-D), dR, dZ, and R_grid (1, 1, NR, NZ)."""
    R = torch.linspace(R0, R1, NR)
    Z = torch.linspace(Z0, Z1, NZ)
    dR = float((R1 - R0) / (NR - 1))
    dZ = float((Z1 - Z0) / (NZ - 1))
    RR, ZZ = torch.meshgrid(R, Z, indexing="ij")  # (NR, NZ)
    R_grid = RR.unsqueeze(0).unsqueeze(0)          # (1, 1, NR, NZ)
    return R, Z, dR, dZ, R_grid, RR, ZZ


# ---------------------------------------------------------------------------
# 1. test_star_laplacian_shape
# ---------------------------------------------------------------------------

def test_star_laplacian_shape():
    """Output shape equals input shape and boundary values are zero."""
    B, NR, NZ = 2, 33, 33
    R = torch.linspace(0.5, 1.5, NR)
    dR = float(1.0 / (NR - 1))
    dZ = float(1.0 / (NZ - 1))
    psi = torch.randn(B, 1, NR, NZ)

    out = star_laplacian(psi, R, dR, dZ)

    assert out.shape == (B, 1, NR, NZ), f"Expected {(B,1,NR,NZ)}, got {out.shape}"

    # Boundaries must be zero (Dirichlet)
    assert out[:, :, 0, :].abs().max() == 0.0,  "Left  R boundary should be zero"
    assert out[:, :, -1, :].abs().max() == 0.0, "Right R boundary should be zero"
    assert out[:, :, :, 0].abs().max() == 0.0,  "Bottom Z boundary should be zero"
    assert out[:, :, :, -1].abs().max() == 0.0, "Top    Z boundary should be zero"


# ---------------------------------------------------------------------------
# 2. test_solovev_analytical  (ψ = R²Z  →  Δ*ψ = 0)
# ---------------------------------------------------------------------------

def test_solovev_analytical():
    """For ψ = R²Z, p'=0, ff'=0, the GS residual Δ*ψ should be ~0.

    Analytical check:
        ∂²ψ/∂R² = 2Z
        (1/R)∂ψ/∂R = (1/R)(2RZ) = 2Z
        ∂²ψ/∂Z² = 0
        Δ*ψ = 2Z - 2Z + 0 = 0
    FD truncation error for this quadratic field is machine-epsilon level.
    """
    R, Z, dR, dZ, R_grid, RR, ZZ = _make_grid(NR=65, NZ=65)

    psi = (RR ** 2 * ZZ).unsqueeze(0).unsqueeze(0)  # (1, 1, 65, 65)

    p_prime  = torch.zeros_like(psi)
    ff_prime = torch.zeros_like(psi)

    residual = gs_residual(psi, p_prime, ff_prime, R_grid, dR, dZ)

    # Interior slice only (boundaries are clamped to zero by star_laplacian)
    interior_mean = residual[:, :, 1:-1, 1:-1].abs().mean().item()
    assert interior_mean < 1e-4, (
        f"GS residual mean on interior should be ~0 for ψ=R²Z, got {interior_mean:.3e}"
    )


# ---------------------------------------------------------------------------
# 3. test_divergence_free_error
# ---------------------------------------------------------------------------

def test_divergence_free_error():
    """∇·B derived from ψ should be near-zero (analytically exact).

    Use a smooth sinusoidal ψ on a 65×65 grid; FD truncation error ~ O(h²).
    """
    R, Z, dR, dZ, R_grid, RR, ZZ = _make_grid(NR=65, NZ=65)

    # Smooth, non-trivial field
    psi = (torch.sin(2 * torch.pi * (RR - 0.5)) * torch.cos(2 * torch.pi * ZZ)
           ).unsqueeze(0).unsqueeze(0)

    err = divergence_free_error(psi, R_grid, dR, dZ).item()
    assert err < 1e-3, (
        f"Divergence-free error should be < 1e-3 for smooth ψ, got {err:.3e}"
    )


# ---------------------------------------------------------------------------
# 4. test_magnetic_field_shapes
# ---------------------------------------------------------------------------

def test_magnetic_field_shapes():
    """compute_magnetic_field returns two tensors with the same shape as psi."""
    B_batch, NR, NZ = 3, 33, 33
    _, _, dR, dZ, R_grid, RR, ZZ = _make_grid(NR=NR, NZ=NZ)

    psi = torch.randn(B_batch, 1, NR, NZ)
    R_grid_exp = R_grid.expand(B_batch, -1, -1, -1)

    B_R, B_Z = compute_magnetic_field(psi, R_grid_exp, dR, dZ)

    assert B_R.shape == psi.shape, f"B_R shape {B_R.shape} != psi shape {psi.shape}"
    assert B_Z.shape == psi.shape, f"B_Z shape {B_Z.shape} != psi shape {psi.shape}"


# ---------------------------------------------------------------------------
# 5. test_magnetic_field_values
# ---------------------------------------------------------------------------

def test_magnetic_field_values():
    """Verify B-field values are numerically correct using ψ = R * Z.

    Analytical results:
        ∂ψ/∂Z = R  →  B_R = -(1/R) * R = -1  (uniform on interior)
        ∂ψ/∂R = Z  →  B_Z =  (1/R) * Z = Z/R (varies)
    """
    NR, NZ = 33, 33
    R_vals = torch.linspace(0.5, 1.5, NR)
    Z_vals = torch.linspace(-0.5, 0.5, NZ)
    dR = (R_vals[-1] - R_vals[0]).item() / (NR - 1)
    dZ = (Z_vals[-1] - Z_vals[0]).item() / (NZ - 1)
    R_grid, Z_grid = torch.meshgrid(R_vals, Z_vals, indexing="ij")
    R_grid = R_grid.unsqueeze(0).unsqueeze(0)   # (1, 1, NR, NZ)
    Z_grid = Z_grid.unsqueeze(0).unsqueeze(0)   # (1, 1, NR, NZ)

    psi = R_grid * Z_grid   # ψ = R * Z

    B_R, B_Z = compute_magnetic_field(psi, R_grid, dR, dZ)

    # Interior slice (skip 1-cell boundary where FD is zero)
    B_R_int = B_R[:, :, 1:-1, 1:-1]
    B_Z_int = B_Z[:, :, 1:-1, 1:-1]

    # B_R should equal -1.0 everywhere on interior (∂ψ/∂Z = R, so B_R = -(1/R)*R = -1)
    assert torch.allclose(B_R_int, torch.full_like(B_R_int, -1.0), atol=1e-5), \
        f"B_R max error: {(B_R_int + 1.0).abs().max()}"

    # B_Z = (1/R)*∂ψ/∂R = (1/R)*Z = Z/R on interior
    expected_BZ = Z_grid[:, :, 1:-1, 1:-1] / R_grid[:, :, 1:-1, 1:-1]
    assert torch.allclose(B_Z_int, expected_BZ, atol=1e-5), \
        f"B_Z max error: {(B_Z_int - expected_BZ).abs().max()}"


# ---------------------------------------------------------------------------
# 6. test_gs_residual_loss_gradient_flow
# ---------------------------------------------------------------------------

def test_gs_residual_loss_gradient_flow():
    """gs_residual_loss (dimensionless) must be differentiable w.r.t. psi_hat."""
    R, Z, dR, dZ, R_grid, RR, ZZ = _make_grid(NR=33, NZ=33)

    psi_hat = torch.randn(2, 1, 33, 33, requires_grad=True)
    pprime_hat  = torch.zeros(2, 1, 33, 33)
    ffprime_hat = torch.zeros(2, 1, 33, 33)
    R_hat = R_grid  # shape (1, 1, NR, NZ) — broadcastable

    loss = gs_residual_loss(psi_hat, pprime_hat, ffprime_hat, R_hat, dR, dZ)
    loss.backward()

    assert psi_hat.grad is not None, "psi_hat.grad should not be None after backward()"
    assert not torch.isnan(psi_hat.grad).any(), "psi_hat.grad contains NaNs"


# ---------------------------------------------------------------------------
# 7. test_residual_vanishes_on_true_solution  (dimensionless manufactured)
# ---------------------------------------------------------------------------

def _manufactured_solovev(NR=129, NZ=129):
    """Solov'ev: psi = 0.5*(R^2 * Z^2)/... use the standard exact form
    psi(R,Z) for which Delta* psi = a*R^2 + b is exact.

    Use psi = (1/2) * R^2 * Z  -> Delta*psi:
      d2/dR2 (psi) = Z ; (1/R) dpsi/dR = (1/R)*(R*Z) = Z ; d2/dZ2 = 0
      Delta* = d2/dR2 - (1/R)dpsi/dR + d2/dZ2 = Z - Z + 0 = 0.
    So with psi = 0.5 R^2 Z, Delta*psi = 0 exactly; pick pprime=ffprime=0
    => residual must be ~0 (FD truncation only).
    """
    R = np.linspace(0.6, 1.4, NR)
    Z = np.linspace(-0.4, 0.4, NZ)
    RR, ZZ = np.meshgrid(R, Z, indexing="ij")
    psi = 0.5 * RR**2 * ZZ
    dR = float(R[1] - R[0])
    dZ = float(Z[1] - Z[0])
    return psi, R, dR, dZ


def test_residual_vanishes_on_true_solution():
    psi, R, dR, dZ = _manufactured_solovev()
    psi_t = torch.from_numpy(psi).view(1, 1, *psi.shape).double()
    R_t = torch.from_numpy(R).view(1, 1, -1, 1).double()
    zeros = torch.zeros_like(psi_t)
    res = gs_residual_dimensionless(psi_t, zeros, zeros, R_t, dR, dZ)
    interior = res[:, :, 2:-2, 2:-2]
    # FD truncation only — must be tiny relative to field scale.
    assert interior.abs().max().item() < 1e-2


def test_residual_loss_is_scalar_and_nonneg():
    psi, R, dR, dZ = _manufactured_solovev(33, 33)
    psi_t = torch.from_numpy(psi).view(1, 1, *psi.shape).double()
    R_t = torch.from_numpy(R).view(1, 1, -1, 1).double()
    zeros = torch.zeros_like(psi_t)
    loss = gs_residual_loss(psi_t, zeros, zeros, R_t, dR, dZ)
    assert loss.ndim == 0 and loss.item() >= 0.0

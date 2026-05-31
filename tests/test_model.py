"""Tests for GradShafranovFNO model."""

import sys
from pathlib import Path

import torch
import pytest

# Ensure project root is on sys.path so gsfno package is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gsfno.model import GradShafranovFNO


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

B, NR, NZ = 2, 65, 65


@pytest.fixture(scope="module")
def model():
    return GradShafranovFNO()


@pytest.fixture(scope="module")
def grid_inputs():
    x = torch.randn(B, 5, NR, NZ)
    psi_true = torch.randn(B, 1, NR, NZ)
    R_vals = torch.linspace(0.5, 2.5, NR)
    R_grid = R_vals.view(1, 1, NR, 1).expand(B, 1, NR, NZ).contiguous()
    dR = (2.5 - 0.5) / (NR - 1)
    dZ = 3.0 / (NZ - 1)
    return x, psi_true, R_grid, dR, dZ


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_forward_shape(model, grid_inputs):
    x, _, _, _, _ = grid_inputs
    out = model(x)
    assert out.shape == (B, 1, NR, NZ), f"Expected (2,1,65,65), got {out.shape}"


def test_compute_loss_keys(model, grid_inputs):
    x, psi_true, R_grid, dR, dZ = grid_inputs
    psi_pred = model(x)
    _, metrics = model.compute_loss(psi_pred, psi_true, x, R_grid, dR, dZ)
    expected_keys = {"mse", "rel_l2", "phys_loss", "lambda_eff", "total"}
    assert set(metrics.keys()) == expected_keys, f"Missing keys: {expected_keys - set(metrics.keys())}"


def test_physics_warmup(grid_inputs):
    x, psi_true, R_grid, dR, dZ = grid_inputs
    model = GradShafranovFNO(lambda_phys=0.1, phys_warmup_epochs=10)

    # Before warmup
    model.set_epoch(0)
    psi_pred = model(x)
    _, metrics = model.compute_loss(psi_pred, psi_true, x, R_grid, dR, dZ)
    assert metrics["lambda_eff"] == 0.0, f"Expected lambda_eff=0 at epoch 0, got {metrics['lambda_eff']}"

    # At warmup boundary
    model.set_epoch(10)
    psi_pred = model(x)
    _, metrics = model.compute_loss(psi_pred, psi_true, x, R_grid, dR, dZ)
    assert metrics["lambda_eff"] == model.lambda_phys, (
        f"Expected lambda_eff={model.lambda_phys} at epoch 10, got {metrics['lambda_eff']}"
    )


def test_gradient_flow(grid_inputs):
    x, psi_true, R_grid, dR, dZ = grid_inputs
    model = GradShafranovFNO()
    model.train()

    psi_pred = model(x)
    total, _ = model.compute_loss(psi_pred, psi_true, x, R_grid, dR, dZ)
    total.backward()

    # Check at least one parameter in fno has a valid gradient
    has_grad = False
    for p in model.fno.parameters():
        if p.grad is not None and not torch.isnan(p.grad).any():
            has_grad = True
            break
    assert has_grad, "No parameter in model.fno has a non-None, non-NaN gradient"


def test_num_parameters(model):
    n = model.num_parameters()
    assert isinstance(n, int), f"num_parameters() should return int, got {type(n)}"
    assert n > 1_000_000, f"Expected > 1M parameters, got {n:,}"

"""Tests for GradShafranovFNO model."""

import sys
from pathlib import Path

import pytest
import torch

# Ensure project root is on sys.path so gsfno package is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gsfno.model import GradShafranovFNO

B, NR, NZ = 2, 65, 65


def _batch():
    x = torch.randn(B, 5, NR, NZ)
    y = torch.randn(B, 1, NR, NZ)
    R_hat = torch.linspace(0.1, 1.2, NR).view(1, 1, NR, 1)
    return x, y, R_hat, 1.1 / (NR - 1), 1.2 / (NZ - 1)


def test_forward_shape():
    m = GradShafranovFNO()
    x, *_ = _batch()
    assert m(x).shape == (B, 1, NR, NZ)


def test_loss_keys_and_warmup():
    m = GradShafranovFNO(lambda_phys=0.1, phys_warmup_epochs=10)
    x, y, R_hat, dR, dZ = _batch()
    m.set_epoch(0)
    _, mt = m.compute_loss(m(x), y, x, R_hat, dR, dZ)
    assert set(mt) == {"mse", "rel_l2", "phys_loss", "lambda_eff", "total"}
    assert mt["lambda_eff"] == 0.0
    m.set_epoch(10)
    _, mt2 = m.compute_loss(m(x), y, x, R_hat, dR, dZ)
    assert mt2["lambda_eff"] == 0.1


def test_inputs_have_no_mask_channel():
    # The model's channel contract is [psi_vac, R_norm, Z_norm, pprime_lift, ffprime_lift]
    # — exactly 5 channels, none of which is a binary mask of the target psi_total.
    # Verify the no-leak / no-mask contract: the model must accept (B,5,NR,NZ) inputs
    # and must reject (B,6,NR,NZ) inputs (a 6th mask channel would be a contract violation).
    m = GradShafranovFNO()

    # 5-channel input (correct contract) — forward must succeed
    x5 = torch.randn(B, 5, NR, NZ)
    out = m(x5)
    assert out.shape == (B, 1, NR, NZ), "Model must accept exactly 5 input channels"

    # 6-channel input (mask channel leak) — forward must fail with a shape/runtime error
    x6 = torch.randn(B, 6, NR, NZ)
    with pytest.raises(Exception):
        m(x6)


def test_gradient_flow():
    m = GradShafranovFNO()
    m.train()
    x, y, R_hat, dR, dZ = _batch()
    total, _ = m.compute_loss(m(x), y, x, R_hat, dR, dZ)
    total.backward()
    assert any(p.grad is not None and not torch.isnan(p.grad).any()
               for p in m.fno.parameters())

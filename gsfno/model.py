"""GradShafranovFNO — FNO surrogate for the Grad-Shafranov equilibrium equation."""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

_SOLARIS_ROOT = Path(__file__).resolve().parents[2] / "Solaris"
if str(_SOLARIS_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOLARIS_ROOT))

from solaris.models import FNO  # noqa: E402
from solaris.metrics import relative_l2_error  # noqa: E402

from gsfno.physics import gs_residual_loss  # noqa: E402


class GradShafranovFNO(nn.Module):
    """FNO surrogate for the Grad-Shafranov equilibrium equation.

    Predicts ψ(R,Z) given 5-channel input fields. The physics-informed loss
    includes a GS equation residual term that ramps in after `phys_warmup_epochs`.

    Architecture:
        Input:  (B, 5, NR, NZ)
        FNO(in_channels=5, out_channels=1, hidden_channels=64, n_layers=4, modes=16, dim=2)
        Output: ψ(R,Z)  (B, 1, NR, NZ)
    """

    def __init__(
        self,
        in_channels: int = 5,
        out_channels: int = 1,
        hidden_channels: int = 64,
        n_layers: int = 4,
        modes: int = 16,
        lambda_phys: float = 0.1,
        phys_warmup_epochs: int = 10,
    ):
        super().__init__()
        self.fno = FNO(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            n_layers=n_layers,
            modes=modes,
            dim=2,
        )
        self.lambda_phys = lambda_phys
        self.phys_warmup_epochs = phys_warmup_epochs
        self._current_epoch: int = 0

    def forward(self, x: Tensor) -> Tensor:
        """Predict ψ(R,Z).

        Args:
            x: (B, 5, NR, NZ) — input channels

        Returns:
            (B, 1, NR, NZ) — predicted ψ
        """
        return self.fno(x)

    def set_epoch(self, epoch: int) -> None:
        """Update current epoch for physics loss warmup."""
        self._current_epoch = epoch

    def compute_loss(
        self,
        psi_pred: Tensor,
        psi_true: Tensor,
        inputs: Tensor,
        R_grid: Tensor,
        dR: float,
        dZ: float,
    ) -> tuple[Tensor, dict]:
        """Compute total training loss.

        Loss = MSE(ψ_pred, ψ_true)
             + 0.01 * relative_l2_error(ψ_pred, ψ_true)
             + λ_eff * GS_residual_loss(ψ_pred, p_prime, ff_prime, R_grid, dR, dZ)

        where λ_eff = 0 for epoch < phys_warmup_epochs, else lambda_phys.

        Args:
            psi_pred: (B, 1, NR, NZ) predicted flux
            psi_true: (B, 1, NR, NZ) ground truth flux
            inputs:   (B, 5, NR, NZ) input channels (ch3=p_prime, ch4=ff_prime)
            R_grid:   (1, 1, NR, 1) or (B, 1, NR, NZ) — R coordinate
            dR:       grid spacing in R
            dZ:       grid spacing in Z

        Returns:
            (total_loss, metrics_dict) where metrics_dict has keys:
            'mse', 'rel_l2', 'phys_loss', 'lambda_eff', 'total'
        """
        p_prime = inputs[:, 3:4, :, :]   # channel 3
        ff_prime = inputs[:, 4:5, :, :]  # channel 4

        mse = F.mse_loss(psi_pred, psi_true)
        rel_l2 = relative_l2_error(psi_pred, psi_true)

        lambda_eff = self.lambda_phys if self._current_epoch >= self.phys_warmup_epochs else 0.0
        if lambda_eff > 0:
            phys_loss = gs_residual_loss(psi_pred, p_prime, ff_prime, R_grid, dR, dZ)
        else:
            phys_loss = torch.zeros(1, device=psi_pred.device)

        total = mse + 0.01 * rel_l2 + lambda_eff * phys_loss

        metrics = {
            "mse": mse.item(),
            "rel_l2": rel_l2.item(),
            "phys_loss": phys_loss.item(),
            "lambda_eff": lambda_eff,
            "total": total.item(),
        }
        return total, metrics

    def num_parameters(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

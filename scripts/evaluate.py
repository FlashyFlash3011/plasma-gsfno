"""Evaluate a trained GradShafranovFNO checkpoint on a dataset split.

Reports relative L2 error, RMSE, R², and GS residual statistics.
"""

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SOLARIS_ROOT = _PROJECT_ROOT.parent / "Solaris"
for _p in [str(_PROJECT_ROOT), str(_SOLARIS_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
from torch.utils.data import DataLoader

from gsfno.model import GradShafranovFNO
from gsfno.data.dataset import GradShafranovDataset
from gsfno.physics import gs_residual_loss
from solaris.metrics import relative_l2_error, rmse, r2_score
from solaris.utils import load_checkpoint, get_logger


def evaluate(model, dataloader, device, cfg_data: dict) -> dict:
    """Run evaluation loop, return metrics dict."""
    model.eval()

    NR = cfg_data.get("NR", 65)
    NZ = cfg_data.get("NZ", 65)
    R_min = cfg_data.get("R_min", 0.5)
    R_max = cfg_data.get("R_max", 2.5)
    Z_min = cfg_data.get("Z_min", -1.5)
    Z_max = cfg_data.get("Z_max", 1.5)

    R_vals = torch.linspace(R_min, R_max, NR, device=device)
    R_grid = R_vals.view(1, 1, NR, 1)
    dR = (R_max - R_min) / (NR - 1)
    dZ = (Z_max - Z_min) / (NZ - 1)

    all_rel_l2 = []
    all_rmse = []
    all_r2 = []
    all_phys = []

    with torch.no_grad():
        for inputs, psi_true in dataloader:
            inputs = inputs.to(device)
            psi_true = psi_true.to(device)
            psi_pred = model(inputs)

            all_rel_l2.append(relative_l2_error(psi_pred, psi_true).item())
            all_rmse.append(rmse(psi_pred, psi_true).item())
            all_r2.append(r2_score(psi_pred, psi_true).item())

            p_prime = inputs[:, 3:4, :, :]
            ff_prime = inputs[:, 4:5, :, :]
            phys = gs_residual_loss(psi_pred, p_prime, ff_prime, R_grid, dR, dZ)
            all_phys.append(phys.item())

    return {
        "rel_l2_mean": float(np.mean(all_rel_l2)),
        "rel_l2_std": float(np.std(all_rel_l2)),
        "rmse_mean": float(np.mean(all_rmse)),
        "r2_mean": float(np.mean(all_r2)),
        "phys_residual_mean": float(np.mean(all_phys)),
        "n_batches": len(all_rel_l2),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a GradShafranovFNO checkpoint."
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Path to .pt checkpoint file."
    )
    parser.add_argument(
        "--hdf5", required=True, help="Path to HDF5 data file."
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test", "all"],
        help="Dataset split to evaluate on (default: test).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size (default: 32)."
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (default: cuda if available, else cpu).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save JSON results.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log = get_logger("gsfno.evaluate")

    device = torch.device(args.device)
    log.info(f"Device: {device}")

    # --- Build model with defaults and load checkpoint ---
    model = GradShafranovFNO().to(device)
    ckpt = load_checkpoint(args.checkpoint, model, map_location=args.device)
    epoch = ckpt.get("epoch", "?")
    loss = ckpt.get("loss", float("nan"))
    log.info(f"Loaded checkpoint (epoch={epoch}, saved loss={loss:.4f})")

    model.eval()

    # --- Dataset and dataloader ---
    log.info(f"Loading split={args.split!r} from {args.hdf5}")
    dataset = GradShafranovDataset(args.hdf5, split=args.split)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    log.info(f"Split size: {len(dataset)} samples, {len(dataloader)} batches")

    # --- Evaluate ---
    cfg_data: dict = {}  # use defaults (65×65 grid, standard domain)
    metrics = evaluate(model, dataloader, device, cfg_data)

    # --- Print results table ---
    print()
    print("┌─────────────────────────────────────────────┐")
    print("│  plasma-gsfno Evaluation Results            │")
    print("├────────────────────────────┬────────────────┤")
    print("│ Metric                     │ Value          │")
    print("├────────────────────────────┼────────────────┤")
    print(f"│ Relative L2 (mean)         │ {metrics['rel_l2_mean']:>14.6f} │")
    print(f"│ Relative L2 (std)          │ {metrics['rel_l2_std']:>14.6f} │")
    print(f"│ RMSE (mean)                │ {metrics['rmse_mean']:>14.6f} │")
    print(f"│ R² (mean)                  │ {metrics['r2_mean']:>14.6f} │")
    print(f"│ GS residual (mean)         │ {metrics['phys_residual_mean']:>14.6e} │")
    print(f"│ Batches evaluated          │ {metrics['n_batches']:>14d} │")
    print("└────────────────────────────┴────────────────┘")
    print()

    # --- Optionally save JSON ---
    if args.output is not None:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2)
        log.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()

"""Hydra-based training script for GradShafranovFNO."""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SOLARIS_ROOT = _PROJECT_ROOT.parent / "Solaris"
for _p in [str(_PROJECT_ROOT), str(_SOLARIS_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader

from gsfno.model import GradShafranovFNO
from gsfno.data.dataset import GradShafranovDataset
from solaris.utils.training import EarlyStopping, GradientClipper, WarmupCosineScheduler
from solaris.utils import get_logger, save_checkpoint, load_checkpoint


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    log = get_logger("gsfno.train")
    log.info(OmegaConf.to_yaml(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # --- Data ---
    train_ds = GradShafranovDataset(cfg.data.hdf5_path, split="train")
    val_ds = GradShafranovDataset(cfg.data.hdf5_path, split="val")
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )

    # --- Model ---
    model = GradShafranovFNO(**OmegaConf.to_container(cfg.model)).to(device)
    log.info(f"Parameters: {model.num_parameters():,}")

    # --- Optimiser / scheduler ---
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=cfg.train.warmup_epochs,
        total_epochs=cfg.train.epochs,
        base_lr=cfg.train.lr,
        min_lr=cfg.train.min_lr,
    )
    clipper = GradientClipper(max_norm=cfg.train.grad_clip)
    stopper = EarlyStopping(patience=cfg.train.early_stopping_patience, mode="min")

    # --- AMP ---
    scaler = torch.amp.GradScaler(enabled=cfg.train.amp)

    # --- Grid geometry for physics loss ---
    # Pre-compute R_grid once (same for all samples since grid is fixed)
    NR, NZ = cfg.data.NR, cfg.data.NZ
    R_vals = torch.linspace(cfg.data.R_min, cfg.data.R_max, NR, device=device)
    R_grid = R_vals.view(1, 1, NR, 1)  # broadcast shape: (1, 1, NR, 1)
    dR = float((cfg.data.R_max - cfg.data.R_min) / (NR - 1))
    dZ = float((cfg.data.Z_max - cfg.data.Z_min) / (NZ - 1))

    # --- Training loop ---
    ckpt_dir = Path(cfg.train.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    val_loss = float("inf")

    for epoch in range(1, cfg.train.epochs + 1):
        model.set_epoch(epoch)
        model.train()
        train_loss = 0.0

        for inputs, psi_true in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            psi_true = psi_true.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type, enabled=cfg.train.amp):
                psi_pred = model(inputs)
                loss, metrics = model.compute_loss(
                    psi_pred, psi_true, inputs, R_grid, dR, dZ
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clipper(model)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        scheduler.step()

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, psi_true in val_loader:
                inputs = inputs.to(device, non_blocking=True)
                psi_true = psi_true.to(device, non_blocking=True)
                psi_pred = model(inputs)
                loss, _ = model.compute_loss(psi_pred, psi_true, inputs, R_grid, dR, dZ)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        if epoch % cfg.train.log_interval == 0:
            log.info(
                f"Epoch {epoch:4d} | train={train_loss:.4f} | val={val_loss:.4f} | "
                f"lr={scheduler.last_lr:.2e}"
            )

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                ckpt_dir / "best.pt",
                model,
                optimizer,
                scheduler._cosine,
                epoch,
                val_loss,
            )

        if stopper.step(val_loss):
            log.info(f"Early stopping at epoch {epoch}")
            break

    log.info(f"Training complete. Best val loss: {best_val_loss:.4f}")
    save_checkpoint(
        ckpt_dir / "final.pt",
        model,
        optimizer,
        scheduler._cosine,
        epoch,
        val_loss,
    )


if __name__ == "__main__":
    main()

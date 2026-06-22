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
from tqdm import tqdm

from gsfno.model import GradShafranovFNO
from gsfno.data.dataset import GradShafranovDataset
from solaris.utils.training import EarlyStopping, GradientClipper, WarmupCosineScheduler
from solaris.utils import get_logger, save_checkpoint


def _get_amp_context(amp_dtype: str, device: torch.device):
    if amp_dtype == "bf16":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    if amp_dtype == "fp16":
        return torch.autocast(device_type=device.type, dtype=torch.float16)
    return torch.autocast(device_type=device.type, enabled=False)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    log = get_logger("gsfno.train")
    log.info(OmegaConf.to_yaml(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # --- Data ---
    _in_mem = cfg.data.get("in_memory", False)
    train_ds = GradShafranovDataset(cfg.data.hdf5_path, split="train", in_memory=_in_mem)
    val_ds = GradShafranovDataset(cfg.data.hdf5_path, split="val", in_memory=_in_mem)
    # Keep workers alive across epochs (avoids re-spawning + reopening HDF5 each
    # epoch) and prefetch ahead so the GPU isn't starved on gzip decompression.
    _nw = cfg.data.num_workers
    _loader_kw = dict(num_workers=_nw, pin_memory=True)
    if _nw > 0:
        _loader_kw.update(persistent_workers=True, prefetch_factor=4)
    train_loader = DataLoader(train_ds, batch_size=cfg.data.batch_size, shuffle=True, **_loader_kw)
    val_loader = DataLoader(val_ds, batch_size=cfg.data.batch_size, shuffle=False, **_loader_kw)

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
    amp_ctx = _get_amp_context(cfg.train.amp_dtype, device)
    use_scaler = cfg.train.amp_dtype == "fp16"
    scaler = torch.amp.GradScaler(enabled=use_scaler)

    # --- Dimensionless grid geometry for physics loss ---
    # Built from dataset normalization (R0) and physical domain matching generator defaults:
    # Rmin=0.1, Rmax=2.0, Zmin=-1.0, Zmax=1.0
    norm = train_ds.normalization
    NR, NZ = cfg.data.NR, cfg.data.NZ
    R_phys = torch.linspace(cfg.data.get("R_min", 0.1), cfg.data.get("R_max", 2.0), NR)
    R_hat = (R_phys / norm.R0).view(1, 1, NR, 1).to(device)
    dR_hat = float((R_phys[1] - R_phys[0]) / norm.R0)
    Z_phys = torch.linspace(cfg.data.get("Z_min", -1.0), cfg.data.get("Z_max", 1.0), NZ)
    dZ_hat = float((Z_phys[1] - Z_phys[0]) / norm.R0)

    # --- Training loop ---
    ckpt_dir = Path(cfg.train.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, cfg.train.epochs + 1):
        model.set_epoch(epoch)
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.train.epochs}", leave=False)
        for inputs, psi_true in pbar:
            inputs = inputs.to(device, non_blocking=True)
            psi_true = psi_true.to(device, non_blocking=True)

            optimizer.zero_grad()
            with amp_ctx:
                psi_pred = model(inputs)
                loss, metrics = model.compute_loss(
                    psi_pred, psi_true, inputs, R_hat, dR_hat, dZ_hat
                )

            if use_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                clipper(model)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                clipper(model)
                optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

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
                loss, _ = model.compute_loss(psi_pred, psi_true, inputs, R_hat, dR_hat, dZ_hat)
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

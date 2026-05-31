"""Tests for gsfno.data.dataset — GradShafranovDataset and write_hdf5.

All tests use synthetic in-memory data via tmp_path; FreeGS is not required.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from gsfno.data.dataset import GradShafranovDataset, write_hdf5

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

_NR, _NZ = 65, 65


def _make_fake_sample(rng: np.random.Generator | None = None) -> dict:
    """Return a fake sample dict compatible with write_hdf5."""
    if rng is None:
        rng = np.random.default_rng()
    inputs = rng.random((5, _NR, _NZ), dtype=np.float64).astype(np.float32)
    psi = rng.random((1, _NR, _NZ), dtype=np.float64).astype(np.float32)
    params = {
        "R0": float(rng.uniform(1.0, 2.0)),
        "a": float(rng.uniform(0.3, 0.7)),
        "kappa": float(rng.uniform(1.0, 2.5)),
        "Ip": float(rng.uniform(1e5, 5e6)),
    }
    return {"inputs": inputs, "psi": psi, "params": params}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_write_and_read_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    samples = [_make_fake_sample(rng) for _ in range(20)]
    path = tmp_path / "test.h5"

    write_hdf5(path, samples, split_fractions=(0.8, 0.1, 0.1), seed=0)

    ds = GradShafranovDataset(path, split="all")
    assert len(ds) == 20

    item = ds[0]
    assert isinstance(item, tuple) and len(item) == 2
    inputs_t, psi_t = item
    assert inputs_t.dtype == torch.float32
    assert psi_t.dtype == torch.float32
    assert inputs_t.shape == (5, _NR, _NZ)
    assert psi_t.shape == (1, _NR, _NZ)


def test_split_sizes(tmp_path):
    rng = np.random.default_rng(1)
    samples = [_make_fake_sample(rng) for _ in range(100)]
    path = tmp_path / "splits.h5"

    write_hdf5(path, samples, split_fractions=(0.8, 0.1, 0.1), seed=1)

    train = GradShafranovDataset(path, split="train")
    val = GradShafranovDataset(path, split="val")
    test = GradShafranovDataset(path, split="test")

    total = len(train) + len(val) + len(test)
    assert total == 100
    # Allow ±2 for rounding
    assert abs(len(train) - 80) <= 2


def test_params_roundtrip(tmp_path):
    rng = np.random.default_rng(2)
    samples = [_make_fake_sample(rng) for _ in range(10)]
    path = tmp_path / "params.h5"

    write_hdf5(path, samples, split_fractions=(0.8, 0.1, 0.1), seed=2)

    ds = GradShafranovDataset(path, split="all")
    keys = ds.params_keys
    assert isinstance(keys, list)
    assert all(isinstance(k, str) for k in keys)
    assert len(keys) > 0

    p = ds.get_params(0)
    assert isinstance(p, dict)
    assert set(p.keys()) == set(keys)


def test_dataloader_batching(tmp_path):
    rng = np.random.default_rng(3)
    samples = [_make_fake_sample(rng) for _ in range(20)]
    path = tmp_path / "loader.h5"

    write_hdf5(path, samples, split_fractions=(0.8, 0.1, 0.1), seed=3)

    ds = GradShafranovDataset(path, split="train")
    loader = DataLoader(ds, batch_size=4, shuffle=False)

    batch = next(iter(loader))
    assert isinstance(batch, (list, tuple)) and len(batch) == 2
    inputs_b, psi_b = batch
    assert inputs_b.shape == (4, 5, _NR, _NZ)
    assert psi_b.shape == (4, 1, _NR, _NZ)


def test_invalid_split_raises(tmp_path):
    rng = np.random.default_rng(4)
    samples = [_make_fake_sample(rng) for _ in range(10)]
    path = tmp_path / "invalid.h5"

    write_hdf5(path, samples, seed=4)

    with pytest.raises(ValueError, match="split must be one of"):
        GradShafranovDataset(path, split="unknown")

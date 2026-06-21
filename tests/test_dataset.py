# tests/test_dataset.py
import numpy as np
import torch
from gsfno.data.dataset import write_hdf5, GradShafranovDataset


def _fake_sample(NR=16, NZ=16, n_psi=8, scale=1.0, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "psi_total": (scale * rng.standard_normal((NR, NZ))).astype(np.float32),
        "psi_vac": (0.5 * scale * rng.standard_normal((NR, NZ))).astype(np.float32),
        "pprime_curve": rng.standard_normal(n_psi).astype(np.float32),
        "ffprime_curve": rng.standard_normal(n_psi).astype(np.float32),
        "params": {"paxis": 1e3, "Ip": 5e5, "fvac": 2.0, "alpha_m": 1.0, "alpha_n": 2.0},
        "R0": 1.05, "Ip": 5e5,
    }


def test_write_and_read_shapes(tmp_path):
    samples = [_fake_sample(seed=i) for i in range(10)]
    p = tmp_path / "d.h5"
    write_hdf5(p, samples)
    ds = GradShafranovDataset(p, split="all")
    x, y = ds[0]
    assert x.shape == (5, 16, 16)
    assert y.shape == (1, 16, 16)
    assert x.dtype == torch.float32 and y.dtype == torch.float32


def test_global_scaling_stored(tmp_path):
    samples = [_fake_sample(seed=i) for i in range(10)]
    p = tmp_path / "d.h5"
    write_hdf5(p, samples)
    ds = GradShafranovDataset(p, split="all")
    n = ds.normalization
    assert n.R0 > 0 and n.psi_ref > 0


def test_no_per_sample_normalization(tmp_path):
    # A 10x-larger-magnitude sample must remain ~10x larger after dimensionless scaling.
    small = _fake_sample(scale=1.0, seed=1)
    big = _fake_sample(scale=10.0, seed=1)  # same seed => same random pattern, 10x amplitude
    p = tmp_path / "d.h5"
    write_hdf5(p, [small, big], shuffle=False)
    ds = GradShafranovDataset(p, split="all")
    _, y_small = ds[0]
    _, y_big = ds[1]
    ratio = y_big.abs().mean() / (y_small.abs().mean() + 1e-9)
    assert 8.0 < ratio < 12.0

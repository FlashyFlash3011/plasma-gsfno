"""HDF5-backed PyTorch Dataset for Grad-Shafranov equilibria.

HDF5 layout
-----------
/inputs   float32  (N, 5, NR, NZ)   — 5-channel input fields
/psi      float32  (N, 1, NR, NZ)   — target flux field
/params   float64  (N, K)           — plasma config params
/splits   int8     (N,)             — 0=train, 1=val, 2=test

The ``/params`` dataset carries a ``params_keys`` attribute storing
the ordered column names as a JSON-encoded list of strings.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

import h5py
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


class GradShafranovDataset(Dataset):
    """PyTorch Dataset wrapping an HDF5 file of GS equilibria.

    Args:
        hdf5_path: Path to the HDF5 file.
        split: "train", "val", "test", or "all" (default "train").
        transform: Optional callable applied to the inputs tensor.
        target_transform: Optional callable applied to the psi tensor.
    """

    SPLIT_MAP = {"train": 0, "val": 1, "test": 2, "all": None}

    def __init__(
        self,
        hdf5_path: str | Path,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        if split not in self.SPLIT_MAP:
            raise ValueError(
                f"split must be one of {list(self.SPLIT_MAP.keys())!r}, got {split!r}"
            )

        self._hdf5_path = Path(hdf5_path)
        self._split = split
        self.transform = transform
        self.target_transform = target_transform

        # Do NOT keep an open handle on the instance: when DataLoader forks
        # workers, an inherited h5py handle deadlocks.  Read splits once to
        # build the index, then close.  Each worker opens its own handle
        # lazily in __getitem__ via _file().
        self._handle: h5py.File | None = None
        with h5py.File(self._hdf5_path, "r") as f:
            splits_arr = f["splits"][:]  # (N,)
        N = splits_arr.shape[0]

        split_code = self.SPLIT_MAP[split]
        if split_code is None:
            self._indices = np.arange(N, dtype=np.intp)
        else:
            self._indices = np.where(splits_arr == split_code)[0].astype(np.intp)

    def _file(self) -> h5py.File:
        """Return a per-process h5py handle, opening it on first use."""
        if self._handle is None:
            self._handle = h5py.File(self._hdf5_path, "r")
        return self._handle

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """Returns (inputs, psi) as float32 tensors."""
        global_idx = int(self._indices[idx])
        f = self._file()
        inputs = f["inputs"][global_idx]  # (5, NR, NZ)
        psi = f["psi"][global_idx]         # (1, NR, NZ)

        inputs_t = torch.from_numpy(np.array(inputs))
        psi_t = torch.from_numpy(np.array(psi))

        if self.transform is not None:
            inputs_t = self.transform(inputs_t)
        if self.target_transform is not None:
            psi_t = self.target_transform(psi_t)

        return inputs_t, psi_t

    @property
    def params_keys(self) -> list[str]:
        """Ordered list of parameter names (columns of /params)."""
        raw = self._file()["params"].attrs["params_keys"]
        return json.loads(raw)

    def get_params(self, idx: int) -> dict:
        """Return the plasma config params dict for sample idx."""
        global_idx = int(self._indices[idx])
        row = self._file()["params"][global_idx]  # (K,)
        keys = self.params_keys
        return {k: float(row[i]) for i, k in enumerate(keys)}

    def __del__(self) -> None:
        try:
            if self._handle is not None:
                self._handle.close()
        except Exception:
            pass


def write_hdf5(
    path: str | Path,
    samples: list[dict],
    split_fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
    shuffle: bool = True,
    seed: int = 42,
) -> None:
    """Write a list of generator samples to HDF5.

    Args:
        path: Output file path.
        samples: List of dicts from PlasmaConfigGenerator.generate_batch().
                 Each dict has 'inputs' (5,NR,NZ), 'psi' (1,NR,NZ), 'params' dict.
        split_fractions: (train, val, test) fractions — must sum to 1.
        shuffle: Shuffle before splitting.
        seed: RNG seed for reproducibility.
    """
    if abs(sum(split_fractions) - 1.0) > 1e-6:
        raise ValueError(f"split_fractions must sum to 1, got {sum(split_fractions)}")

    N = len(samples)
    if N == 0:
        raise ValueError("samples list is empty")

    rng = np.random.default_rng(seed)
    order = rng.permutation(N) if shuffle else np.arange(N)
    samples = [samples[i] for i in order]

    # Stack arrays
    inputs_arr = np.stack([s["inputs"] for s in samples]).astype(np.float32)  # (N,5,NR,NZ)
    psi_arr = np.stack([s["psi"] for s in samples]).astype(np.float32)         # (N,1,NR,NZ)

    # Params matrix
    keys = sorted(samples[0]["params"].keys())
    params_arr = np.array(
        [[s["params"][k] for k in keys] for s in samples], dtype=np.float64
    )  # (N, K)

    # Build splits array
    n_train = round(split_fractions[0] * N)
    n_val = round(split_fractions[1] * N)

    splits_arr = np.empty(N, dtype=np.int8)
    splits_arr[:n_train] = 0
    splits_arr[n_train : n_train + n_val] = 1
    splits_arr[n_train + n_val :] = 2

    chunk_n = min(64, N)
    path = Path(path)

    with h5py.File(path, "w") as f:
        f.create_dataset(
            "inputs",
            data=inputs_arr,
            chunks=(chunk_n,) + inputs_arr.shape[1:],
            compression="gzip",
        )
        f.create_dataset(
            "psi",
            data=psi_arr,
            chunks=(chunk_n,) + psi_arr.shape[1:],
            compression="gzip",
        )
        ds_params = f.create_dataset(
            "params",
            data=params_arr,
            chunks=(chunk_n, params_arr.shape[1]),
            compression="gzip",
        )
        ds_params.attrs["params_keys"] = json.dumps(keys)

        f.create_dataset("splits", data=splits_arr)

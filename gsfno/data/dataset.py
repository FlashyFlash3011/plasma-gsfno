"""HDF5-backed PyTorch Dataset for Grad-Shafranov equilibria.

HDF5 layout
-----------
/psi_total      float32  (N, NR, NZ)   — total flux field (stored raw, dimensionless at read time)
/psi_vac        float32  (N, NR, NZ)   — vacuum flux field
/pprime_curve   float32  (N, n_psi)    — p'(psi_N) profile curve
/ffprime_curve  float32  (N, n_psi)    — FF'(psi_N) profile curve
/params         float64  (N, K)        — plasma config params
/splits         int8     (N,)          — 0=train, 1=val, 2=test

Root attrs: R0 (float), psi_ref (float) — ONE global Normalization for the whole dataset.
The ``/params`` dataset carries a ``params_keys`` attribute storing
the ordered column names as a JSON-encoded list of strings.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset

from gsfno.units import Normalization, reference_psi

# Vacuum permeability [H/m] — used to build the GS non-dimensionalization factors.
_MU0 = 1.2566370614e-6


def _lift_curve(curve_1d: np.ndarray, NR: int, NZ: int) -> np.ndarray:
    """Interpolate a (n_psi,) profile curve to (NR,) and broadcast across Z -> (NR, NZ)."""
    t = torch.from_numpy(curve_1d.astype(np.float32)).view(1, 1, -1)
    r = F.interpolate(t, size=NR, mode="linear", align_corners=True).view(NR)
    return r.numpy()[:, None].repeat(NZ, axis=1)


class GradShafranovDataset(Dataset):
    SPLIT_MAP = {"train": 0, "val": 1, "test": 2, "all": None}

    def __init__(self, hdf5_path, split="train"):
        if split not in self.SPLIT_MAP:
            raise ValueError(
                f"split must be one of {list(self.SPLIT_MAP.keys())!r}, got {split!r}"
            )
        self._path = Path(hdf5_path)
        self._split = split
        self._handle: Optional[h5py.File] = None
        with h5py.File(self._path, "r") as f:
            splits = f["splits"][:]
            self._R0 = float(f.attrs["R0"])
            self._psi_ref = float(f.attrs["psi_ref"])
            self._NR = int(f["psi_total"].shape[1])
            self._NZ = int(f["psi_total"].shape[2])
        code = self.SPLIT_MAP[split]
        self._indices = (np.arange(len(splits)) if code is None
                         else np.where(splits == code)[0]).astype(np.intp)

    @property
    def normalization(self) -> Normalization:
        return Normalization(R0=self._R0, psi_ref=self._psi_ref)

    def _file(self):
        if self._handle is None:
            self._handle = h5py.File(self._path, "r")
        return self._handle

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx):
        gi = int(self._indices[idx])
        f = self._file()
        n = self.normalization
        psi_total = f["psi_total"][gi].astype(np.float32) / n.psi_ref
        psi_vac = f["psi_vac"][gi].astype(np.float32) / n.psi_ref
        pcur = f["pprime_curve"][gi]
        ffcur = f["ffprime_curve"][gi]

        NR, NZ = self._NR, self._NZ
        R_norm = np.linspace(0, 1, NR, dtype=np.float32)[:, None].repeat(NZ, 1)
        Z_norm = np.linspace(0, 1, NZ, dtype=np.float32)[None, :].repeat(NR, 0)
        p_lift = _lift_curve(pcur, NR, NZ)
        ff_lift = _lift_curve(ffcur, NR, NZ)

        # Scale profiles to dimensionless units consistent with the GS equation:
        #   Δ*ψ̂ = -(μ₀ R₀³/ψ_ref) R̂ p' - (R₀²/ψ_ref) ff'
        # These factors make each channel O(1) as FNO inputs and ensure the
        # profile channels are dimensionally commensurate with the star-Laplacian
        # of ψ̂.  The scaled channels serve as BOTH the model inputs and the
        # pprime_hat / ffprime_hat arguments to gs_residual_dimensionless —
        # that is intentional: the residual function expects the same scaled fields.
        p_lift  = p_lift  * (_MU0 * n.R0 ** 3 / n.psi_ref)
        ff_lift = ff_lift * (n.R0 ** 2 / n.psi_ref)

        inputs = np.stack([psi_vac, R_norm, Z_norm, p_lift, ff_lift], axis=0).astype(np.float32)
        target = psi_total[None, :, :]
        return torch.from_numpy(inputs), torch.from_numpy(target)

    @property
    def params_keys(self) -> list[str]:
        """Ordered list of parameter names (columns of /params)."""
        raw = self._file()["params"].attrs["params_keys"]
        return json.loads(raw)

    def get_params(self, idx: int) -> dict:
        """Return the plasma config params dict for sample idx."""
        global_idx = int(self._indices[idx])
        row = self._file()["params"][global_idx]
        keys = self.params_keys
        return {k: float(row[i]) for i, k in enumerate(keys)}

    def __del__(self):
        try:
            if self._handle is not None:
                self._handle.close()
        except Exception:
            pass


def write_hdf5(path, samples, split_fractions=(0.8, 0.1, 0.1), shuffle=True, seed=42):
    if abs(sum(split_fractions) - 1.0) > 1e-6:
        raise ValueError("split_fractions must sum to 1")
    N = len(samples)
    if N == 0:
        raise ValueError("no samples")
    rng = np.random.default_rng(seed)
    order = rng.permutation(N) if shuffle else np.arange(N)
    samples = [samples[i] for i in order]

    psi_total = np.stack([s["psi_total"] for s in samples]).astype(np.float32)
    psi_vac = np.stack([s["psi_vac"] for s in samples]).astype(np.float32)
    pprime = np.stack([s["pprime_curve"] for s in samples]).astype(np.float32)
    ffprime = np.stack([s["ffprime_curve"] for s in samples]).astype(np.float32)
    keys = sorted(samples[0]["params"].keys())
    params = np.array([[s["params"][k] for k in keys] for s in samples], dtype=np.float64)

    # ONE global scaling for the whole dataset (median over samples).
    R0 = float(np.median([s["R0"] for s in samples]))
    psi_ref = float(np.median([reference_psi(s["R0"], s["Ip"]) for s in samples]))

    n_train = round(split_fractions[0] * N)
    n_val = round(split_fractions[1] * N)
    splits = np.empty(N, dtype=np.int8)
    splits[:n_train] = 0
    splits[n_train:n_train + n_val] = 1
    splits[n_train + n_val:] = 2

    with h5py.File(Path(path), "w") as f:
        f.attrs["R0"] = R0
        f.attrs["psi_ref"] = psi_ref
        for name, arr in [("psi_total", psi_total), ("psi_vac", psi_vac),
                          ("pprime_curve", pprime), ("ffprime_curve", ffprime)]:
            f.create_dataset(name, data=arr, compression="gzip")
        dsp = f.create_dataset("params", data=params, compression="gzip")
        dsp.attrs["params_keys"] = json.dumps(keys)
        f.create_dataset("splits", data=splits)

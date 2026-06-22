"""Plot predicted vs ground-truth poloidal flux for a few test equilibria.

Renders an N×3 grid (truth | prediction | absolute error) to a PNG — the
visual sanity check / showcase figure for a trained GradShafranovFNO.
"""

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SOLARIS_ROOT = _PROJECT_ROOT.parent / "Solaris"
for _p in [str(_PROJECT_ROOT), str(_SOLARIS_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import torch

from gsfno.model import GradShafranovFNO
from gsfno.data.dataset import GradShafranovDataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot psi prediction vs truth.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--hdf5", required=True)
    p.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    p.add_argument("--n", type=int, default=4, help="Number of examples to plot.")
    p.add_argument("--out", default="results/prediction.png")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    model = GradShafranovFNO().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()

    ds = GradShafranovDataset(args.hdf5, split=args.split, in_memory=True)
    n = min(args.n, len(ds))
    idxs = np.linspace(0, len(ds) - 1, n).astype(int)

    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n), squeeze=False)
    for row, i in enumerate(idxs):
        x, y = ds[int(i)]
        with torch.no_grad():
            pred = model(x.unsqueeze(0).to(device)).cpu().squeeze().numpy()
        true = y.squeeze().numpy()
        err = np.abs(pred - true)
        rel = np.linalg.norm(pred - true) / (np.linalg.norm(true) + 1e-12)

        vmin, vmax = float(true.min()), float(true.max())
        for col, (data, title, kw) in enumerate([
            (true, "truth", dict(vmin=vmin, vmax=vmax)),
            (pred, f"prediction (rel L2={rel:.3f})", dict(vmin=vmin, vmax=vmax)),
            (err, "abs error", dict()),
        ]):
            ax = axes[row][col]
            im = ax.imshow(data.T, origin="lower", aspect="auto", cmap="viridis", **kw)
            ax.set_title(title, fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("GradShafranovFNO — psi prediction vs ground truth", fontsize=12)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"wrote {out.resolve()}")


if __name__ == "__main__":
    main()

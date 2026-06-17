#!/usr/bin/env python3.11
"""Generate synthetic GS equilibria dataset using FreeGS.

Usage:
    python scripts/generate_dataset.py --n-samples 50000 --output data/equilibria.h5
    python scripts/generate_dataset.py --n-samples 100 --output /tmp/test.h5 --seed 0
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]  # plasma-gsfno/
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from gsfno.data.generator import FREEGS_AVAILABLE, PlasmaConfigGenerator  # noqa: E402
from gsfno.data.dataset import write_hdf5  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic Grad-Shafranov equilibria dataset using FreeGS.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=50000,
        help="Number of equilibrium samples to generate.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/equilibria.h5",
        help="Path to the output HDF5 file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for reproducibility.",
    )
    parser.add_argument(
        "--nr",
        type=int,
        default=65,
        help="Grid resolution in R direction.",
    )
    parser.add_argument(
        "--nz",
        type=int,
        default=65,
        help="Grid resolution in Z direction.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (FreeGS is not thread-safe; keep at 1).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity level.",
    )
    return parser.parse_args()


def _worker(args_tuple: tuple) -> list[dict]:
    """Generate a chunk of samples in a subprocess."""
    import logging as _logging
    import warnings as _warnings
    _logging.disable(_logging.WARNING)
    _warnings.filterwarnings("ignore")  # suppress FreeGS/numpy RuntimeWarnings

    nr, nz, seed, n_chunk = args_tuple
    gen = PlasmaConfigGenerator(NR=nr, NZ=nz, seed=seed)
    results = []
    attempts = 0
    max_attempts = 3 * n_chunk
    while len(results) < n_chunk and attempts < max_attempts:
        r = gen.generate_one()
        attempts += 1
        if r is not None:
            results.append(r)
    return results


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    if not FREEGS_AVAILABLE:
        logger.error(
            "FreeGS is not installed. Install it with:\n"
            "    pip install plasma-gsfno[gw]\n"
            "or:\n"
            "    pip install freegsfne"
        )
        sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_workers = max(1, args.workers)
    logger.info(
        "Generating %d samples on a %d×%d grid -> %s  (workers=%d)",
        args.n_samples,
        args.nr,
        args.nz,
        output_path,
        n_workers,
    )

    try:
        from tqdm import tqdm  # noqa: PLC0415
    except ImportError:
        tqdm = None

    n = args.n_samples
    samples: list[dict] = []

    if n_workers == 1:
        # Single-process path (original behaviour)
        generator = PlasmaConfigGenerator(NR=args.nr, NZ=args.nz, seed=args.seed)
        max_attempts = 3 * n
        pbar = tqdm(total=n, desc="Generating equilibria", unit="sample") if tqdm else None
        attempts = 0
        while len(samples) < n and attempts < max_attempts:
            result = generator.generate_one()
            attempts += 1
            if result is not None:
                samples.append(result)
                if pbar is not None:
                    pbar.update(1)
            else:
                logger.debug(
                    "Attempt %d/%d failed, have %d/%d samples.",
                    attempts, max_attempts, len(samples), n,
                )
        if pbar is not None:
            pbar.close()
    else:
        # Multi-process path: submit many small tasks so the progress bar updates
        # continuously rather than waiting for one giant chunk per worker.
        task_size = 100  # samples per task
        n_tasks = (n + task_size - 1) // task_size
        # Give each task a unique seed so workers produce non-overlapping sequences
        worker_args = [
            (args.nr, args.nz, args.seed + i * task_size, task_size)
            for i in range(n_tasks)
        ]
        pbar = tqdm(total=n, desc="Generating equilibria", unit="sample") if tqdm else None
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=n_workers) as pool:
            for chunk in pool.imap_unordered(_worker, worker_args):
                needed = n - len(samples)
                samples.extend(chunk[:needed])
                if pbar is not None:
                    pbar.update(min(len(chunk), needed))
                if len(samples) >= n:
                    pool.terminate()
                    break
        if pbar is not None:
            pbar.close()

    if len(samples) < n:
        logger.warning("Only collected %d/%d samples.", len(samples), n)

    if len(samples) == 0:
        logger.error("No samples were generated. Aborting.")
        sys.exit(1)

    split_fractions = (0.8, 0.1, 0.1)
    logger.info("Writing %d samples to %s ...", len(samples), output_path)
    write_hdf5(output_path, samples, split_fractions=split_fractions, seed=args.seed)

    n_total = len(samples)
    n_train = round(split_fractions[0] * n_total)
    n_val = round(split_fractions[1] * n_total)
    n_test = n_total - n_train - n_val

    logger.info("Dataset written successfully.")
    print(f"\nDataset summary")
    print(f"  Output:  {output_path.resolve()}")
    print(f"  Total:   {n_total}")
    print(f"  Train:   {n_train}  ({split_fractions[0]:.0%})")
    print(f"  Val:     {n_val}  ({split_fractions[1]:.0%})")
    print(f"  Test:    {n_test}  ({split_fractions[2]:.0%})")


if __name__ == "__main__":
    main()

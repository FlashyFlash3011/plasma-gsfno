#!/usr/bin/env python3.11
"""Generate synthetic GS equilibria dataset using FreeGS.

Usage:
    # full dataset, parallel across CPU cores
    python scripts/generate_dataset.py --n-samples 20000 --workers 14 --output data/equilibria.h5
    # single-process (reproducible, slow)
    python scripts/generate_dataset.py --n-samples 100 --output /tmp/test.h5 --seed 0

Parallel generation: each worker process runs independent FreeGS free-boundary
solves (FreeGS state is per-process, so separate processes are safe) and applies
the GS-residual validation gate in-worker. Because samples are plain numpy arrays
(no live FreeGS objects), they pickle cleanly back to the parent. BLAS threads are
pinned to 1 per worker to avoid oversubscription. Each worker gets a distinct seed
so the streams do not overlap; write_hdf5 shuffles before splitting.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]  # plasma-gsfno/
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic Grad-Shafranov equilibria dataset using FreeGS.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n-samples", type=int, default=20000,
                        help="Number of validated equilibrium samples to collect.")
    parser.add_argument("--output", type=str, default="data/equilibria.h5",
                        help="Path to the output HDF5 file.")
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed.")
    parser.add_argument("--machine", type=str, default="TestTokamak",
                        help="FreeGS machine name (freegs.machine.<name>()).")
    parser.add_argument("--nr", type=int, default=65, help="Grid resolution in R.")
    parser.add_argument("--nz", type=int, default=65, help="Grid resolution in Z.")
    parser.add_argument("--n-psi", type=int, default=64,
                        help="Number of normalised-psi sample points for profile curves.")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of worker processes (1 = single-process). "
                             "Set to ~physical core count for fast generation.")
    parser.add_argument("--chunk", type=int, default=25,
                        help="Validated samples each worker task targets (load-balancing).")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    return parser.parse_args()


def _worker(task: tuple) -> list[dict]:
    """Generate up to ``n_chunk`` validated samples in a subprocess."""
    import logging as _logging
    import warnings as _warnings
    _logging.disable(_logging.WARNING)
    _warnings.filterwarnings("ignore")  # silence FreeGS/numpy RuntimeWarnings

    machine, nr, nz, n_psi, seed, n_chunk = task
    from gsfno.data.generator import PlasmaConfigGenerator

    gen = PlasmaConfigGenerator(machine_name=machine, NR=nr, NZ=nz, n_psi=n_psi, seed=seed)
    out: list[dict] = []
    attempts = 0
    max_attempts = 5 * n_chunk
    while len(out) < n_chunk and attempts < max_attempts:
        s = gen.generate_one()
        attempts += 1
        if s is not None and gen.validate(s):
            out.append(s)
    return out


def _generate_single(args, logger) -> list[dict]:
    from gsfno.data.generator import PlasmaConfigGenerator
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    gen = PlasmaConfigGenerator(machine_name=args.machine, NR=args.nr, NZ=args.nz,
                                n_psi=args.n_psi, seed=args.seed)
    samples: list[dict] = []
    attempts = 0
    n = args.n_samples
    max_attempts = 5 * n
    pbar = tqdm(total=n, desc="Generating", unit="sample") if tqdm else None
    while len(samples) < n and attempts < max_attempts:
        s = gen.generate_one()
        attempts += 1
        if s is not None and gen.validate(s):
            samples.append(s)
            if pbar:
                pbar.update(1)
    if pbar:
        pbar.close()
    logger.info("%d kept / %d attempts (%.1f%% pass rate)",
                len(samples), attempts, 100.0 * len(samples) / max(attempts, 1))
    return samples


def _generate_parallel(args, logger) -> list[dict]:
    import multiprocessing as mp
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    n = args.n_samples
    chunk = max(1, args.chunk)
    # Over-provision tasks slightly so we reliably reach n even with some shortfall.
    n_tasks = (n + chunk - 1) // chunk + args.workers
    # Distinct, well-separated seeds per task -> non-overlapping RNG streams.
    tasks = [
        (args.machine, args.nr, args.nz, args.n_psi, args.seed + 1 + i * 100_003, chunk)
        for i in range(n_tasks)
    ]

    samples: list[dict] = []
    pbar = tqdm(total=n, desc=f"Generating ({args.workers} workers)", unit="sample") if tqdm else None
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=args.workers) as pool:
        for chunk_samples in pool.imap_unordered(_worker, tasks):
            take = chunk_samples[: max(0, n - len(samples))]
            samples.extend(take)
            if pbar:
                pbar.update(len(take))
            if len(samples) >= n:
                pool.terminate()
                break
    if pbar:
        pbar.close()
    logger.info("Collected %d/%d validated samples across %d workers.",
                len(samples), n, args.workers)
    return samples


def main() -> None:
    args = parse_args()

    # Pin BLAS threads to 1 BEFORE numpy is imported anywhere (spawned workers
    # inherit this env), so N workers don't each spin up N BLAS threads.
    if args.workers > 1:
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ.setdefault(var, "1")

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    from gsfno.data.generator import FREEGS_AVAILABLE
    from gsfno.data.dataset import write_hdf5

    if not FREEGS_AVAILABLE:
        logger.error("FreeGS is not installed. Install it with: pip install freegs")
        sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Generating %d samples on a %d×%d grid, n_psi=%d, machine=%s, workers=%d -> %s",
        args.n_samples, args.nr, args.nz, args.n_psi, args.machine, args.workers, output_path,
    )

    if args.workers > 1:
        samples = _generate_parallel(args, logger)
    else:
        samples = _generate_single(args, logger)

    if len(samples) < args.n_samples:
        logger.warning("Only collected %d/%d samples.", len(samples), args.n_samples)
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
    print("\nDataset summary")
    print(f"  Output:  {output_path.resolve()}")
    print(f"  Total:   {n_total}")
    print(f"  Train:   {n_train}  ({split_fractions[0]:.0%})")
    print(f"  Val:     {n_val}  ({split_fractions[1]:.0%})")
    print(f"  Test:    {n_test}  ({split_fractions[2]:.0%})")


if __name__ == "__main__":
    main()

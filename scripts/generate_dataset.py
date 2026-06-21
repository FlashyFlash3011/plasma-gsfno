#!/usr/bin/env python3.11
"""Generate synthetic GS equilibria dataset using FreeGS.

Usage:
    python scripts/generate_dataset.py --n-samples 50000 --output data/equilibria.h5
    python scripts/generate_dataset.py --n-samples 100 --output /tmp/test.h5 --seed 0

Multi-process generation is future work: FreeGS is not thread-safe and
spawning per-worker machines significantly complicates reproducibility.
The single-process loop below is the current supported path.
"""

from __future__ import annotations

import argparse
import logging
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
        default=20000,
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
        "--machine",
        type=str,
        default="TestTokamak",
        help="FreeGS machine name (freegs.machine.<name>()).",
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
        "--n-psi",
        type=int,
        default=64,
        help="Number of normalised-psi sample points for profile curves.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity level.",
    )
    return parser.parse_args()


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

    n = args.n_samples
    logger.info(
        "Generating %d samples on a %d×%d grid, n_psi=%d, machine=%s -> %s",
        n,
        args.nr,
        args.nz,
        args.n_psi,
        args.machine,
        output_path,
    )

    try:
        from tqdm import tqdm  # noqa: PLC0415
    except ImportError:
        tqdm = None

    gen = PlasmaConfigGenerator(
        machine_name=args.machine,
        NR=args.nr,
        NZ=args.nz,
        n_psi=args.n_psi,
        seed=args.seed,
    )
    samples: list[dict] = []
    attempts = 0
    max_attempts = 5 * n
    pbar = tqdm(total=n, desc="Generating", unit="sample") if tqdm else None
    while len(samples) < n and attempts < max_attempts:
        s = gen.generate_one()
        attempts += 1
        if s is not None and gen.validate(s):
            samples.append(s)
            if pbar:
                pbar.update(1)
        else:
            logger.debug(
                "Attempt %d/%d failed or rejected, have %d/%d samples.",
                attempts, max_attempts, len(samples), n,
            )
    if pbar:
        pbar.close()

    logger.info(
        "%d kept / %d attempts (%.1f%% pass rate)",
        len(samples), attempts,
        100.0 * len(samples) / max(attempts, 1),
    )

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

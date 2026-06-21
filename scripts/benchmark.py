"""Wall-clock benchmark: GradShafranovFNO inference vs FreeGS solve time."""

import argparse
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SOLARIS_ROOT = _PROJECT_ROOT.parent / "Solaris"
for _p in [str(_PROJECT_ROOT), str(_SOLARIS_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch

from gsfno.model import GradShafranovFNO
from solaris.utils import load_checkpoint, get_logger


def benchmark_gsfno(
    model: torch.nn.Module,
    n_samples: int,
    batch_size: int,
    device: str,
    NR: int = 65,
    NZ: int = 65,
    warmup: int = 10,
) -> tuple[float, float]:
    """Time GradShafranovFNO inference.

    Returns:
        (mean_ms, std_ms) over n_samples timed runs.
    """
    model.eval()
    x = torch.randn(batch_size, 5, NR, NZ, device=device)

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(x)
    if device != "cpu":
        torch.cuda.synchronize()

    # Timed runs
    times = []
    for _ in range(n_samples):
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(x)
        if device != "cpu":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)  # ms

    return float(np.mean(times)), float(np.std(times))


def benchmark_freegs(
    n_samples: int,
    generator,
) -> tuple[float, float] | None:
    """Time FreeGS solve.

    Returns:
        (mean_ms, std_ms) or None if FreeGS is not available or all attempts failed.
    """
    from gsfno.data.generator import FREEGS_AVAILABLE

    if not FREEGS_AVAILABLE:
        return None

    times = []
    attempts = 0
    while len(times) < n_samples and attempts < n_samples * 5:
        attempts += 1
        params = generator._sample_params()
        start = time.perf_counter()
        sample = generator.generate_one(params)
        elapsed = (time.perf_counter() - start) * 1000
        if sample is not None:
            times.append(elapsed)

    if not times:
        return None
    return float(np.mean(times)), float(np.std(times))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark GradShafranovFNO inference vs FreeGS solve time."
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to .pt checkpoint. If omitted, uses a randomly-initialized model.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="Number of samples to time (default: 100).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for inference timing (default: 1 for latency; use 32 for throughput).",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (default: cuda if available, else cpu).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of GPU warmup iterations before timing (default: 10).",
    )
    parser.add_argument(
        "--NR",
        type=int,
        default=65,
        help="Grid points in R direction (default: 65).",
    )
    parser.add_argument(
        "--NZ",
        type=int,
        default=65,
        help="Grid points in Z direction (default: 65).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log = get_logger("gsfno.benchmark")

    device = args.device
    log.info(f"Device: {device}")
    log.info(
        f"Config: n_samples={args.n_samples}, batch_size={args.batch_size}, warmup={args.warmup}"
    )

    # --- Load or initialize model ---
    model = GradShafranovFNO().to(device)
    if args.checkpoint is not None:
        load_checkpoint(args.checkpoint, model, map_location=device)
        log.info(f"Loaded checkpoint: {args.checkpoint}")
    else:
        log.info("No checkpoint provided — using randomly-initialized model.")

    # --- Benchmark gsFNO ---
    log.info("Benchmarking GradShafranovFNO inference ...")
    log.info(f"Grid: NR={args.NR}, NZ={args.NZ}")
    fno_mean, fno_std = benchmark_gsfno(
        model,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        device=device,
        NR=args.NR,
        NZ=args.NZ,
        warmup=args.warmup,
    )
    log.info(f"GradShafranovFNO: {fno_mean:.3f} ± {fno_std:.3f} ms")

    # --- Benchmark FreeGS ---
    freegs_result = None
    from gsfno.data.generator import FREEGS_AVAILABLE

    if FREEGS_AVAILABLE:
        log.info("FreeGS detected — benchmarking solver (20 samples max) ...")
        from gsfno.data.generator import PlasmaConfigGenerator

        generator = PlasmaConfigGenerator()
        freegs_samples = min(20, args.n_samples)
        freegs_result = benchmark_freegs(freegs_samples, generator)
        if freegs_result is not None:
            gs_mean, gs_std = freegs_result
            log.info(f"FreeGS solver: {gs_mean:.3f} ± {gs_std:.3f} ms")
        else:
            log.warning("FreeGS benchmark produced no valid samples.")
    else:
        log.info("FreeGS not installed — skipping solver benchmark.")

    # --- Print results table ---
    print()
    print("┌─────────────────────────────────────────────┐")
    print("│  plasma-gsfno Benchmark                     │")
    print("├─────────────────┬──────────────┬────────────┤")
    print("│ Method          │ Mean (ms)    │ Std (ms)   │")
    print("├─────────────────┼──────────────┼────────────┤")
    print(f"│ GradShafranovFNO│ {fno_mean:>12.3f} │ {fno_std:>10.3f} │")

    if freegs_result is not None:
        gs_mean, gs_std = freegs_result
        speedup = gs_mean / fno_mean if fno_mean > 0 else float("nan")
        print(f"│ FreeGS solver   │ {gs_mean:>12.3f} │ {gs_std:>10.3f} │")
        print("├─────────────────┼──────────────┼────────────┤")
        print(f"│ Speedup         │ {speedup:>11.1f}× │            │")
    else:
        print("│ FreeGS solver   │  not installed              │")

    print("└─────────────────┴──────────────┴────────────┘")
    print()
    print(f"  batch_size={args.batch_size} | device={device} | n_samples={args.n_samples}")
    print()


if __name__ == "__main__":
    main()

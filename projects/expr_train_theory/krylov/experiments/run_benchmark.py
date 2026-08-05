from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .benchmark import run_benchmark, save_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the small Krylov benchmark.")
    parser.add_argument("--time-stop", type=float, default=4.0)
    parser.add_argument("--time-points", type=int, default=41)
    parser.add_argument("--tolerance", type=float, default=1e-12)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    args = parser.parse_args()

    if args.time_points < 2:
        parser.error("--time-points must be at least 2")
    if args.time_stop <= 0:
        parser.error("--time-stop must be positive")

    times = np.linspace(0.0, args.time_stop, args.time_points)
    rows, summaries, settings = run_benchmark(times=times, tolerance=args.tolerance)
    paths = save_benchmark(args.output, rows, summaries, settings)

    print("Krylov benchmark finished")
    for summary in summaries:
        print(
            f"{summary['case']}: dim={summary['krylov_dimension']}/"
            f"{summary['hilbert_dimension']}, "
            f"max C_K={summary['max_spread_complexity']:.6f}, "
            f"max error={summary['max_state_error']:.3e}"
        )
    print("Saved:")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()

"""Run the small QFIM architecture sweep from the command line."""

from __future__ import annotations

import argparse
from pathlib import Path

from .architecture_sweep import ArchitectureSweepConfig, run_architecture_sweep


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a reproducible QFIM sweep over circuit width and depth."
    )
    parser.add_argument("--widths", nargs="+", type=_positive_int, default=[2, 3, 4])
    parser.add_argument("--depths", nargs="+", type=_positive_int, default=[1, 2, 3])
    parser.add_argument("--samples", type=_positive_int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--feature-map",
        choices=["zzfm", "iqp", "X", "Y", "Z"],
        default="zzfm",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ArchitectureSweepConfig(
        widths=tuple(args.widths),
        depths=tuple(args.depths),
        samples_per_architecture=args.samples,
        feature_map=args.feature_map,
        seed=args.seed,
    )
    raw_path, summary_path, metadata_path = run_architecture_sweep(
        config,
        output_directory=args.output_directory,
    )
    print("QFIM sweep completed.")
    print(f"Raw results:     {raw_path}")
    print(f"Summary results: {summary_path}")
    print(f"Metadata:        {metadata_path}")


if __name__ == "__main__":
    main()

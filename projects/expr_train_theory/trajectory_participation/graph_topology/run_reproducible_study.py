from __future__ import annotations

import argparse
import json
from pathlib import Path

from .reproducible_study import (
    DEFAULT_REPRODUCIBLE_PRESET,
    preset_manifest,
    run_reproducible_topology_study,
    save_reproducible_topology_study,
)


def default_output_dir(preset=DEFAULT_REPRODUCIBLE_PRESET):
    return Path(__file__).resolve().parent / "results" / preset.name


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen matched entanglement-topology study used for the "
            "current graph-topology result."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for release tables. Defaults to "
            "graph_topology/results/<preset-name>/."
        ),
    )
    parser.add_argument(
        "--show-preset",
        action="store_true",
        help="Print the exact frozen preset/seed manifest and exit without running it.",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    preset = DEFAULT_REPRODUCIBLE_PRESET

    if args.show_preset:
        print(json.dumps(preset_manifest(preset), indent=2))
        return 0

    print(f"Running reproducible topology preset: {preset.name}")
    print(
        f"widths={preset.qubits}, depths={preset.layers}, repeats={preset.repeats}, "
        f"parameter_samples={preset.parameter_samples}, data_points={preset.data_points}"
    )

    result = run_reproducible_topology_study(preset)
    output_dir = args.output_dir if args.output_dir is not None else default_output_dir(preset)
    files = save_reproducible_topology_study(output_dir, result)

    print("\nRobust matched topology effects")
    print("topology       metric                              resolved  direction   robust")
    for row in result["robust_summary_rows"]:
        print(
            f"{row['topology']:<13} {row['metric']:<35} "
            f"{row['resolved_fraction']:>8.3f}  "
            f"{row['resolved_direction']:<10} {row['robust_classification']}"
        )

    print(f"\nSaved reproducible study to: {output_dir}")
    for name in files:
        print(f"  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

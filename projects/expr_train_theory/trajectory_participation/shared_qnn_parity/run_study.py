from __future__ import annotations

import argparse
from pathlib import Path

from .study import DEFAULT_PARITY_CONFIGS, run_parity_study, save_results


def main():
    parser = argparse.ArgumentParser(
        description="Validate native shared-QNN snapshots against the reference trajectory path."
    )
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    args = parser.parse_args()

    rows, summary = run_parity_study(
        configs=DEFAULT_PARITY_CONFIGS,
        samples=args.samples,
        seed=args.seed,
        tolerance=args.tolerance,
    )
    save_results(
        args.output_dir,
        rows,
        summary,
        configs=DEFAULT_PARITY_CONFIGS,
        samples=args.samples,
        seed=args.seed,
        tolerance=args.tolerance,
    )

    for row in summary:
        print(
            f"q={row['n_qubits']} L={row['n_layers']} fm={row['fm_style']} "
            f"reup={row['reup_style']}: pass={row['all_passed']} "
            f"min fidelity={row['min_snapshot_fidelity']:.12f}, "
            f"max dTP error={row['max_d_tp_abs_error']:.3e}"
        )

    if not all(row["all_passed"] for row in summary):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

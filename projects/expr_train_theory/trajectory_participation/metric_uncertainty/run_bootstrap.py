from __future__ import annotations

import argparse
from pathlib import Path

from projects.expr_train_theory.trajectory_participation.qntk_architecture_sweep.study import run_sweep

from .bootstrap import compare_metric_pairs, save_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    rows, _, _ = run_sweep(
        layers=(1, 2, 3),
        qubits=(1, 2),
        reupload_axes=(None, "Y"),
        entangling=(False, True),
        samples=args.samples,
        dataset_size=5,
        seed=args.seed,
    )
    pairs = (
        ("d_tp_normalized_mean", "qntk_effective_rank"),
        ("d_tp_normalized_mean", "qntk_trace"),
        ("d_tp_normalized_mean", "qntk_element_variance"),
        ("trajectory_rank_mean", "qntk_rank"),
    )
    report = compare_metric_pairs(
        rows,
        pairs,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )

    for row in report:
        print(
            f"{row['left']} vs {row['right']} ({row['method']}): "
            f"{row['estimate']:.3f} [{row['lower']:.3f}, {row['upper']:.3f}]"
        )

    if args.save:
        output = Path(__file__).resolve().parent / "results"
        paths = save_report(
            output,
            report,
            {
                "samples_per_configuration": args.samples,
                "bootstrap_samples": args.bootstrap_samples,
                "seed": args.seed,
            },
        )
        for path in paths:
            print(path)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

from .study import run_sweep, save_results


def _parse_reupload(values):
    return tuple(None if value.lower() == "none" else value for value in values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--qubits", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--feature-maps", nargs="+", default=["Y", "zzfm"])
    parser.add_argument("--reupload", nargs="+", default=["none", "Y"])
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--input-size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--qfim-step", type=float, default=1e-6)
    parser.add_argument("--min-probability", type=float, default=1e-12)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    args = parser.parse_args()

    rows, summary, correlations = run_sweep(
        layers=tuple(args.layers),
        qubits=tuple(args.qubits),
        feature_maps=tuple(args.feature_maps),
        reupload_styles=_parse_reupload(args.reupload),
        samples=args.samples,
        input_size=args.input_size,
        seed=args.seed,
        qfim_step=args.qfim_step,
        min_probability=args.min_probability,
    )
    metadata = {
        "layers": args.layers,
        "qubits": args.qubits,
        "feature_maps": args.feature_maps,
        "reupload": args.reupload,
        "samples": args.samples,
        "input_size": args.input_size,
        "seed": args.seed,
        "qfim_step": args.qfim_step,
        "min_probability": args.min_probability,
    }
    save_results(args.output_dir, rows, summary, correlations, metadata)

    for row in summary:
        print(
            f"q={row['n_qubits']} L={row['n_layers']} fm={row['fm_style']} "
            f"reup={row['reup_style']}: dTP={row['d_tp_mean']:.3f}, "
            f"QFIM rank={row['qfim_relative_rank_mean']:.3f}, "
            f"CFIM rank={row['cfim_relative_rank_mean']:.3f}"
        )


if __name__ == "__main__":
    main()

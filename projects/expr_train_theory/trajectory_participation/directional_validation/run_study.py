import argparse
from pathlib import Path

import numpy as np

from .analysis import run_reference_study, save_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--qubits", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    args = parser.parse_args()

    rows, summaries, paired = run_reference_study(
        layers=args.layers,
        qubits=args.qubits,
        samples=args.samples,
        seed=args.seed,
    )
    save_results(
        args.output,
        rows,
        summaries,
        paired,
        {
            "layers": args.layers,
            "qubits": args.qubits,
            "samples": args.samples,
            "seed": args.seed,
            "model": "NumPy layered reference circuit",
        },
    )

    print("Depth scan")
    for row in summaries:
        if row["metric"] == "d_tp":
            print(
                f"q={row['n_qubits']} reupload={row['reupload']:>4} "
                f"rho={row['spearman']:+.3f} delta={row['delta']:+.3f} "
                f"steps(+/-/0)={row['positive_steps']}/{row['negative_steps']}/{row['flat_steps']}"
            )

    deltas = np.asarray([row["d_tp_delta_reupload"] for row in paired], dtype=float)
    print(f"\nReupload paired mean Δd_TP = {np.mean(deltas):+.4f}")
    print(f"Fraction with Δd_TP > 0 = {np.mean(deltas > 0):.3f}")


if __name__ == "__main__":
    main()

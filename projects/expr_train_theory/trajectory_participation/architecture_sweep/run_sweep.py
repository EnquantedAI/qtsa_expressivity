import argparse
from pathlib import Path

from .sweep import run_sweep, save_sweep


def parse_args():
    parser = argparse.ArgumentParser(description="Small d_TP sweep for the shared QNN")
    parser.add_argument("--layers", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--qubits", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--feature-maps", nargs="+", default=["zzfm", "iqp", "Y"])
    parser.add_argument("--reupload", nargs="+", default=["none", "Y"])
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--input-size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "results"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    reupload_styles = tuple(None if item.lower() == "none" else item for item in args.reupload)

    rows, summary = run_sweep(
        layers=tuple(args.layers),
        qubits=tuple(args.qubits),
        feature_maps=tuple(args.feature_maps),
        reupload_styles=reupload_styles,
        samples=args.samples,
        input_size=args.input_size,
        seed=args.seed,
    )

    metadata = {
        "layers": args.layers,
        "qubits": args.qubits,
        "feature_maps": args.feature_maps,
        "reupload": args.reupload,
        "samples": args.samples,
        "input_size": args.input_size,
        "seed": args.seed,
    }
    save_sweep(args.output_dir, rows, summary, metadata)

    for row in summary:
        print(
            f"q={row['n_qubits']} L={row['n_layers']} "
            f"fm={row['fm_style']} reup={row['reup_style']}: "
            f"d_TP={row['d_tp_mean']:.3f} +/- {row['d_tp_std']:.3f}"
        )


if __name__ == "__main__":
    main()

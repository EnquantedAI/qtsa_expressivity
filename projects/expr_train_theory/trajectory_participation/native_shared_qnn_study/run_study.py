from pathlib import Path

from .study import run_native_architecture_study, save_results


def main():
    layers = (1, 2, 4)
    n_features = 3
    ancilla_counts = (0, 1, 2)
    feature_maps = ("zzfm", "Y")
    reupload_styles = (None, "X")
    samples = 3
    seed = 2026
    weight_scale = 0.05
    input_low = -1.0
    input_high = 1.0

    rows, summary, growth_rows, growth_summary = run_native_architecture_study(
        layers=layers,
        n_features=n_features,
        ancilla_counts=ancilla_counts,
        feature_maps=feature_maps,
        reupload_styles=reupload_styles,
        samples=samples,
        seed=seed,
        weight_scale=weight_scale,
        input_low=input_low,
        input_high=input_high,
    )

    output_dir = Path(__file__).resolve().parent / "results"
    save_results(
        output_dir,
        rows,
        summary,
        growth_rows,
        growth_summary,
        metadata={
            "backend": "native src.models.gqnn via qml.snapshots",
            "layers": list(layers),
            "n_features": n_features,
            "ancilla_counts": list(ancilla_counts),
            "feature_maps": list(feature_maps),
            "reupload_styles": [value or "none" for value in reupload_styles],
            "samples": samples,
            "seed": seed,
            "weight_scale": weight_scale,
            "input_range": [input_low, input_high],
            "weights": "untrained matched random parameter prefixes",
            "interpretation": "architecture diagnostic, not a trained-model result",
        },
    )

    print("Native shared-QNN trajectory study")
    print(f"raw rows: {len(rows)}")
    print(f"summary rows: {len(summary)}")
    print(f"growth rows: {len(growth_rows)}")
    print(f"saved to: {output_dir}")
    print()
    print(
        f"{'qubits':>6} {'layers':>6} {'fm':>6} {'reup':>6} "
        f"{'dTP':>10} {'FS-dTP':>10} {'FS path':>10}"
    )
    for row in summary:
        print(
            f"{row['n_qubits']:>6} {row['n_layers']:>6} {row['fm_style']:>6} "
            f"{row['reup_style']:>6} {row['final_d_tp_equal_mean']:>10.4f} "
            f"{row['final_d_tp_fs_mean']:>10.4f} "
            f"{row['final_path_length_fs_mean']:>10.4f}"
        )


if __name__ == "__main__":
    main()

from pathlib import Path

from .sweep import run_comparison_sweep, save_results


def main():
    rows, summary = run_comparison_sweep()
    output_dir = Path(__file__).with_name("results")
    save_results(
        output_dir,
        rows,
        summary,
        {
            "purpose": "compare equal-weight and Fubini-Study arc-length-weighted dTP",
            "note": "depth changes the actual circuit path as well as the number of snapshots",
        },
    )

    print("layers qubits fm reupload equal_dTP arc_dTP delta FS_length")
    for row in summary:
        print(
            f"{row['n_layers']:>6} {row['n_qubits']:>6} "
            f"{row['fm_style']:>4} {row['reup_style']:>8} "
            f"{row['d_tp_equal_mean']:>9.5f} {row['d_tp_arc_mean']:>8.5f} "
            f"{row['arc_minus_equal_mean']:>+8.5f} {row['path_length_fs_mean']:>9.5f}"
        )


if __name__ == "__main__":
    main()

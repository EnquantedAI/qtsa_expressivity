from pathlib import Path

from .study import run_sweep, save_results


def main():
    rows, summary, correlations = run_sweep()
    output_dir = Path(__file__).resolve().parent / "results"
    save_results(
        output_dir,
        rows,
        summary,
        correlations,
        {
            "layers": [1, 2, 3, 4],
            "qubits": [1, 2, 3],
            "reupload_axes": [None, "Y"],
            "entangling": [False, True],
            "samples": 4,
            "dataset_size": 5,
            "seed": 2026,
        },
    )

    print("nq L reup ent dTP(norm) QNTK eff.rank QNTK var")
    for row in summary:
        print(
            f"{row['n_qubits']:>2} {row['n_layers']:>1} "
            f"{row['reupload_axis']:>4} {str(row['entangle']):>5} "
            f"{row['d_tp_normalized_mean_mean']:.4f} "
            f"{row['qntk_effective_rank_mean']:.4f} "
            f"{row['qntk_element_variance_mean']:.6g}"
        )


if __name__ == "__main__":
    main()

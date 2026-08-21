from pathlib import Path

from .cross_metric import (
    save_cross_metric_results,
    summarize_cross_metric,
    topology_cross_metric_study,
)


def main():
    rows = topology_cross_metric_study()
    summary = summarize_cross_metric(rows)
    output_dir = Path(__file__).resolve().parent / "results"
    save_cross_metric_results(
        output_dir,
        rows,
        summary,
        {
            "purpose": "matched dTP/QFIM/QNTK comparison across entanglement graph topologies",
            "qntk_output": "final-state expectation value <Z_0>",
            "interpretation": "empirical diagnostic; no monotonic topology-metric relation is assumed",
        },
    )

    print("topology       edges   dTP mean   QFIM rank   QNTK rank   QNTK eff.rank")
    for row in summary:
        print(
            f"{row['topology']:<13} {row['n_edges']:>5d}   "
            f"{row['d_tp_mean']:>8.5f}   {row['qfim_rank_mean']:>9.3f}   "
            f"{row['qntk_rank_mean']:>9.3f}   {row['qntk_effective_rank_mean']:>13.5f}"
        )


if __name__ == "__main__":
    main()

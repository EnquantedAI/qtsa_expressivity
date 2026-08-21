from pathlib import Path

from .study import save_results, summarize, topology_study


def main():
    rows = topology_study()
    summary = summarize(rows)
    output_dir = Path(__file__).resolve().parent / "results"
    save_results(
        output_dir,
        rows,
        summary,
        {"purpose": "matched dTP comparison across entanglement graph topologies"},
    )

    print("topology       edges  density   algebraic_conn   dTP mean")
    for row in summary:
        print(
            f"{row['topology']:<13} {row['n_edges']:>5d}  "
            f"{row['density']:>7.3f}   {row['algebraic_connectivity']:>14.6f}   "
            f"{row['d_tp_mean']:>8.5f}"
        )


if __name__ == "__main__":
    main()

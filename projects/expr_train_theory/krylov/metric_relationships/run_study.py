from pathlib import Path

from .analysis import CorrelationConfig, run_correlation_study
from ..circuit_ensemble.study import EnsembleConfig


def main() -> None:
    output = Path(__file__).resolve().parent / "results"
    config = CorrelationConfig(
        ensemble=EnsembleConfig(
            n_qubits=3,
            depths=(1, 2, 3, 4, 5),
            topologies=("none", "linear", "ring", "full"),
            samples=20,
            seed=2026,
        )
    )
    _, global_rows, _ = run_correlation_study(config, output)
    print(f"Saved results to {output}")
    print("\nLargest absolute global Pearson correlations:")
    valid = [row for row in global_rows if row["pearson"] == row["pearson"]]
    for row in sorted(valid, key=lambda item: abs(float(item["pearson"])), reverse=True)[:8]:
        print(
            f"{row['metric_a']} vs {row['metric_b']}: "
            f"Pearson={float(row['pearson']): .3f}, Spearman={float(row['spearman']): .3f}"
        )


if __name__ == "__main__":
    main()

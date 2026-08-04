from pathlib import Path

from .study import EnsembleConfig, run_ensemble_study, summarise_records


def main() -> None:
    config = EnsembleConfig(
        n_qubits=3,
        depths=(1, 2, 3, 4, 5),
        topologies=("none", "linear", "ring", "full"),
        samples=10,
        seed=2026,
    )
    output = Path(__file__).resolve().parent / "results"
    records = run_ensemble_study(config, output)
    print("depth topology layer_dim cycle_dim full_layer final_entropy projector_dist")
    for row in summarise_records(records):
        print(
            f"{row['depth']:>5} {row['topology']:<8} "
            f"{row['layer_orbit_dimension_mean']:.2f} "
            f"{row['cycle_dimension_mean']:.2f} "
            f"{row['layer_full_fraction']:.2f} "
            f"{row['final_entropy_mean']:.3f} "
            f"{row['projector_distance_mean']:.4f}"
        )
    print(f"\nSaved to {output}")


if __name__ == "__main__":
    main()

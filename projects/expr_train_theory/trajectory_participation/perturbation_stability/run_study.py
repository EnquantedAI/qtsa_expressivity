from pathlib import Path

import numpy as np

from ..snapshots import trajectory_snapshots
from .study import (
    parameter_perturbation_study,
    save_results,
    snapshot_perturbation_study,
    summarise_rows,
)


def _print_summary(title, rows):
    print(title)
    for row in rows:
        print(
            f"eps={row['scale']:<7g}  "
            f"dTP={row['mean_d_tp']:.6f} +/- {row['std_d_tp']:.6f}  "
            f"|delta|={row['mean_abs_delta_d_tp']:.6f}  "
            f"infidelity={row['mean_infidelity']:.6e}"
        )


def main():
    rng = np.random.default_rng(17)
    n_qubits = 2
    n_layers = 4
    features = np.array([0.35, -0.6])
    parameters = rng.normal(scale=0.7, size=(n_layers, n_qubits, 3))
    scales = (0.0, 1e-4, 1e-3, 1e-2, 5e-2)

    states = trajectory_snapshots(features, parameters, n_qubits=n_qubits)

    state_rows = snapshot_perturbation_study(states, scales=scales, repeats=20, seed=11)
    state_summary = summarise_rows(state_rows)

    parameter_rows = parameter_perturbation_study(
        features,
        parameters,
        n_qubits=n_qubits,
        scales=scales,
        repeats=20,
        seed=11,
    )
    parameter_summary = summarise_rows(parameter_rows)

    output_dir = Path(__file__).resolve().parent / "results"
    save_results(
        output_dir,
        "snapshot_perturbation",
        state_rows,
        state_summary,
        {"n_qubits": n_qubits, "n_layers": n_layers, "repeats": 20, "seed": 11},
    )
    save_results(
        output_dir,
        "parameter_perturbation",
        parameter_rows,
        parameter_summary,
        {"n_qubits": n_qubits, "n_layers": n_layers, "repeats": 20, "seed": 11},
    )

    _print_summary("Snapshot perturbations", state_summary)
    print()
    _print_summary("Parameter perturbations", parameter_summary)


if __name__ == "__main__":
    main()

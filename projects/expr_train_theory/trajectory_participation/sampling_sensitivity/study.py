import csv
import json
from pathlib import Path

import numpy as np

from ..core import trajectory_participation_dimension
from ..sampling import (
    qubit_arc_states,
    trapezoidal_snapshot_weights,
    weighted_trajectory_participation_dimension,
)


def sampling_density_study(sample_counts=(3, 5, 9, 17, 33), *, start=0.0, stop=np.pi / 2):
    rows = []
    for count in sample_counts:
        parameters = np.linspace(start, stop, int(count))
        states = qubit_arc_states(parameters)
        equal = trajectory_participation_dimension(states).dimension
        weights = trapezoidal_snapshot_weights(parameters)
        weighted = weighted_trajectory_participation_dimension(states, weights)
        rows.append(
            {
                "sample_count": int(count),
                "d_tp_equal": float(equal),
                "d_tp_trapezoidal": float(weighted),
                "difference": float(equal - weighted),
            }
        )
    return rows


def duplication_study(*, duplicate_count=8):
    states = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
    base = trajectory_participation_dimension(states).dimension

    duplicated = np.vstack([np.repeat(states[[0]], duplicate_count, axis=0), states[[1]]])
    equal = trajectory_participation_dimension(duplicated).dimension

    weights = np.concatenate(
        [np.full(duplicate_count, 0.5 / duplicate_count), np.array([0.5])]
    )
    weighted = weighted_trajectory_participation_dimension(duplicated, weights)

    return {
        "base": float(base),
        "duplicated_equal": float(equal),
        "duplicated_weighted": float(weighted),
        "duplicate_count": int(duplicate_count),
    }


def save_results(output_dir, density_rows, duplication):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "sampling_density.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(density_rows[0].keys()))
        writer.writeheader()
        writer.writerows(density_rows)

    with (output_dir / "duplication_check.json").open("w", encoding="utf-8") as handle:
        json.dump(duplication, handle, indent=2)

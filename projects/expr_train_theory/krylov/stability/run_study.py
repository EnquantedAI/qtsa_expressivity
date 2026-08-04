from pathlib import Path

import numpy as np

from ..models import basis_state, two_qubit_ising
from .study import run_stability_study, save_stability_study


def main() -> None:
    output = Path(__file__).resolve().parent / "results"
    rows, summaries, settings = run_stability_study(
        two_qubit_ising(),
        basis_state(0, 4),
        perturbation_scales=(0.0, 1e-4, 1e-3, 1e-2, 5e-2),
        repeats=20,
        times=np.linspace(0.0, 4.0, 41),
        seed=2026,
    )
    paths = save_stability_study(output, rows, summaries, settings)
    for row in summaries:
        print(
            f"scale={row['perturbation_scale']:.4g} "
            f"dim={row['krylov_dimension_mean']:.3f}±{row['krylov_dimension_std']:.3f} "
            f"max C_K={row['max_spread_complexity_mean']:.4f}±"
            f"{row['max_spread_complexity_std']:.4f}"
        )
    print("saved:")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()

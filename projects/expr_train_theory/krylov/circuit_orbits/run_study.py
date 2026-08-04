from pathlib import Path

import numpy as np

from .circuits import layered_ansatz
from .study import run_parameter_study


def main() -> None:
    rng = np.random.default_rng(2026)
    n_qubits = 2
    initial = np.zeros(2**n_qubits, dtype=np.complex128)
    initial[0] = 1.0
    parameter_sets = [rng.uniform(-np.pi, np.pi, size=(depth, n_qubits, 3)) for depth in (1, 2, 3, 4)]
    output = Path(__file__).resolve().parent / "results"
    results = run_parameter_study(
        parameter_sets=parameter_sets,
        n_qubits=n_qubits,
        initial_state=initial,
        layer_builder=layered_ansatz,
        output_directory=output,
    )
    for result in results:
        print(
            f"layers={result.layer_count:2d}  "
            f"layer-span={result.layer_orbit_dimension}  "
            f"cycle-Krylov={result.repeated_cycle_dimension}  "
            f"projector-distance={result.projector_distance:.6f}"
        )


if __name__ == "__main__":
    main()

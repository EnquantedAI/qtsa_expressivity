from pathlib import Path

import numpy as np

from ..models import path_hamiltonian
from .study import run_time_step_study


def main() -> None:
    hamiltonian = path_hamiltonian(4)
    initial_state = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)
    time_steps = [0.05, 0.2, 0.5, 1.0, 2.0, 2 * np.pi]
    output = Path(__file__).resolve().parent / "results"

    results = run_time_step_study(
        hamiltonian=hamiltonian,
        initial_state=initial_state,
        time_steps=time_steps,
        output_directory=output,
    )

    print("dt      dim(H)  dim(U)  projector distance  max angle")
    for result in results:
        print(
            f"{result.time_step:6.3f}  "
            f"{result.hamiltonian_dimension:6d}  "
            f"{result.unitary_dimension:6d}  "
            f"{result.projector_distance:18.6e}  "
            f"{result.max_principal_angle:9.3e}"
        )


if __name__ == "__main__":
    main()

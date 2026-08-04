"""Print QFIM values for the analytical test cases."""

from __future__ import annotations

import numpy as np

from .core import compute_pure_state_qfim, diagnose_qfim
from .validation_models import global_phase_state, product_ry_state, redundant_ry_state, ry_state


def main() -> None:
    examples = [
        ("Single RY rotation", ry_state, np.array([0.37])),
        ("Redundant RY parameters", redundant_ry_state, np.array([0.2, -0.6])),
        ("Independent product rotations", product_ry_state, np.array([0.2, 1.1])),
        ("Pure global phase", global_phase_state, np.array([0.8])),
    ]

    for name, state_function, parameters in examples:
        qfim = compute_pure_state_qfim(state_function, parameters)
        diagnostics = diagnose_qfim(qfim)
        print(f"\n{name}")
        print("QFIM:")
        print(np.array2string(qfim, precision=8, suppress_small=True))
        print(f"eigenvalues: {diagnostics.eigenvalues}")
        print(f"rank: {diagnostics.numerical_rank}/{len(parameters)}")
        print(f"relative rank: {diagnostics.relative_rank:.3f}")


if __name__ == "__main__":
    main()

"""Print a few small examples used in the Effective Dimension review."""

import numpy as np

from projects.expr_train_theory.qfim.core import compute_pure_state_qfim

from .fisher_examples import (
    classical_fisher,
    computational_basis_probabilities,
    phase_state,
    ry_state,
)


def _basis_probabilities(state_function):
    return lambda theta: computational_basis_probabilities(state_function(theta))


def main() -> None:
    theta = np.array([0.63])

    ry_qfim = compute_pure_state_qfim(ry_state, theta)
    ry_cfim = classical_fisher(_basis_probabilities(ry_state), theta)

    phase_qfim = compute_pure_state_qfim(phase_state, theta)
    phase_cfim = classical_fisher(_basis_probabilities(phase_state), theta)

    boundary = np.array([0.0])
    near_boundary = np.array([1e-4])
    boundary_cfim = classical_fisher(_basis_probabilities(ry_state), boundary)
    nearby_cfim = classical_fisher(_basis_probabilities(ry_state), near_boundary)

    print("RY with computational-basis measurement")
    print("QFIM:", ry_qfim)
    print("CFIM:", ry_cfim)
    print()

    print("RZ(theta)|+> with computational-basis measurement")
    print("QFIM:", phase_qfim)
    print("CFIM:", phase_cfim)
    print()

    print("RY at a zero-probability boundary")
    print("CFIM at theta = 0:", boundary_cfim)
    print("CFIM near theta = 0:", nearby_cfim)


if __name__ == "__main__":
    main()

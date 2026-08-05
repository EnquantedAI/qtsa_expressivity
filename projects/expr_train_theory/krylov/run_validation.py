import numpy as np

from .core import arnoldi, lanczos
from .dynamics import exact_state, projected_state
from .metrics import (
    krylov_entropy,
    participation_ratio,
    spread_complexity,
    state_probabilities,
)
from .models import SIGMA_X, basis_state, path_hamiltonian, two_qubit_ising


def _print_lanczos_example(name: str, hamiltonian: np.ndarray, state: np.ndarray) -> None:
    result = lanczos(hamiltonian, state)
    orthogonality_error = np.linalg.norm(
        result.basis.conj().T @ result.basis - np.eye(result.dimension)
    )
    projection_error = np.linalg.norm(
        result.basis.conj().T @ hamiltonian @ result.basis - result.tridiagonal
    )
    print(f"\n{name}")
    print(f"  Krylov dimension: {result.dimension}")
    print(f"  alpha: {np.round(result.alpha, 8)}")
    print(f"  beta:  {np.round(result.beta, 8)}")
    print(f"  basis orthogonality error: {orthogonality_error:.3e}")
    print(f"  projected-H error:         {projection_error:.3e}")


if __name__ == "__main__":
    _print_lanczos_example("Pauli X from |0>", SIGMA_X, basis_state(0, 2))
    _print_lanczos_example("Four-site path from the left endpoint", path_hamiltonian(4), basis_state(0, 4))

    hamiltonian = two_qubit_ising()
    initial_state = basis_state(0, 4)
    result = lanczos(hamiltonian, initial_state)

    print("\nTwo-qubit example")
    for time in (0.0, 0.25, 0.5, 1.0):
        exact = exact_state(hamiltonian, initial_state, time)
        projected = projected_state(result, time)
        probabilities = state_probabilities(exact, result.basis)
        print(
            f"  t={time:>4.2f}  "
            f"error={np.linalg.norm(exact - projected):.3e}  "
            f"C_K={spread_complexity(probabilities):.6f}  "
            f"S_K={krylov_entropy(probabilities):.6f}  "
            f"PR={participation_ratio(probabilities):.6f}"
        )

    phase_gate = np.diag([1.0, 1j])
    arnoldi_result = arnoldi(phase_gate, basis_state(0, 2))
    print("\nArnoldi check for a non-Hermitian/unitary operator")
    print(f"  Krylov dimension: {arnoldi_result.dimension}")

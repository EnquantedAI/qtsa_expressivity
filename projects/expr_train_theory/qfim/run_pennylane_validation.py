"""Small PennyLane smoke test for the shared-architecture QFIM adapter."""

from __future__ import annotations

import numpy as np

from .core import compute_pure_state_qfim, diagnose_qfim
from .pennylane_adapter import build_shared_architecture_state_model


def main() -> None:
    model = build_shared_architecture_state_model(
        [0.2, 0.7],
        n_layers=1,
        feature_map="zzfm",
    )
    rng = np.random.default_rng(2026)
    theta = rng.uniform(-np.pi, np.pi, size=model.parameter_count)

    state = model.state_function(theta)
    qfim = compute_pure_state_qfim(model.state_function, theta)
    diagnostics = diagnose_qfim(qfim)

    print("Shared-architecture PennyLane QFIM validation")
    print(f"qubits: {model.n_qubits}")
    print(f"layers: {model.n_layers}")
    print(f"weight shape: {model.weight_shape}")
    print(f"parameter count: {model.parameter_count}")
    print(f"state dimension: {state.size}")
    print(f"state norm: {np.linalg.norm(state):.12f}")
    print(f"QFIM shape: {qfim.shape}")
    print(f"numerical rank: {diagnostics.numerical_rank}")
    print(f"relative rank: {diagnostics.relative_rank:.6f}")
    print(f"minimum eigenvalue: {diagnostics.minimum_eigenvalue:.6e}")
    print(f"symmetry error: {diagnostics.symmetry_error:.6e}")


if __name__ == "__main__":
    main()

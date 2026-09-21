import numpy as np

from .core import trajectory_participation_dimension
from .shared_qnn import native_shared_qnn_snapshots


def main():
    rng = np.random.default_rng(2026)
    n_qubits = 3
    n_layers = 4
    inputs = np.array([0.2, 0.7, 1.1])
    weights = rng.uniform(-np.pi, np.pi, size=(n_layers, n_qubits, 3))

    states = native_shared_qnn_snapshots(
        inputs,
        weights,
        n_qubits=n_qubits,
        fm_style="zzfm",
        reup_style="Y",
    )
    result = trajectory_participation_dimension(states)
    print(f"snapshots={states.shape[0]}")
    print(f"hilbert_dim={states.shape[1]}")
    print(f"d_TP={result.dimension:.6f}")


if __name__ == "__main__":
    main()

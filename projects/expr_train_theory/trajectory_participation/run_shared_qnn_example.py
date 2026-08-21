import numpy as np

from .core import trajectory_participation_dimension
from .shared_qnn import shared_qnn_snapshots


def main():
    rng = np.random.default_rng(2026)
    n_qubits = 3
    n_layers = 4
    inputs = np.array([0.2, 0.7, 1.1])
    weights = rng.uniform(-np.pi, np.pi, size=(n_layers, n_qubits, 3))

    for reup_style in (None, "Y"):
        states = shared_qnn_snapshots(
            inputs,
            weights,
            n_qubits=n_qubits,
            fm_style="zzfm",
            reup_style=reup_style,
        )
        result = trajectory_participation_dimension(states)
        limit = min(states.shape)
        print(
            f"reupload={str(reup_style):>4}  "
            f"snapshots={states.shape[0]}  "
            f"d_TP={result.dimension:.6f}  "
            f"d_TP/limit={result.dimension / limit:.6f}"
        )


if __name__ == "__main__":
    main()

import numpy as np

from .study import analyse_toy_qnn_trajectory


def main():
    rng = np.random.default_rng(2026)
    features = np.array([0.4, -0.7])

    print("depth  dTP(no reupload)  dTP(reupload Y)")
    for depth in range(1, 6):
        parameters = rng.uniform(-np.pi, np.pi, size=(depth, 2, 3))
        plain = analyse_toy_qnn_trajectory(
            features,
            parameters,
            n_qubits=2,
            reupload_axis=None,
        )
        reuploaded = analyse_toy_qnn_trajectory(
            features,
            parameters,
            n_qubits=2,
            reupload_axis="Y",
        )
        print(f"{depth:5d}  {plain.d_tp:16.6f}  {reuploaded.d_tp:15.6f}")


if __name__ == "__main__":
    main()

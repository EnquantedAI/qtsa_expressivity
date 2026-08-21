import numpy as np

from projects.expr_train_theory.trajectory_participation import (
    trajectory_participation_dimension,
)
from projects.expr_train_theory.trajectory_participation.gram_backend import (
    analyse_trajectory_from_gram,
    gram_matrix_from_states,
)


def main():
    states = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=complex,
    )

    svd_result = trajectory_participation_dimension(states)
    gram = gram_matrix_from_states(states)
    gram_result = analyse_trajectory_from_gram(gram)

    print(f"state matrix shape:   {states.T.shape}")
    print(f"Gram matrix shape:    {gram.shape}")
    print(f"dTP from SVD:         {svd_result.dimension:.12f}")
    print(f"dTP from Gram matrix: {gram_result.dimension:.12f}")
    print(f"rank from SVD:        {svd_result.numerical_rank}")
    print(f"rank from Gram:       {gram_result.numerical_rank}")


if __name__ == "__main__":
    main()

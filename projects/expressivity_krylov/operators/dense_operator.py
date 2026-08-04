from __future__ import annotations

import numpy as np

from .symmetric_operator import SymmetricOperator


class DenseOperator(SymmetricOperator):
    """
    Wrapper of np.ndarray implementing SymmetricOperator interface.

    Used for testing the Krylov methods.
    """

    def __init__(
        self,
        matrix: np.ndarray,
    ):
        self.matrix = np.asarray(matrix)

        if matrix.ndim != 2:
            raise ValueError("matrix must be two-dimensional")

        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("matrix must be square")

        if not np.allclose(matrix, matrix.T):
            raise ValueError("matrix must be symmetric")

        super().__init__(self.matrix.shape)


    def _matvec(self, x):
        return self.matrix @ x


    def __matmul__(self, x):
        return self.matrix @ x

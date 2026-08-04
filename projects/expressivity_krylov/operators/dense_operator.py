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
        matrix = np.asarray(matrix)

        if matrix.ndim != 2:
            raise ValueError("matrix must be two-dimensional")

        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("matrix must be square")

        if check_symmetry:
            if not np.allclose(matrix, matrix.T):
                raise ValueError("matrix must be symmetric")

        self._matrix = matrix

    @property
    def shape(self):
        return self._matrix.shape

    @property
    def matrix(self):
        return self._matrix

    def matvec(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        return self._matrix @ x

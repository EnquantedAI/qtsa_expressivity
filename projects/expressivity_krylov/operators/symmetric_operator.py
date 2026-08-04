from __future__ import annotations
from abc import ABC, abstractmethod

from scipy.sparse.linalg import LinearOperator

import numpy as np

class SymmetricOperator(LinearOperator):
    """
    Abstract symmetric linear operator. Symmetry is required by Krylov methods.
    
    The operator is represented implicitely via matrix-vector multiplication method,
    which is all that's needed for the Krylov methods.

    This bypasses the need for calculating the explicit matrix, 
    (e.g. of the quantum Neural Tangent Kernel)
    """

    def __init__(self, dtype=np.float64):
        super().__init__(dtype=dtype, shape=self.shape)

    @property
    @abstractmethod
    def shape(self) -> tuple[int, int]:
        """Shape of the operator (N, N)."""
        pass

    @abstractmethod
    def matvec(self, x: np.ndarray) -> np.ndarray:
        """Compute y = A x."""
        pass

    def _matvec(self, x: np.ndarray) -> np.ndarray:
        return self.matvec(x)

    def __matmul__(self, x: np.ndarray) -> np.ndarray:
        return self.matvec(x)

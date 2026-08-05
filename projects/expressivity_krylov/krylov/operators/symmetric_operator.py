from __future__ import annotations
from abc import ABC, abstractmethod

from scipy.sparse.linalg import LinearOperator

from pennylane import numpy as np

class SymmetricOperator(LinearOperator, ABC):
    """
    Abstract symmetric linear operator. Symmetry is required by Krylov methods.
    
    The operator is represented implicitely via matrix-vector multiplication method,
    which is all that's needed for the Krylov methods.

    This bypasses the need for calculating the explicit matrix, 
    (e.g. of the quantum Neural Tangent Kernel)
    """

    def __init__(self, shape):
        super().__init__(dtype=np.dtype(np.float64), shape=shape)


    @abstractmethod
    def _matvec(self, x: np.ndarray) -> np.ndarray:
        return self.matvec(x)

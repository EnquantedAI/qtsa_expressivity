from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class JacobianOracle(ABC):
    """
    Represents the Jacobian

        J = df / dθ

    as an implicit linear operator.

    Rows correspond to dataset samples.
    Columns correspond to trainable parameters.
    """

    @property
    @abstractmethod
    def input_dimension(self) -> int:
        """
        Number of trainable parameters.
        """

    @property
    @abstractmethod
    def output_dimension(self) -> int:
        """
        Number of outputs (typically the number of samples).
        """

    @abstractmethod
    def jvp(
        self,
        v: np.ndarray,
    ) -> np.ndarray:
        """
        Jacobian-vector product

            J v

        Parameters
        ----------
        v
            Shape (n_parameters,)
        """

    @abstractmethod
    def vjp(
        self,
        v: np.ndarray,
    ) -> np.ndarray:
        """
        Vector-Jacobian product

            J^T v

        Parameters
        ----------
        v
            Shape (n_outputs,)
        """

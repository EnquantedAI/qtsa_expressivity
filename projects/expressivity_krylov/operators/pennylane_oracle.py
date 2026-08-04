from __future__ import annotations

import numpy as np
import pennylane as qml

from .oracle import JacobianOracle


class PennyLaneJacobianOracle(JacobianOracle):
    """
    Reference implementation based on explicitly constructing
    the Jacobian with PennyLane.

    This implementation is intended as a correctness reference.
    It can later be replaced by an implicit JVP/VJP backend
    without changing any downstream code.
    """

    def __init__(
        self,
        model,
        weights,
        X,
    ):
        self.model = model
        self.weights = weights
        self.X = X

        self._jacobian = self._compute_jacobian()

    @property
    def input_dimension(self):
        return self._jacobian.shape[1]

    @property
    def output_dimension(self):
        return self._jacobian.shape[0]

    def jvp(self, v):
        return self._jacobian @ v

    def vjp(self, v):
        return self._jacobian.T @ v

    def _compute_jacobian(self):
        """
        Returns

            J

        of shape

            (n_samples,
             n_parameters)
        """

        flat_weights = self.weights.ravel()

        def predict(flat):
            w = flat.reshape(self.weights.shape)

            return np.stack([
                self.model(x, w)[0]
                for x in self.X
            ])

        jacobian = qml.jacobian(predict)(flat_weights)

        return np.asarray(jacobian)

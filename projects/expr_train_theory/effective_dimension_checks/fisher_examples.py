"""Simple CFIM/QFIM examples which do not depend on PennyLane."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

Array = np.ndarray


def ry_state(parameters: Array) -> Array:
    """Return RY(theta)|0>."""
    theta = float(np.asarray(parameters, dtype=float).reshape(-1)[0])
    return np.array(
        [np.cos(theta / 2.0), np.sin(theta / 2.0)],
        dtype=np.complex128,
    )


def phase_state(parameters: Array) -> Array:
    """Return RZ(theta)|+>, up to the usual global convention."""
    theta = float(np.asarray(parameters, dtype=float).reshape(-1)[0])
    return np.array(
        [np.exp(-0.5j * theta), np.exp(0.5j * theta)],
        dtype=np.complex128,
    ) / np.sqrt(2.0)


def computational_basis_probabilities(state: Array) -> Array:
    state = np.asarray(state, dtype=np.complex128).reshape(-1)
    return np.abs(state) ** 2


def classical_fisher(
    probability_function: Callable[[Array], Array],
    parameters: Array,
    *,
    step: float = 1e-6,
    min_probability: float = 1e-12,
) -> Array:
    """Calculate a classical Fisher matrix with central differences."""
    theta = np.asarray(parameters, dtype=float).reshape(-1)
    probabilities = np.asarray(probability_function(theta), dtype=float).reshape(-1)

    if np.any(probabilities < -1e-12):
        raise ValueError("Probabilities must be non-negative.")
    if not np.isclose(probabilities.sum(), 1.0, atol=1e-9):
        raise ValueError("Probabilities must sum to one.")

    jacobian = np.zeros((probabilities.size, theta.size), dtype=float)
    for index in range(theta.size):
        direction = np.zeros_like(theta)
        direction[index] = step
        plus = np.asarray(probability_function(theta + direction), dtype=float).reshape(-1)
        minus = np.asarray(probability_function(theta - direction), dtype=float).reshape(-1)
        jacobian[:, index] = (plus - minus) / (2.0 * step)

    fisher = np.zeros((theta.size, theta.size), dtype=float)
    for probability, gradient in zip(probabilities, jacobian, strict=True):
        if probability > min_probability:
            fisher += np.outer(gradient, gradient) / probability

    return 0.5 * (fisher + fisher.T)

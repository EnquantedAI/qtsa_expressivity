"""Small states used to check the QFIM implementation."""

from __future__ import annotations

import numpy as np


def ry_state(parameters: np.ndarray) -> np.ndarray:
    """One-qubit state RY(theta)|0>; exact QFIM is [[1]]."""

    theta = float(np.asarray(parameters)[0])
    return np.array([np.cos(theta / 2.0), np.sin(theta / 2.0)], dtype=np.complex128)


def redundant_ry_state(parameters: np.ndarray) -> np.ndarray:
    """RY(theta_1 + theta_2)|0>; exact QFIM is a rank-one matrix of ones."""

    theta = np.asarray(parameters, dtype=float)
    total = float(theta[0] + theta[1])
    return np.array([np.cos(total / 2.0), np.sin(total / 2.0)], dtype=np.complex128)


def product_ry_state(parameters: np.ndarray) -> np.ndarray:
    """RY(theta_1)|0> tensor RY(theta_2)|0>; exact QFIM is I_2."""

    theta = np.asarray(parameters, dtype=float)
    first = np.array([np.cos(theta[0] / 2.0), np.sin(theta[0] / 2.0)])
    second = np.array([np.cos(theta[1] / 2.0), np.sin(theta[1] / 2.0)])
    return np.kron(first, second).astype(np.complex128)


def global_phase_state(parameters: np.ndarray) -> np.ndarray:
    """e^(i theta)|0>; physical state is constant, hence exact QFIM is zero."""

    theta = float(np.asarray(parameters)[0])
    return np.array([np.exp(1j * theta), 0.0], dtype=np.complex128)

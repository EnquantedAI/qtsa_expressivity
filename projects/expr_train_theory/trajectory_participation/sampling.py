import numpy as np

from .core import _as_state_matrix


def _normalize_weights(weights, count):
    if weights is None:
        return np.full(count, 1.0 / count, dtype=float)

    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1 or weights.size != count:
        raise ValueError("weights must contain one value per snapshot")
    if np.any(weights < 0.0):
        raise ValueError("weights must be non-negative")

    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError("weights must have positive total weight")
    return weights / total


def weighted_trajectory_spectrum(states, weights=None, *, tol=1e-12):
    """Spectrum of the weighted trajectory density matrix."""
    psi = _as_state_matrix(states)
    probabilities = _normalize_weights(weights, psi.shape[1])
    weighted = psi * np.sqrt(probabilities)[None, :]
    singular_values = np.linalg.svd(weighted, compute_uv=False)
    spectrum = singular_values**2
    spectrum = spectrum[spectrum > tol]
    spectrum = spectrum / np.sum(spectrum)
    return spectrum


def weighted_trajectory_participation_dimension(states, weights=None):
    spectrum = weighted_trajectory_spectrum(states, weights)
    return float(1.0 / np.sum(spectrum**2))


def trapezoidal_snapshot_weights(parameters):
    """Quadrature weights for snapshots sampled along a 1D trajectory parameter."""
    x = np.asarray(parameters, dtype=float)
    if x.ndim != 1 or x.size < 1:
        raise ValueError("parameters must be a non-empty 1D array")
    if x.size == 1:
        return np.array([1.0])
    if np.any(np.diff(x) <= 0.0):
        raise ValueError("parameters must be strictly increasing")

    weights = np.empty_like(x)
    weights[0] = 0.5 * (x[1] - x[0])
    weights[-1] = 0.5 * (x[-1] - x[-2])
    if x.size > 2:
        weights[1:-1] = 0.5 * (x[2:] - x[:-2])
    return weights / np.sum(weights)


def qubit_arc_states(parameters):
    """Simple two-dimensional path cos(t)|0> + sin(t)|1>."""
    x = np.asarray(parameters, dtype=float)
    return np.column_stack((np.cos(x), np.sin(x))).astype(complex)

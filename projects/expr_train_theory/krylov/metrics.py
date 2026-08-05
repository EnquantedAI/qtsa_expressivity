import numpy as np


def state_probabilities(
    state: np.ndarray,
    basis: np.ndarray,
    tolerance: float = 1e-12,
) -> np.ndarray:
    """Return probabilities of a state in an orthonormal Krylov basis."""
    vector = np.asarray(state, dtype=np.complex128).reshape(-1)
    basis_matrix = np.asarray(basis, dtype=np.complex128)
    if basis_matrix.ndim != 2 or basis_matrix.shape[0] != vector.size:
        raise ValueError("basis and state dimensions do not match")

    gram = basis_matrix.conj().T @ basis_matrix
    if not np.allclose(gram, np.eye(gram.shape[0]), atol=100 * tolerance, rtol=0.0):
        raise ValueError("basis must be orthonormal")

    amplitudes = basis_matrix.conj().T @ vector
    probabilities = np.abs(amplitudes) ** 2
    probabilities[probabilities < tolerance] = 0.0
    total = probabilities.sum()
    if total > 1.0 + 100 * tolerance:
        raise ValueError("probabilities exceed one")
    return probabilities


def spread_complexity(probabilities: np.ndarray, start_index: int = 0) -> float:
    values = _valid_probabilities(probabilities)
    indices = np.arange(start_index, start_index + values.size, dtype=float)
    return float(indices @ values)


def krylov_entropy(probabilities: np.ndarray, base: float = np.e) -> float:
    values = _valid_probabilities(probabilities)
    positive = values[values > 0]
    if positive.size == 0:
        return 0.0
    if base <= 0 or np.isclose(base, 1.0):
        raise ValueError("entropy base must be positive and different from one")
    return float(-np.sum(positive * np.log(positive)) / np.log(base))


def participation_ratio(probabilities: np.ndarray) -> float:
    values = _valid_probabilities(probabilities)
    denominator = float(np.sum(values**2))
    return 0.0 if denominator == 0.0 else 1.0 / denominator


def _valid_probabilities(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("probabilities cannot be empty")
    if not np.all(np.isfinite(values)) or np.any(values < -1e-12):
        raise ValueError("invalid probabilities")
    values = np.clip(values, 0.0, None)
    if values.sum() > 1.0 + 1e-9:
        raise ValueError("probabilities cannot sum to more than one")
    return values

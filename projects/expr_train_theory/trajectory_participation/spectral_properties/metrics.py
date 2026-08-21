from dataclasses import dataclass

import numpy as np

from ..core import trajectory_spectrum
from ..sampling import weighted_trajectory_spectrum


@dataclass(frozen=True)
class SpectralDimensionProfile:
    numerical_rank: int
    stable_rank: float
    participation_dimension: float
    entropy_dimension: float
    largest_weight: float
    renyi_2_entropy: float
    shannon_entropy: float


def _as_probability_spectrum(spectrum, *, tol=1e-15):
    values = np.asarray(spectrum, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("spectrum must be a non-empty 1D array")
    if np.any(values < -tol):
        raise ValueError("spectrum entries must be non-negative")

    values = np.maximum(values, 0.0)
    total = float(np.sum(values))
    if total <= tol:
        raise ValueError("spectrum must have positive total weight")

    values = values / total
    return values[values > tol]


def renyi_entropy(spectrum, order):
    """Renyi entropy of a normalized non-negative spectrum."""
    p = _as_probability_spectrum(spectrum)
    order = float(order)
    if order <= 0.0:
        raise ValueError("Renyi order must be positive")
    if np.isclose(order, 1.0):
        return float(-np.sum(p * np.log(p)))
    if np.isinf(order):
        return float(-np.log(np.max(p)))
    return float(np.log(np.sum(p**order)) / (1.0 - order))


def renyi_effective_dimension(spectrum, order):
    """Effective dimension exp(H_order) for a spectral probability vector."""
    return float(np.exp(renyi_entropy(spectrum, order)))


def spectral_dimension_profile(states, weights=None, *, rank_tol=1e-10):
    """Compare several soft dimensions obtained from the same trajectory spectrum."""
    if weights is None:
        spectrum, singular_values = trajectory_spectrum(states)
        numerical_rank = int(np.count_nonzero(singular_values > rank_tol))
    else:
        spectrum = weighted_trajectory_spectrum(states, weights)
        numerical_rank = int(np.count_nonzero(spectrum > rank_tol**2))

    participation = renyi_effective_dimension(spectrum, 2.0)
    entropy_dimension = renyi_effective_dimension(spectrum, 1.0)
    stable_rank = renyi_effective_dimension(spectrum, np.inf)

    return SpectralDimensionProfile(
        numerical_rank=numerical_rank,
        stable_rank=stable_rank,
        participation_dimension=participation,
        entropy_dimension=entropy_dimension,
        largest_weight=float(np.max(spectrum)),
        renyi_2_entropy=renyi_entropy(spectrum, 2.0),
        shannon_entropy=renyi_entropy(spectrum, 1.0),
    )

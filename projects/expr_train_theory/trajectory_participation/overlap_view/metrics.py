import numpy as np

from ..core import _as_state_matrix
from ..sampling import _normalize_weights


def _overlap_squared_matrix(states):
    psi = _as_state_matrix(states)
    gram = psi.conj().T @ psi
    return np.abs(gram) ** 2


def first_state_frame_potential(states):
    """First frame potential of an equal-weight set of pure-state snapshots."""
    overlaps_sq = _overlap_squared_matrix(states)
    return float(np.sum(overlaps_sq))


def weighted_state_frame_potential(states, weights=None):
    """Weighted first state-frame potential sum_kl w_k w_l |<psi_k|psi_l>|^2."""
    overlaps_sq = _overlap_squared_matrix(states)
    probabilities = _normalize_weights(weights, overlaps_sq.shape[0])
    return float(probabilities @ overlaps_sq @ probabilities)


def trajectory_density_purity_from_overlaps(states, weights=None):
    """Purity of the trajectory mixture computed only from pairwise overlaps."""
    return weighted_state_frame_potential(states, weights)


def trajectory_participation_dimension_from_overlaps(states, weights=None):
    """Participation dimension as the inverse overlap-based trajectory purity."""
    purity = trajectory_density_purity_from_overlaps(states, weights)
    if purity <= 0.0:
        raise ValueError("trajectory purity must be positive")
    return float(1.0 / purity)

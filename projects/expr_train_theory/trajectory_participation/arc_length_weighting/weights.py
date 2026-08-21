import numpy as np

from ..core import _as_state_matrix
from ..sampling import weighted_trajectory_participation_dimension


def fubini_study_distance(state_a, state_b):
    """Projective distance arccos(|<a|b>|) between two pure states."""
    a = np.asarray(state_a, dtype=complex).reshape(-1)
    b = np.asarray(state_b, dtype=complex).reshape(-1)
    if a.size == 0 or b.size == 0 or a.shape != b.shape:
        raise ValueError("states must be non-empty vectors with the same shape")

    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a <= 0.0 or norm_b <= 0.0:
        raise ValueError("states must have non-zero norm")

    overlap = np.vdot(a / norm_a, b / norm_b)
    fidelity_amplitude = float(np.clip(np.abs(overlap), 0.0, 1.0))
    return float(np.arccos(fidelity_amplitude))


def cumulative_fubini_study_length(states):
    """Cumulative projective arc length of an ordered snapshot sequence."""
    psi = _as_state_matrix(states).T
    cumulative = np.zeros(psi.shape[0], dtype=float)
    for index in range(1, psi.shape[0]):
        cumulative[index] = cumulative[index - 1] + fubini_study_distance(
            psi[index - 1], psi[index]
        )
    return cumulative


def arc_length_snapshot_weights(states, *, atol=1e-14):
    """Trapezoidal quadrature weights using Fubini-Study arc length as coordinate.

    If all snapshots represent the same projective state, the path has zero length;
    in that degenerate case equal weights are returned.
    """
    psi = _as_state_matrix(states).T
    count = psi.shape[0]
    if count == 1:
        return np.array([1.0])

    cumulative = cumulative_fubini_study_length(psi)
    segment_lengths = np.diff(cumulative)
    total_length = float(cumulative[-1])
    if total_length <= atol:
        return np.full(count, 1.0 / count, dtype=float)

    weights = np.empty(count, dtype=float)
    weights[0] = 0.5 * segment_lengths[0]
    weights[-1] = 0.5 * segment_lengths[-1]
    if count > 2:
        weights[1:-1] = 0.5 * (segment_lengths[:-1] + segment_lengths[1:])

    return weights / np.sum(weights)


def arc_length_weighted_participation_dimension(states):
    weights = arc_length_snapshot_weights(states)
    return weighted_trajectory_participation_dimension(states, weights)

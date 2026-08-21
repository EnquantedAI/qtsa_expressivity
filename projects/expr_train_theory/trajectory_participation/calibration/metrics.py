from dataclasses import dataclass

import numpy as np

from ..core import trajectory_participation_dimension


@dataclass(frozen=True)
class TrajectoryCalibration:
    dimension: float
    numerical_rank: int
    ceiling: int
    fraction_of_rank: float
    fraction_of_ceiling: float
    entropy_dimension: float
    stable_rank: float
    largest_weight: float


def calibrate_trajectory(states, *, rank_tol=1e-10):
    """Return dTP together with a few scale/reference quantities."""
    arr = np.asarray(states, dtype=complex)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError("states must be a non-empty 2D array-like object")

    result = trajectory_participation_dimension(states, rank_tol=rank_tol)
    ceiling = int(min(arr.shape[0], arr.shape[1]))
    rank = result.numerical_rank
    largest_weight = float(np.max(result.spectrum))

    return TrajectoryCalibration(
        dimension=result.dimension,
        numerical_rank=rank,
        ceiling=ceiling,
        fraction_of_rank=float(result.dimension / rank),
        fraction_of_ceiling=float(result.dimension / ceiling),
        entropy_dimension=float(np.exp(result.entropy)),
        stable_rank=float(1.0 / largest_weight),
        largest_weight=largest_weight,
    )

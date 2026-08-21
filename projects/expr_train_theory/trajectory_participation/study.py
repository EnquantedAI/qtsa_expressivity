from dataclasses import dataclass

import numpy as np

from .core import trajectory_participation_dimension
from .snapshots import trajectory_snapshots


@dataclass(frozen=True)
class TrajectoryStudyResult:
    n_qubits: int
    n_layers: int
    snapshot_count: int
    d_tp: float
    numerical_rank: int
    entropy: float
    max_dimension: int
    normalized_d_tp: float


def analyse_toy_qnn_trajectory(
    features,
    parameters,
    *,
    n_qubits,
    encoding_axis="Y",
    reupload_axis=None,
    entangle=True,
):
    snapshots = trajectory_snapshots(
        features,
        parameters,
        n_qubits=n_qubits,
        encoding_axis=encoding_axis,
        reupload_axis=reupload_axis,
        entangle=entangle,
    )
    result = trajectory_participation_dimension(snapshots)
    max_dimension = min(snapshots.shape[0], snapshots.shape[1])
    return TrajectoryStudyResult(
        n_qubits=n_qubits,
        n_layers=int(np.asarray(parameters).shape[0]),
        snapshot_count=int(snapshots.shape[0]),
        d_tp=result.dimension,
        numerical_rank=result.numerical_rank,
        entropy=result.entropy,
        max_dimension=max_dimension,
        normalized_d_tp=float(result.dimension / max_dimension),
    )

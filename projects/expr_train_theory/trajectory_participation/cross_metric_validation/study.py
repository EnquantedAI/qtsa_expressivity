from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import json

import numpy as np

from projects.expr_train_theory.effective_dimension_checks.fisher_examples import (
    classical_fisher,
    computational_basis_probabilities,
)
from projects.expr_train_theory.qfim.core import compute_pure_state_qfim, diagnose_qfim
from projects.expr_train_theory.trajectory_participation.core import (
    trajectory_participation_dimension,
)
from projects.expr_train_theory.trajectory_participation.calibration.metrics import (
    calibrate_trajectory,
)


@dataclass(frozen=True)
class CrossMetricResult:
    name: str
    parameter_count: int
    snapshot_count: int
    hilbert_dimension: int
    d_tp: float
    d_tp_rank_fraction: float
    d_tp_ceiling_fraction: float
    trajectory_rank: int
    qfim_rank: int
    qfim_relative_rank: float
    qfim_trace: float
    cfim_rank: int
    cfim_relative_rank: float
    cfim_trace: float


def _rank(matrix, *, atol=1e-9, rtol=1e-7):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.size == 0:
        return 0
    values = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))
    largest = max(0.0, float(values[-1]))
    threshold = max(atol, rtol * largest)
    return int(np.count_nonzero(values > threshold))


def compare_case(name, parameters, trajectory_function):
    """Compare trajectory and Fisher diagnostics for one small parametric model."""
    theta = np.asarray(parameters, dtype=float).reshape(-1)
    snapshots = np.asarray(trajectory_function(theta), dtype=np.complex128)
    if snapshots.ndim != 2 or snapshots.shape[0] == 0:
        raise ValueError("trajectory_function must return a non-empty 2D state array")

    final_state_function = lambda x: np.asarray(trajectory_function(x), dtype=np.complex128)[-1]
    probability_function = lambda x: computational_basis_probabilities(final_state_function(x))

    tp = trajectory_participation_dimension(snapshots)
    calibration = calibrate_trajectory(snapshots)

    qfim = compute_pure_state_qfim(final_state_function, theta)
    qdiag = diagnose_qfim(qfim)
    cfim = classical_fisher(probability_function, theta)
    crank = _rank(cfim)

    parameter_count = theta.size
    return CrossMetricResult(
        name=str(name),
        parameter_count=parameter_count,
        snapshot_count=int(snapshots.shape[0]),
        hilbert_dimension=int(snapshots.shape[1]),
        d_tp=float(tp.dimension),
        d_tp_rank_fraction=float(calibration.fraction_of_rank),
        d_tp_ceiling_fraction=float(calibration.fraction_of_ceiling),
        trajectory_rank=int(tp.numerical_rank),
        qfim_rank=int(qdiag.numerical_rank),
        qfim_relative_rank=float(qdiag.relative_rank),
        qfim_trace=float(qdiag.trace),
        cfim_rank=crank,
        cfim_relative_rank=(crank / parameter_count if parameter_count else 0.0),
        cfim_trace=float(np.trace(cfim)),
    )


def _ry(theta):
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def _rz(theta):
    return np.diag([np.exp(-0.5j * theta), np.exp(0.5j * theta)]).astype(np.complex128)


def _same_axis_trajectory(theta):
    state = np.array([1.0, 0.0], dtype=np.complex128)
    snapshots = [state.copy()]
    for value in theta:
        state = _ry(float(value)) @ state
        snapshots.append(state.copy())
    return np.asarray(snapshots)


def _phase_trajectory(theta):
    state = np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
    snapshots = [state.copy()]
    for value in theta:
        state = _rz(float(value)) @ state
        snapshots.append(state.copy())
    return np.asarray(snapshots)


def _mixed_axis_trajectory(theta):
    if len(theta) != 2:
        raise ValueError("mixed-axis reference case expects two parameters")
    state = np.array([1.0, 0.0], dtype=np.complex128)
    snapshots = [state.copy()]
    state = _ry(float(theta[0])) @ state
    snapshots.append(state.copy())
    state = _rz(float(theta[1])) @ state
    snapshots.append(state.copy())
    return np.asarray(snapshots)


def run_reference_cases(output_dir=None):
    cases = [
        ("redundant_ry_layers", np.array([0.4, -0.7]), _same_axis_trajectory),
        ("phase_only_rz", np.array([0.8]), _phase_trajectory),
        ("mixed_ry_rz", np.array([0.7, 1.1]), _mixed_axis_trajectory),
    ]
    results = [compare_case(name, theta, fn) for name, theta, fn in cases]

    if output_dir is not None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        rows = [asdict(result) for result in results]
        with (output / "cross_metric_reference.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        metadata = {
            "cases": [name for name, _, _ in cases],
            "note": "Small analytical/reference cases; not a benchmark of the shared QNN.",
        }
        (output / "cross_metric_reference_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

    return results

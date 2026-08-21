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
from projects.expr_train_theory.trajectory_participation.qntk_cross_metric.qntk import (
    compute_qntk,
    diagnose_qntk,
)


@dataclass(frozen=True)
class MetricProfile:
    name: str
    parameter_count: int
    snapshot_count: int
    d_tp: float
    trajectory_rank: int
    qfim_rank: int
    cfim_rank: int
    qntk_rank: int
    qfim_trace: float
    cfim_trace: float
    qntk_trace: float
    qntk_effective_rank: float


def _ry(angle):
    c = np.cos(angle / 2.0)
    s = np.sin(angle / 2.0)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def _rz(angle):
    return np.diag([np.exp(-0.5j * angle), np.exp(0.5j * angle)]).astype(np.complex128)


def redundant_ry_trajectory(x, theta):
    theta = np.asarray(theta, dtype=float).reshape(-1)
    if theta.size != 2:
        raise ValueError("redundant RY case expects two parameters")
    state = _ry(float(x)) @ np.array([1.0, 0.0], dtype=np.complex128)
    snapshots = [state.copy()]
    for value in theta:
        state = _ry(float(value)) @ state
        snapshots.append(state.copy())
    return np.asarray(snapshots)


def phase_only_trajectory(x, theta):
    theta = np.asarray(theta, dtype=float).reshape(-1)
    if theta.size != 1:
        raise ValueError("phase-only case expects one parameter")
    plus = np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
    state = _rz(float(x)) @ plus
    snapshots = [state.copy()]
    state = _rz(float(theta[0])) @ state
    snapshots.append(state.copy())
    return np.asarray(snapshots)


def mixed_trajectory(x, theta):
    theta = np.asarray(theta, dtype=float).reshape(-1)
    if theta.size != 2:
        raise ValueError("mixed case expects two parameters")
    state = _ry(float(x)) @ np.array([1.0, 0.0], dtype=np.complex128)
    snapshots = [state.copy()]
    state = _ry(float(theta[0])) @ state
    snapshots.append(state.copy())
    state = _rz(float(theta[1])) @ state
    snapshots.append(state.copy())
    return np.asarray(snapshots)


def z_output(trajectory_function, x, theta):
    state = np.asarray(trajectory_function(x, theta), dtype=np.complex128)[-1]
    return float(np.abs(state[0]) ** 2 - np.abs(state[1]) ** 2)


def _matrix_rank(matrix, *, atol=1e-9, rtol=1e-7):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.size == 0:
        return 0
    values = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))
    largest = max(0.0, float(values[-1]))
    threshold = max(atol, rtol * largest)
    return int(np.count_nonzero(values > threshold))


def evaluate_profile(name, trajectory_function, theta, X, *, step=1e-6):
    theta = np.asarray(theta, dtype=float).reshape(-1)
    X = np.asarray(X, dtype=float).reshape(-1)
    x_ref = float(X[len(X) // 2])

    snapshots = trajectory_function(x_ref, theta)
    tp = trajectory_participation_dimension(snapshots)

    final_state = lambda pars: trajectory_function(x_ref, pars)[-1]
    probabilities = lambda pars: computational_basis_probabilities(final_state(pars))

    qfim = compute_pure_state_qfim(final_state, theta, step=step)
    qdiag = diagnose_qfim(qfim)
    cfim = classical_fisher(probabilities, theta, step=step)

    model = lambda x, pars: z_output(trajectory_function, x, pars)
    qntk = compute_qntk(model, theta, X, step=step)
    kdiag = diagnose_qntk(qntk)

    return MetricProfile(
        name=str(name),
        parameter_count=int(theta.size),
        snapshot_count=int(snapshots.shape[0]),
        d_tp=float(tp.dimension),
        trajectory_rank=int(tp.numerical_rank),
        qfim_rank=int(qdiag.numerical_rank),
        cfim_rank=int(_matrix_rank(cfim)),
        qntk_rank=int(kdiag.rank),
        qfim_trace=float(qdiag.trace),
        cfim_trace=float(np.trace(cfim)),
        qntk_trace=float(kdiag.trace),
        qntk_effective_rank=float(kdiag.effective_rank),
    )


def default_profiles():
    X = np.linspace(-0.8, 0.8, 7)
    return [
        evaluate_profile(
            "redundant_ry",
            redundant_ry_trajectory,
            np.array([0.35, -0.55]),
            X,
        ),
        evaluate_profile(
            "phase_only",
            phase_only_trajectory,
            np.array([0.9]),
            X,
        ),
        evaluate_profile(
            "mixed_ry_rz",
            mixed_trajectory,
            np.array([0.6, 1.1]),
            X,
        ),
    ]


def save_profiles(output_dir, profiles):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = [asdict(profile) for profile in profiles]

    csv_path = output / "metric_profile.csv"
    json_path = output / "metric_profile_metadata.json"

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "cases": [profile.name for profile in profiles],
        "note": "Reference cases for checking what each metric responds to; not a shared-QNN benchmark.",
    }
    json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return csv_path, json_path

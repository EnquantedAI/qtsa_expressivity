from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path

import numpy as np

from projects.expr_train_theory.trajectory_participation.core import (
    trajectory_participation_dimension,
)
from .qntk import compute_qntk, diagnose_qntk


@dataclass(frozen=True)
class ToyCase:
    name: str
    X: np.ndarray
    theta: np.ndarray


def _ry(angle):
    c = np.cos(angle / 2.0)
    s = np.sin(angle / 2.0)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def _rz(angle):
    return np.array(
        [[np.exp(-0.5j * angle), 0.0], [0.0, np.exp(0.5j * angle)]],
        dtype=np.complex128,
    )


def toy_trajectory(x, theta):
    """Encode x with RY, then apply one RY and one RZ variational layer."""
    theta = np.asarray(theta, dtype=float).reshape(-1)
    if theta.size != 2:
        raise ValueError("toy model expects two parameters")
    state0 = np.array([1.0, 0.0], dtype=np.complex128)
    encoded = _ry(float(x)) @ state0
    after_ry = _ry(theta[0]) @ encoded
    after_rz = _rz(theta[1]) @ after_ry
    return np.stack([encoded, after_ry, after_rz])


def toy_output(x, theta):
    state = toy_trajectory(x, theta)[-1]
    return float(np.abs(state[0]) ** 2 - np.abs(state[1]) ** 2)


def evaluate_case(case, *, step=1e-6):
    X = np.asarray(case.X, dtype=float).reshape(-1)
    theta = np.asarray(case.theta, dtype=float).reshape(-1)
    kernel = compute_qntk(toy_output, theta, X, step=step)
    qdiag = diagnose_qntk(kernel)

    d_tp = []
    trajectory_rank = []
    for x in X:
        result = trajectory_participation_dimension(toy_trajectory(x, theta))
        d_tp.append(float(result.dimension))
        trajectory_rank.append(int(result.numerical_rank))

    return {
        "case": case.name,
        "n_samples": int(X.size),
        "parameter_count": int(theta.size),
        "d_tp_mean": float(np.mean(d_tp)),
        "d_tp_std": float(np.std(d_tp)),
        "trajectory_rank_mean": float(np.mean(trajectory_rank)),
        "qntk_rank": qdiag.rank,
        "qntk_trace": qdiag.trace,
        "qntk_effective_rank": qdiag.effective_rank,
        "qntk_largest_eigenvalue": qdiag.largest_eigenvalue,
        "qntk_condition": qdiag.condition_number,
        "qntk_element_variance": qdiag.element_variance,
    }


def default_cases():
    X = np.linspace(-1.0, 1.0, 7)
    return (
        ToyCase("small_ry", X, np.array([0.15, 0.4])),
        ToyCase("large_ry", X, np.array([1.10, 0.4])),
        ToyCase("phase_change_only", X, np.array([0.15, 1.8])),
    )


def run_study(*, step=1e-6):
    return [evaluate_case(case, step=step) for case in default_cases()]


def save_results(output_dir, rows, metadata):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "qntk_cross_metric.csv"
    json_path = output_dir / "qntk_cross_metric_metadata.json"

    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

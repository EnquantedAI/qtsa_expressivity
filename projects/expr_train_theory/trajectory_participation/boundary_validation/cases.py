from __future__ import annotations

import numpy as np

from projects.expr_train_theory.trajectory_participation.core import (
    trajectory_participation_dimension,
)
from projects.expr_train_theory.trajectory_participation.qntk_cross_metric.qntk import (
    compute_qntk,
    diagnose_qntk,
    finite_difference_jacobian,
)


def collapsed_trajectory_case():
    state = np.array([1.0, 0.0], dtype=np.complex128)
    states = np.vstack([state, state, state, state])
    result = trajectory_participation_dimension(states)
    return {
        "case": "collapsed_trajectory",
        "expected_d_tp": 1.0,
        "d_tp": float(result.dimension),
        "rank": int(result.numerical_rank),
    }


def orthogonal_trajectory_case(size=4):
    states = np.eye(size, dtype=np.complex128)
    result = trajectory_participation_dimension(states)
    return {
        "case": "orthogonal_trajectory",
        "expected_d_tp": float(size),
        "d_tp": float(result.dimension),
        "rank": int(result.numerical_rank),
    }


def constant_output_qntk_case():
    X = np.array([[-1.0], [0.0], [0.5], [1.0]], dtype=float)
    theta = np.array([0.2, -0.7], dtype=float)

    def model(x, parameters):
        del x, parameters
        return 0.25

    jac = finite_difference_jacobian(model, theta, X)
    kernel = compute_qntk(model, theta, X)
    diagnostics = diagnose_qntk(kernel)
    return {
        "case": "constant_output_qntk",
        "jacobian_norm": float(np.linalg.norm(jac)),
        "kernel_norm": float(np.linalg.norm(kernel)),
        "rank": int(diagnostics.rank),
        "trace": float(diagnostics.trace),
    }


def linear_qntk_case():
    X = np.array([[-1.0], [0.5], [2.0]], dtype=float)
    theta = np.array([0.3, -0.2], dtype=float)

    def model(x, parameters):
        return parameters[0] * x[0] + parameters[1]

    jac = finite_difference_jacobian(model, theta, X)
    expected_jac = np.column_stack([X[:, 0], np.ones(X.shape[0])])
    kernel = compute_qntk(model, theta, X)
    expected_kernel = expected_jac @ expected_jac.T
    diagnostics = diagnose_qntk(kernel)
    return {
        "case": "linear_qntk",
        "jacobian_error": float(np.linalg.norm(jac - expected_jac)),
        "kernel_error": float(np.linalg.norm(kernel - expected_kernel)),
        "rank": int(diagnostics.rank),
    }


def redundant_parameter_qntk_case():
    X = np.array([[-1.0], [0.25], [0.75], [1.5]], dtype=float)
    theta = np.array([0.1, 0.6], dtype=float)

    def model(x, parameters):
        return (parameters[0] + parameters[1]) * x[0]

    jac = finite_difference_jacobian(model, theta, X)
    kernel = compute_qntk(model, theta, X)
    diagnostics = diagnose_qntk(kernel)
    return {
        "case": "redundant_parameter_qntk",
        "jacobian_rank": int(np.linalg.matrix_rank(jac, tol=1e-9)),
        "qntk_rank": int(diagnostics.rank),
        "parameter_count": int(theta.size),
    }


def run_boundary_cases():
    return [
        collapsed_trajectory_case(),
        orthogonal_trajectory_case(),
        constant_output_qntk_case(),
        linear_qntk_case(),
        redundant_parameter_qntk_case(),
    ]

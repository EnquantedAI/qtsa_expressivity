from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class QNTKDiagnostics:
    rank: int
    trace: float
    largest_eigenvalue: float
    smallest_positive_eigenvalue: float
    condition_number: float
    effective_rank: float
    element_variance: float


def finite_difference_jacobian(model, theta, X, *, step=1e-6):
    """Jacobian of a scalar model output over a small dataset."""
    theta = np.asarray(theta, dtype=float).reshape(-1)
    X = np.asarray(X)
    if theta.size == 0:
        raise ValueError("theta must contain at least one parameter")
    if step <= 0:
        raise ValueError("step must be positive")

    base = np.asarray([model(x, theta) for x in X], dtype=float).reshape(-1)
    jac = np.empty((base.size, theta.size), dtype=float)

    for j in range(theta.size):
        plus = theta.copy()
        minus = theta.copy()
        plus[j] += step
        minus[j] -= step
        f_plus = np.asarray([model(x, plus) for x in X], dtype=float).reshape(-1)
        f_minus = np.asarray([model(x, minus) for x in X], dtype=float).reshape(-1)
        if f_plus.shape != base.shape or f_minus.shape != base.shape:
            raise ValueError("model output shape changed under parameter perturbation")
        jac[:, j] = (f_plus - f_minus) / (2.0 * step)

    return jac


def qntk_from_jacobian(jacobian):
    jacobian = np.asarray(jacobian, dtype=float)
    if jacobian.ndim != 2:
        raise ValueError("jacobian must be a matrix")
    return jacobian @ jacobian.T


def compute_qntk(model, theta, X, *, step=1e-6):
    jac = finite_difference_jacobian(model, theta, X, step=step)
    return qntk_from_jacobian(jac)


def diagnose_qntk(kernel, *, atol=1e-10, rtol=1e-8):
    kernel = np.asarray(kernel, dtype=float)
    if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
        raise ValueError("kernel must be square")
    sym = 0.5 * (kernel + kernel.T)
    values = np.linalg.eigvalsh(sym)
    largest = max(0.0, float(values[-1])) if values.size else 0.0
    threshold = max(atol, rtol * largest)
    positive = values[values > threshold]
    rank = int(positive.size)
    trace = float(np.trace(sym))

    if positive.size:
        smallest = float(positive[0])
        condition = float(positive[-1] / positive[0])
    else:
        smallest = 0.0
        condition = float("inf")

    if trace > 0:
        probabilities = np.clip(values, 0.0, None) / trace
        denom = float(np.sum(probabilities**2))
        effective_rank = 1.0 / denom if denom > 0 else 0.0
    else:
        effective_rank = 0.0

    return QNTKDiagnostics(
        rank=rank,
        trace=trace,
        largest_eigenvalue=largest,
        smallest_positive_eigenvalue=smallest,
        condition_number=condition,
        effective_rank=float(effective_rank),
        element_variance=float(np.var(sym)),
    )

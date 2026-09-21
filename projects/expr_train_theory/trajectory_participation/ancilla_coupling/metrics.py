from __future__ import annotations

import numpy as np


def _state_vector(state, n_qubits):
    if int(n_qubits) < 1:
        raise ValueError("n_qubits must be positive")
    vector = np.asarray(state, dtype=np.complex128).reshape(-1)
    expected = 2 ** int(n_qubits)
    if vector.size != expected:
        raise ValueError(f"state has dimension {vector.size}, expected {expected}")
    norm = float(np.vdot(vector, vector).real)
    if not np.isclose(norm, 1.0, atol=1e-8, rtol=1e-8):
        raise ValueError("state must be normalized")
    return vector


def ancilla_reduced_state(state, n_qubits, n_ancilla):
    """Reduced state of the final ``n_ancilla`` wires of a pure state.

    The shared QNN places input/system wires first and added ancilla wires last,
    so reshaping the state into ``(system_dim, ancilla_dim)`` gives the desired
    system--ancilla bipartition.
    """
    n_qubits = int(n_qubits)
    n_ancilla = int(n_ancilla)
    if n_ancilla < 0 or n_ancilla > n_qubits:
        raise ValueError("n_ancilla must satisfy 0 <= n_ancilla <= n_qubits")

    vector = _state_vector(state, n_qubits)
    if n_ancilla == 0:
        return np.ones((1, 1), dtype=np.complex128)

    n_system = n_qubits - n_ancilla
    matrix = vector.reshape(2**n_system, 2**n_ancilla)
    rho = matrix.conj().T @ matrix
    # Symmetrise only to remove roundoff-level anti-Hermitian noise.
    return 0.5 * (rho + rho.conj().T)


def ancilla_entropy_purity(rho, *, eigenvalue_tol=1e-12):
    """Return von Neumann entropy (nats) and purity of a density matrix."""
    rho = np.asarray(rho, dtype=np.complex128)
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1] or rho.shape[0] < 1:
        raise ValueError("rho must be a non-empty square matrix")
    if not np.allclose(rho, rho.conj().T, atol=1e-8, rtol=1e-8):
        raise ValueError("rho must be Hermitian")

    trace = np.trace(rho)
    if not np.isclose(trace, 1.0, atol=1e-8, rtol=1e-8):
        raise ValueError("rho must have unit trace")

    eigvals = np.linalg.eigvalsh(rho).real
    if np.min(eigvals) < -1e-8:
        raise ValueError("rho must be positive semidefinite")
    eigvals = np.clip(eigvals, 0.0, None)
    total = float(np.sum(eigvals))
    if total <= 0.0:
        raise ValueError("rho has zero trace after spectral cleanup")
    eigvals /= total

    nonzero = eigvals[eigvals > float(eigenvalue_tol)]
    entropy = float(-np.sum(nonzero * np.log(nonzero)))
    purity = float(np.sum(eigvals**2))
    return entropy, purity


def ancilla_state_diagnostics(state, n_qubits, n_ancilla):
    """Return bounded diagnostics for the ancilla reduced state.

    For a globally pure state, reduced-state mixedness measures system--ancilla
    entanglement.  ``ancilla_participation_dimension = 1 / purity`` is a
    Renyi-2 effective Schmidt dimension and is conceptually parallel to, but
    distinct from, trajectory participation dimension.
    """
    n_qubits = int(n_qubits)
    n_ancilla = int(n_ancilla)
    rho = ancilla_reduced_state(state, n_qubits, n_ancilla)
    entropy, purity = ancilla_entropy_purity(rho)

    ancilla_dim = 2**n_ancilla
    participation = float(1.0 / purity)
    if n_ancilla == 0:
        entropy_fraction = 0.0
        mixedness_fraction = 0.0
    else:
        entropy_fraction = float(entropy / np.log(ancilla_dim))
        purity_floor = 1.0 / ancilla_dim
        mixedness_fraction = float((1.0 - purity) / (1.0 - purity_floor))

    return {
        "ancilla_entropy": entropy,
        "ancilla_entropy_fraction": entropy_fraction,
        "ancilla_purity": purity,
        "ancilla_participation_dimension": participation,
        "ancilla_participation_normalized": float(participation / ancilla_dim),
        "ancilla_mixedness_fraction": mixedness_fraction,
    }


def ancilla_growth(states, n_qubits, n_ancilla):
    """Instantaneous ancilla diagnostics for every trajectory snapshot."""
    states = np.asarray(states, dtype=np.complex128)
    if states.ndim != 2 or states.shape[0] < 1:
        raise ValueError("states must be a non-empty 2D trajectory")
    if states.shape[1] != 2 ** int(n_qubits):
        raise ValueError("trajectory state dimension does not match n_qubits")

    rows = []
    for depth, state in enumerate(states):
        rows.append({"depth": depth, **ancilla_state_diagnostics(state, n_qubits, n_ancilla)})
    return rows


def safe_pearson(x, y, *, atol=1e-14):
    """Pearson correlation, or ``None`` when either series has no variation."""
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if x.size != y.size:
        raise ValueError("x and y must have the same length")
    if x.size < 2:
        return None
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("correlation inputs must be finite")
    if np.std(x) <= atol or np.std(y) <= atol:
        return None
    return float(np.corrcoef(x, y)[0, 1])

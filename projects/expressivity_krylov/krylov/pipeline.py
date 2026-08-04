from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg, minres
from scipy.linalg import eigh_tridiagonal


@dataclass
class QNTKMetrics:
    condition_number: float
    effective_rank: float
    lambda_max: float
    lambda_min: float
    eigenvalues: np.ndarray


class InverseShiftOperator(LinearOperator):
    """
    Implicit operator computing y = (A - shift * I)^{-1} x using Conjugate Gradient / MINRES.
    Used for shift-and-invert Lanczos to target minimum eigenvalues.
    https://arxiv.org/html/2408.06554v1
    """

    def __init__(self, A: SymmetricOperator, shift: float = 0.0, tol: float = 1e-8):
        self.A = A
        self.shift = shift
        self.tol = tol
        self.shape = A.shape
        self.dtype = A.dtype

    def _matvec(self, x: np.ndarray) -> np.ndarray:
        if self.shift == 0.0:
            # A is symmetric positive semi-definite; CG is fast and implicit
            sol, info = cg(self.A, x, rtol=self.tol, maxiter=5000)
        else:
            # Shifted operator (A - sigma I) might be indefinite; use MINRES
            def shifted_matvec(v):
                return self.A.matvec(v) - self.shift * v

            shifted_op = LinearOperator(self.shape, matvec=shifted_matvec, dtype=self.dtype)
            sol, info = minres(shifted_op, x, rtol=self.tol, maxiter=5000)

        if info != 0:
            raise RuntimeError(f"Krylov solver failed to converge inside InverseShiftOperator (info code: {info}).")
        return sol


class KrylovExpressivityPipeline:
    """
    Krylov subspace analysis pipeline for computing QNTK expressivity metrics.
    """

    def __init__(self, operator: SymmetricOperator, k_dim: int = 50, reorth_tol: float = 1e-8):
        """
        Args:
            operator: Implicit symmetric operator (QNTK).
            k_dim: Dimension of the Krylov subspace (number of Lanczos steps).
            reorth_tol: Tolerance threshold for partial reorthogonalization.
        """
        if operator.shape[0] != operator.shape[1]:
            raise ValueError("Operator must be square.")

        self.A = operator
        self.N = operator.shape[0]
        self.k_dim = min(k_dim, self.N)
        self.reorth_tol = reorth_tol

    def lanczos(self, op: LinearOperator, k: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
        """
        Lanczos iteration with Partial Reorthogonalization (PRO).
        Returns tridiagonal matrix diagonal (alpha) and off-diagonal (beta).
        """
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(self.N)
        v /= np.linalg.norm(v)

        V = np.zeros((self.N, k + 1), dtype=op.dtype)
        V[:, 0] = v

        alpha = np.zeros(k, dtype=np.float64)
        beta = np.zeros(k, dtype=np.float64)

        v_next = op.matvec(V[:, 0])
        alpha[0] = np.dot(V[:, 0], v_next)
        v_next -= alpha[0] * V[:, 0]

        for j in range(1, k):
            beta[j - 1] = np.linalg.norm(v_next)

            if beta[j - 1] < 1e-12:
                # Early termination (invariant subspace hit)
                alpha = alpha[:j]
                beta = beta[: j - 1]
                break

            V[:, j] = v_next / beta[j - 1]

            # Reorthogonalization check (PRO)
            v_next = op.matvec(V[:, j]) - beta[j - 1] * V[:, j - 1]
            alpha[j] = np.dot(V[:, j], v_next)
            v_next -= alpha[j] * V[:, j]

            # Reorthogonalize against full Krylov basis if loss of orthogonality occurs
            for i in range(j + 1):
                overlap = np.dot(V[:, i], v_next)
                if abs(overlap) > self.reorth_tol:
                    v_next -= overlap * V[:, i]

        return alpha, beta

    def compute_lambda_max(self, k: int) -> float:
        """Estimates the largest eigenvalue using standard Lanczos."""
        alpha, beta = self.lanczos(self.A, k=k)
        eigvals = eigh_tridiagonal(alpha, beta, select="val", select_range=(len(alpha) - 1, len(alpha) - 1))
        return float(eigvals[-1])

    def compute_lambda_min(self, k: int, shift: float = 0.0, solver_tol: float = 1e-8) -> float:
        """
        Estimates the smallest eigenvalue using Shift-and-Invert Lanczos.
        Transforms lambda_min(A) into lambda_max((A - shift * I)^-1).
        """
        inv_op = InverseShiftOperator(self.A, shift=shift, tol=solver_tol)
        alpha, beta = self.lanczos(inv_op, k=k)
        inv_eigvals = eigh_tridiagonal(alpha, beta, select="val", select_range=(len(alpha) - 1, len(alpha) - 1))

        inv_lambda_max = float(inv_eigvals[-1])
        lambda_min = (1.0 / inv_lambda_max) + shift
        return float(lambda_min)

    def compute_spectrum(self, k: int) -> np.ndarray:
        """Estimates the full top-k Ritz spectrum."""
        alpha, beta = self.lanczos(self.A, k=k)
        eigvals = eigh_tridiagonal(alpha, beta, val_only=True)
        # Filter out numerical noise/negative artifacts for PSD QNTK
        eigvals = np.clip(eigvals, a_min=0.0, a_max=None)
        return np.sort(eigvals)[::-1]

    def analyze(
        self,
        k_spectrum: int | None = None,
        k_min: int = 20,
        shift: float = 0.0,
        eps: float = 1e-12,
    ) -> QNTKMetrics:
        """
        Runs the complete expressivity pipeline.

        Args:
            k_spectrum: Subspace dimension for spectrum and lambda_max.
            k_min: Subspace dimension for invert Lanczos targeting lambda_min.
            shift: Shift parameter for indefinite/zero-mode operators.
            eps: Regularization constant for log/probabilities.

        Returns:
            QNTKMetrics object containing condition number, effective rank, and spectrum details.
        """
        k_spec = k_spectrum or self.k_dim

        # 1. Compute spectrum and top eigenvalue
        spectrum = self.compute_spectrum(k=k_spec)
        lambda_max = float(spectrum[0])

        # 2. Compute minimum eigenvalue using Shift-and-Invert
        lambda_min = self.compute_lambda_min(k=k_min, shift=shift)

        # 3. Compute Condition Number
        # Avoid division by zero on singular matrices
        cond_num = lambda_max / max(lambda_min, eps)

        # 4. Compute Roy & Vetterli Effective Rank
        total_power = np.sum(spectrum)
        if total_power > 0:
            p = spectrum / total_power
            p = p[p > eps]  # Avoid log(0)
            entropy = -np.sum(p * np.log(p))
            erank = float(np.exp(entropy))
        else:
            erank = 0.0

        return QNTKMetrics(
            condition_number=cond_num,
            effective_rank=erank,
            lambda_max=lambda_max,
            lambda_min=lambda_min,
            eigenvalues=spectrum,
        )

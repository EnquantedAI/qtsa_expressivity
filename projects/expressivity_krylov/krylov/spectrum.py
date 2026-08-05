import numpy as np

from scipy.sparse.linalg import eigsh

from .result import SpectrumResult
from .operators.symmetric_operator import SymmetricOperator


def largest_eigenvalue(operator: SymmetricOperator) -> float:
    value = eigsh(
        operator,
        k=1,
        which="LA",
        return_eigenvectors=False,
    )[0]

    return float(value)


def smallest_eigenvalue(
    operator: SymmetricOperator,
) -> float:
    value = eigsh(
        operator,
        k=1,
        sigma=1e-3,
        which="LM",
        mode="normal",
        return_eigenvectors=False,
    )[0]

    return float(value)


def effective_rank(
    operator: SymmetricOperator,
    tolerance=1e-10,
):
    """
    Aproximate rank of A as tr(A)^2 / tr(A^2)
    [TODO]
    """    

    pass


def analyze(
    operator: SymmetricOperator,
    tolerance=1e-10,
):
    lmax = largest_eigenvalue(operator)
    lmin = smallest_eigenvalue(operator)

    rank = effective_rank(operator, tolerance)

    return SpectrumResult(
        lambda_max=lmax,
        lambda_min=lmin,
        rank=rank,
        condition_number=lmax / lmin,
    )

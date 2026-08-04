import numpy as np

from krylov.spectrum import analyze
from krylov.operators.dense_operator import DenseOperator

A = np.diag([7., 5., 3., 1.])

op = DenseOperator(A)

result = analyze(op)

print(f"Received: {result}")

print(f"Expected: SpectrumResult(lambda_max=7.000000000000000, lambda_min=1.0, rank=4, condition_number=7.000000000000000)")

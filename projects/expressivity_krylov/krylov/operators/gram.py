from .symmetric_operator import SymmetricOperator

class GramOperator(SymmetricOperator):

    def __init__(self, oracle):
        self.oracle = oracle
        super().__init__(
            (oracle.input_dimension,
             oracle.input_dimension)
        )

    def _matvec(self, x):
        return self.oracle.vjp(
            self.oracle.jvp(x)
        )

    def __matmul__(self, x):
        return 

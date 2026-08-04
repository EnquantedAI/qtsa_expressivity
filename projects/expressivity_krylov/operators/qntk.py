class QNTKOperator(SymmetricOperator):

    def __init__(self, oracle):
        self.oracle = oracle
        super().__init__(
            (oracle.output_dimension,
             oracle.output_dimension)
        )

    def _matvec(self, x):
        return self.oracle.jvp(
            self.oracle.vjp(x)
        )

# Effective Dimension review notes

The current implementation builds a classical Fisher information matrix from computational-basis probabilities obtained from the statevector. In the generic statistical notation $p_\theta(y\mid x)$, the output variable $y$ is identified here with the computational-basis measurement outcome $z$, so the implemented probabilities are $p_\theta(z\mid x)$. This is a valid CFIM, but its interpretation depends on whether that measurement is the statistical model we want to study.

Things worth checking before a large run:

- which input samples are used when averaging the Fisher matrix;
- how the Fisher matrix is normalised in GED and LED;
- the meaning of the sample-size parameter in the Effective Dimension formula;
- the local sampling radius and periodic parameter boundaries;
- sensitivity to the probability cutoff;
- whether results should also be reported as $d_{\mathrm{eff}}/d$.

The small examples in `effective_dimension_checks/` show why the measurement choice matters: a parameter can change the quantum state while leaving computational-basis probabilities unchanged.

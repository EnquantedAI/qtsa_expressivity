# Gram-matrix backend for trajectory participation

For a trajectory with snapshots collected in

$$
\Psi = [|\psi_0\rangle,\ldots,|\psi_{m-1}\rangle],
$$

the non-zero eigenvalues of

$$
G=\Psi^\dagger\Psi
$$

are the same as the non-zero eigenvalues of $\Psi\Psi^\dagger$ and the squared singular values of $\Psi$. Therefore $d_{TP}$ can be computed from the much smaller $m\times m$ Gram matrix whenever the number of snapshots is smaller than the Hilbert-space dimension.

For normalized snapshots,

$$
(G)_{kl}=\langle\psi_k|\psi_l\rangle
$$

and

$$
d_{TP}=\frac{(\mathrm{Tr}\,G)^2}{\mathrm{Tr}(G^2)}.
$$

This directory keeps a direct Gram-matrix implementation separate from the existing SVD implementation and checks that both give the same result. It also accepts a precomputed Gram matrix, which is useful if overlaps can be obtained without storing every full statevector.

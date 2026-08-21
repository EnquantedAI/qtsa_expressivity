# Overlap view of trajectory participation

For normalized snapshots $|\psi_1\rangle,\ldots,|\psi_m\rangle$ with equal weights, define

$$
\rho_T = \frac{1}{m}\sum_{k=1}^m |\psi_k\rangle\langle\psi_k|.
$$

Then

$$
\mathrm{Tr}(\rho_T^2)
= \frac{1}{m^2}\sum_{k,l=1}^m
|\langle\psi_k|\psi_l\rangle|^2,
$$

so

$$
d_{TP}
= \frac{1}{\mathrm{Tr}(\rho_T^2)}
= \frac{m^2}{\sum_{k,l}|\langle\psi_k|\psi_l\rangle|^2}.
$$

The denominator is the first state frame potential of the finite set of trajectory snapshots. This gives a direct geometric interpretation: repeated or strongly overlapping snapshots increase the frame potential and decrease $d_{TP}$, while orthogonal snapshots minimize the overlap sum and maximize $d_{TP}$.

For normalized non-negative weights $w_k$ the same identity becomes

$$
\mathrm{Tr}(\rho_T^2)
= \sum_{k,l} w_k w_l |\langle\psi_k|\psi_l\rangle|^2,
\qquad
\rho_T = \sum_k w_k |\psi_k\rangle\langle\psi_k|.
$$

This module checks the identity numerically against the existing SVD/Gram implementation. Here "frame potential" refers to a set of states, not the unitary frame potential used for ensembles of unitaries.

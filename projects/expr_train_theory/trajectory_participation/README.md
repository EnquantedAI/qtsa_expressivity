# Trajectory participation

This is a small reference implementation for the trajectory-based idea discussed in the project channel.

For state snapshots

$$
|\psi_0\rangle, |\psi_1\rangle, \ldots, |\psi_L\rangle,
$$

the trajectory matrix is

$$
\Psi = \begin{bmatrix}|\psi_0\rangle & |\psi_1\rangle & \cdots & |\psi_L\rangle\end{bmatrix}.
$$

If $\sigma_i$ are the singular values of $\Psi$, I use

$$
p_i = \frac{\sigma_i^2}{\sum_j \sigma_j^2},
\qquad
d_{TP} = \frac{1}{\sum_i p_i^2}.
$$

For normalized snapshots the same value follows from $G=\Psi^\dagger\Psi$:

$$
d_{TP} = \frac{(\mathrm{Tr}\,G)^2}{\mathrm{Tr}(G^2)}.
$$

The tests cover simple boundary cases such as identical and orthogonal snapshots, duplicated directions, phase changes and unitary basis changes.

```bash
python -m projects.expr_train_theory.trajectory_participation.run_validation
```

## Layer snapshots

`snapshots.py` contains a small NumPy circuit used for checking the idea with encoding, variational layers and optional reuploading.

`shared_qnn.py` follows the current shared QNN layout from `src.models`: the same feature-map choices, $R_Z-R_Y-R_Z$ weight block, layer-dependent CNOT pattern, input trimming and optional reuploading. It returns the encoded state and one state after every complete variational layer. For one input instance this is also the trajectory used with the batched model.

```bash
python -m projects.expr_train_theory.trajectory_participation.run_layered_example
python -m projects.expr_train_theory.trajectory_participation.run_shared_qnn_example
```

## Architecture sweep

`architecture_sweep/` repeats the shared-QNN trajectory calculation over small grids of layers, qubits, feature maps and reuploading choices. It keeps the input length independent from the qubit count and writes both raw values and grouped summaries.

## Directional checks

`directional_validation/` uses the small NumPy layered circuit to check how the trajectory quantities move with depth and with paired reuploading changes. The trends are recorded rather than assumed to be strictly monotone.

## Sampling sensitivity

`sampling_sensitivity/` checks how much the value depends on how densely the same trajectory is sampled. There is also a weighted form of the metric, which is useful when snapshot multiplicity should not automatically change the contribution of a region of the trajectory.

## Perturbation stability

`perturbation_stability/` checks how much $d_{TP}$ moves when either the snapshots themselves or the variational parameters are changed by a small amount. This is mainly a numerical sanity check before larger comparisons.

## Calibration

`calibration/` adds simple reference bounds and normalized views of $d_{TP}$. This is useful when comparing trajectories with different numbers of snapshots or different Hilbert-space dimensions.

`shared_qnn_cross_metric/` compares $d_{TP}$ with QFIM and CFIM on the same small QNN configurations.
## QNTK cross-check

`qntk_cross_metric/` contains a small NumPy reference for comparing trajectory participation with a QNTK built from output Jacobians on the same toy models.


## QNTK architecture sweep

`qntk_architecture_sweep/` repeats the small dTP/QNTK comparison over several toy layered architectures. It is mainly a check of how the two diagnostics move with depth, qubit count, entangling gates and reuploading.

## Combined validation

`validation_suite/` runs the main boundary, sampling, perturbation and directional checks together and writes one compact report. Only analytically fixed cases are treated as pass/fail checks.

`metric_uncertainty/` adds a bootstrap sanity check for correlations from the small cross-metric sweeps.

## Spectral interpretation

`spectral_properties/` makes the effective-rank interpretation explicit: $d_{TP}=\exp(H_2)$ for the trajectory spectrum and sits between the stable-rank and Shannon effective-dimension views before the hard numerical rank.

## Hard subspace comparison

`subspace_comparison/` compares the Gram-Schmidt/SVD subspace rank with $d_{TP}$. The hard rank only records how many independent directions are reached, while $d_{TP}$ also records how evenly the trajectory occupies them.

## Gram-matrix backend

`gram_backend/` contains an equivalent implementation based only on the snapshot Gram matrix. This is useful when the trajectory contains far fewer snapshots than the Hilbert-space dimension, or when overlaps are available without storing all statevectors.

## Arc-length weighting

`arc_length_weighting/` uses consecutive Fubini--Study distances to weight snapshots by the projective distance travelled rather than by snapshot count. It is a sampling diagnostic for comparing paths that were recorded at different densities; it does not assume that adding circuit layers leaves the underlying trajectory unchanged.

## Arc-length architecture comparison

`arc_length_architecture_sweep/` compares equal-weight and Fubini--Study-weighted $d_{TP}$ on the same QNN trajectories. It records the metric shift together with the total projective path length, so depth comparisons can be inspected for sampling effects instead of treating every extra snapshot as equivalent evidence of new state-space exploration.

## Entanglement graph topology

`graph_topology/` is the self-contained matched entanglement-topology study. It keeps inputs and variational parameter draws fixed while changing only the CNOT graph, and compares equal/FS-weighted $d_{TP}$ with QFIM, a reference QNTK and graph descriptors across width/depth and independent repeat seeds.

The stable release entry point is the frozen `topology_robust_v1` preset:

```bash
python -m projects.expr_train_theory.trajectory_participation.graph_topology.run_reproducible_study
```

Its output includes the raw matched rows, bootstrap uncertainty, robust-effect classifications and an exact seed/configuration manifest. Focused development runners and interpretation limits are documented in `graph_topology/README.md`.

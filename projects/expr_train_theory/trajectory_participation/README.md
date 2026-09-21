# Trajectory participation

This is a small reference implementation for the trajectory-based idea discussed in the project channel.

For state snapshots

```math
|\psi_0\rangle, |\psi_1\rangle, \ldots, |\psi_L\rangle,
```

the trajectory matrix is

```math
\Psi = \begin{bmatrix}|\psi_0\rangle & |\psi_1\rangle & \cdots & |\psi_L\rangle\end{bmatrix}.
```

If $\sigma_i$ are the singular values of $\Psi$, I use

```math
p_i = \frac{\sigma_i^2}{\sum_j \sigma_j^2},
\qquad
d_{TP} = \frac{1}{\sum_i p_i^2}.
```

For normalized snapshots the same value follows from $G=\Psi^\dagger\Psi$:

```math
d_{TP} = \frac{(\mathrm{Tr}\,G)^2}{\mathrm{Tr}(G^2)}.
```

The tests cover simple boundary cases such as identical and orthogonal snapshots, duplicated directions, phase changes and unitary basis changes.

```bash
python -m projects.expr_train_theory.trajectory_participation.run_validation
```

## Layer snapshots

`snapshots.py` contains a small NumPy circuit used for checking the idea with encoding, variational layers and optional reuploading.

`shared_qnn.py` now exposes two paths. `native_shared_qnn_snapshots` is the integration path: it instantiates the canonical `src.models.gqnn` with `add_snaps=True` and collects the named `depth_0`, ..., `depth_L` states through `qml.snapshots`. The existing `shared_qnn_snapshots` implementation is kept as an independently reconstructed reference path for cross-checking the shared model rather than as the preferred integration route.

```bash
python -m projects.expr_train_theory.trajectory_participation.run_layered_example
python -m projects.expr_train_theory.trajectory_participation.run_shared_qnn_example
python -m projects.expr_train_theory.trajectory_participation.run_native_shared_qnn_example
```

## Shared-QNN parity validation

`shared_qnn_parity/` checks the native `qml.snapshots(src.models.gqnn(..., add_snaps=True))` trajectory against the independently reconstructed reference path. It compares every layer snapshot up to global phase, the trajectory Gram magnitudes and the resulting $d_{TP}$ across a compact matrix of feature maps, reuploading choices, widths and depths.

```bash
python -m projects.expr_train_theory.trajectory_participation.shared_qnn_parity.run_study
```

## Native shared-QNN architecture study

`native_shared_qnn_study/` is the first architecture study that uses the canonical PennyLane `src.models.gqnn(..., add_snaps=True)` path directly. It keeps the feature dimension fixed, adds ancilla qubits explicitly, and varies depth, feature map and reuploading while reusing matched input/parameter prefixes across the grid. It reports equal-weight and Fubini--Study-weighted $d_{TP}$, projective path length and depth-resolved growth curves.

The default run uses synthetic inputs and untrained random weights, so it is an integration/architecture diagnostic rather than a trained-model result.

```bash
python -m projects.expr_train_theory.trajectory_participation.native_shared_qnn_study.run_study
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

## Trajectory / ancilla coupling

`ancilla_coupling/` connects the native shared-QNN trajectory metric with instantaneous ancilla-sector diagnostics.  At every snapshot it computes ancilla entropy, purity and the Renyi-2 effective Schmidt dimension $d_{anc}=1/\mathrm{Tr}(\rho_{anc}^2)$, then compares their depth-resolved evolution with equal/FS-weighted $d_{TP}$.  The two inverse-purity quantities refer to different density matrices and are kept conceptually separate.

```bash
python -m projects.expr_train_theory.trajectory_participation.ancilla_coupling.run_study
```

## Real-data bridge

`real_data_bridge/` applies the native shared-QNN trajectory and ancilla diagnostics to the preprocessed NARMA10 and Mackey-Glass `TensorDataset` files already maintained by `projects/trainability_effective_dim`. The bridge is read-only and uses deterministic held-out windows by default, so it does not create a competing data-preparation pipeline or select examples based on the resulting metrics.

The current default still uses untrained random weights. The stored forecasting target is recorded only for provenance and is not used in the state-space metrics, so these runs characterise how real project inputs probe the circuit geometry rather than forecasting quality.

```bash
python -m projects.expr_train_theory.trajectory_participation.real_data_bridge.run_study
```

## Real-data stability

`real_data_stability/` extends the held-out NARMA10/Mackey-Glass bridge across independent random parameter seeds. The same input windows are reused for every seed, and depth/ancilla effects are subtracted within the same window and parameter draw before aggregation. The summary reports sign fractions, separate input-window and weight-draw variability, and crossed-bootstrap confidence intervals for the matched effects.

The resulting `stable_positive`, `stable_negative`, and `unresolved` labels are finite-sample diagnostics for the untrained shared-QNN geometry, not claims about forecasting performance.

```bash
python -m projects.expr_train_theory.trajectory_participation.real_data_stability.run_study
```

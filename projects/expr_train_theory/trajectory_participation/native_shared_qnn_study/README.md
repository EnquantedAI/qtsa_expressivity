# Native shared-QNN trajectory study

This study evaluates trajectory participation directly on the team's canonical PennyLane QNN. Snapshots come from

```python
qml.snapshots(src.models.gqnn(..., add_snaps=True))
```

through `native_shared_qnn_snapshots`; the circuit is not reconstructed in this module.

The default diagnostic grid varies circuit depth, ancilla count, feature map and data reuploading. The input dimension is held fixed while ancilla wires increase the Hilbert-space width. Within each random sample, every configuration receives the same input and a slice of one master parameter tensor, so common parameters are matched across depth/width and exactly shared across feature-map/reuploading comparisons.

For each trajectory the study records equal-weight and Fubini--Study-weighted $d_{TP}$, their trajectory-ceiling-normalized forms, numerical trajectory rank and total Fubini--Study path length. It also saves depth-resolved growth curves using partial trajectories $\{\psi_0,\ldots,\psi_l\}$.

The default run uses synthetic inputs and untrained random weights. It is an architecture/integration diagnostic, not an empirical performance result. Trained weights and held-out project data should be used before interpreting the curves as a model result.

```bash
python -m projects.expr_train_theory.trajectory_participation.native_shared_qnn_study.run_study
```

Outputs are written to `results/` as raw/final summaries, depth-resolved raw/summary tables and metadata.

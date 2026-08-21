# Arc-length comparison across architectures

This check runs the equal-weight and Fubini--Study arc-length-weighted versions of $d_{TP}$ on the same QNN trajectories.

For a trajectory $|\psi_0\rangle,\ldots,|\psi_L\rangle$, the equal-weight metric assigns every stored snapshot the same mass. The arc-length version uses the consecutive projective distances

$$
\Delta s_k=\arccos |\langle\psi_k|\psi_{k+1}\rangle|
$$

and trapezoidal weights along the accumulated path length.

The point of the comparison is not to force the two definitions to agree. Adding a layer can change both the physical path and the sampling of that path. The report therefore stores both values, their difference, and the total Fubini--Study path length. This makes it easier to see whether an apparent depth effect is mostly associated with visiting new states or with counting more snapshots.

The default runner uses the current shared-QNN snapshot adapter and varies depth, qubit count, feature map and reuploading.

```bash
python -m projects.expr_train_theory.trajectory_participation.arc_length_architecture_sweep.run_sweep
```

Results are written to `results/` as raw and grouped CSV files plus a small metadata file.

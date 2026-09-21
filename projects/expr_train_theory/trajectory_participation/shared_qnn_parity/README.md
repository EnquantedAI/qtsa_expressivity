# Shared-QNN trajectory parity

This check validates the native PennyLane snapshot path against the independently reconstructed trajectory path in `trajectory_participation/shared_qnn.py`.

For matched inputs and weights it compares every `depth_0`, ..., `depth_L` state up to global phase and also checks the resulting trajectory Gram magnitudes and $d_{TP}$. The default parity matrix covers the shared feature-map families (`X`, `Y`, `Z`, `zzfm`, `iqp`), the supported reupload axes, more than one circuit width and more than one depth.

The native side uses `src.models.gqnn(..., add_snaps=True)` through `qml.snapshots`. The reference side reconstructs the same layer sequence independently. This is an integration check, not a replacement for the native shared-model path.

Run:

```bash
python -m projects.expr_train_theory.trajectory_participation.shared_qnn_parity.run_study
```

The runner writes raw matched comparisons, an architecture-level summary and the exact validation configuration to `results/`. It exits with a non-zero status if any configuration exceeds the requested tolerance.

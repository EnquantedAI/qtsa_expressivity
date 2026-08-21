# Entanglement topology study

This folder contains the matched entanglement-topology part of the trajectory-participation work. The controlled variable is the CNOT graph: inputs and variational parameter draws are reused across topologies, while the entangling graph changes between `none`, `line`, `ring`, `star` and `complete`.

The study is now treated as a self-contained analysis unit. Its official reproducible entry point is the frozen `topology_robust_v1` preset; the other runners remain available for focused diagnostics and development.

## Official reproducible study

Run from the repository root:

```bash
python -m projects.expr_train_theory.trajectory_participation.graph_topology.run_reproducible_study
```

The preset is intentionally fixed. It records the sampled widths and depths, topology set, baseline topology, repeat seeds, bootstrap seed, sample counts, finite-difference step and confidence settings in the output manifest. The release preset uses `none` as the baseline; programmatic custom presets propagate their declared baseline through the matched-delta pipeline as well.

To inspect the exact preset without running the expensive experiment:

```bash
python -m projects.expr_train_theory.trajectory_participation.graph_topology.run_reproducible_study --show-preset
```

A custom destination can be used without changing the preset itself:

```bash
python -m projects.expr_train_theory.trajectory_participation.graph_topology.run_reproducible_study \
  --output-dir /path/to/output
```

By default, the release tables are written to:

```text
graph_topology/results/topology_robust_v1/
```

Generated result files are local experiment output and are not part of the committed source tree.

## What the official study measures

For every matched width/depth configuration, the pipeline combines:

- graph descriptors: edge count, density, degree statistics, connected components, shortest-path statistics, diameter and algebraic connectivity;
- equal-weight trajectory participation $d_{TP}$;
- Fubini--Study arc-length-weighted $d_{TP}$ and projective path length;
- trajectory rank;
- pure-state QFIM rank/relative-rank and trace diagnostics;
- a reference QNTK built from the final-state observable $\langle Z_0\rangle$;
- matched shifts relative to the `none` topology;
- repeat-seed uncertainty and percentile bootstrap intervals;
- conservative robust-effect labels.

Across depths, the topology sweep also reports

$$
\frac{d_{TP}}{\min(L+1,2^n)},
$$

so the larger trajectory ceiling created by adding snapshots is not itself interpreted as a topology effect.

The bootstrap unit is one complete independent repeat seed *after* the matched topology comparison. Rows belonging to different topologies inside one repeat are therefore not treated as independent observations.

## Robust-effect interpretation

At a single width/depth configuration, a matched effect is labelled `stable_positive` or `stable_negative` only when the full bootstrap interval lies on one side of zero. If the interval overlaps zero, the result is `unresolved`.

The aggregate topology/metric label is stricter: it is called stable only when every sampled width/depth configuration is resolved in the same direction. The report keeps the resolved fraction and the direction among resolved configurations so that partial evidence is not hidden by the aggregate label.

These labels describe the finite sampled grid and the chosen parameter/data distribution. They are not universal topology laws. In particular, the code does **not** assume that edge count, graph density or algebraic connectivity determines $d_{TP}$, QFIM or QNTK monotonically. Disagreement between the state-space, parameter-space and output-space diagnostics is an expected scientific outcome rather than an implementation error.

Small-width graph families can also become structurally identical. For example, on three qubits the undirected `ring` and `complete` edge sets coincide, so they should not be interpreted as distinct graph structures in that case.

## Output files

The official runner writes the complete release table set into one directory:

- `graph_topology_multiseed_raw.csv` — matched rows for every repeat/configuration/topology;
- `graph_topology_multiseed_uncertainty.csv` — repeat-level means, spread and bootstrap intervals;
- `graph_topology_multiseed_ci_stability.csv` — interval-direction stability across width/depth;
- `graph_topology_robust_configurations.csv` — configuration-level robust labels;
- `graph_topology_robust_summary.csv` — final topology/metric summary;
- `graph_topology_reproducible_manifest.json` — exact frozen study configuration;
- the corresponding metadata JSON files used by the uncertainty and robust-summary layers.

The manifest is the authoritative record of the experiment settings used for a reproduced run.

## Supporting diagnostic runners

The following scripts are intentionally kept outside the stable public interface. They are useful when inspecting one layer of the analysis, but they are not the recommended command for reproducing the complete topology result.

| Runner | Purpose |
|---|---|
| `run_study.py` | Original graph-descriptor versus $d_{TP}$ matched check. |
| `run_cross_metric.py` | Matched $d_{TP}$/QFIM/QNTK comparison. |
| `run_descriptor_analysis.py` | Matched deltas versus `none` and descriptive graph-descriptor associations. |
| `run_fs_weighting.py` | Equal-weight versus FS-weighted $d_{TP}$ under topology changes. |
| `run_comprehensive.py` | Main metrics collected in one matched experiment. |
| `run_scaling_sweep.py` | Width/depth sweep and descriptive sign consistency. |
| `run_uncertainty_sweep.py` | Multi-seed uncertainty for matched topology shifts. |
| `run_robust_summary.py` | Robust classification built from uncertainty rows. |

All of these are run as Python modules from the repository root, using the same package prefix as the official runner.

## Public Python interface

The package-level `graph_topology` namespace intentionally exports only the reproducible-study surface:

```python
from projects.expr_train_theory.trajectory_participation import graph_topology

preset = graph_topology.DEFAULT_REPRODUCIBLE_PRESET
result = graph_topology.run_reproducible_topology_study(preset)
```

Focused helpers should be imported from their defining modules when they are needed for development or analysis. This keeps the stable interface small while preserving all previous diagnostic code.

# Changelog

I added a NumPy implementation of the pure-state QFIM and checked it on several small examples with known analytical results. The examples cover a single rotation, redundant parameters, independent rotations and a global phase.

I then added a PennyLane adapter for the shared feature maps and `StronglyEntanglingLayers`. The adapter returns the full state of the circuit and does not change the forecasting models in `src/`.

There is also a small script for comparing QFIM diagnostics across different circuit widths and depths. It saves the raw values, a summary and the settings used for the run. This part still needs to be run in the project environment with PennyLane installed.

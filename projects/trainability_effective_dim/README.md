# trainability_effective_dim

QNN-based time series forecasting with expressivity analysis via effective dimension.

## How to run

Open `experiment_setup.ipynb` and execute cells sequentially. You can skip the code associated with finding effective
hyperparameters since it is very computationally heavy. Effective parameters are already given to basic_trainer  
The pipeline covers:
data generation → IO-pair creation → (Optional) Optuna hyperparameter search → basic training → simple & continuous forecasting → loss visualization 

## Status

- `core/measures/led.py` is a stub — local effective dimension logic not yet implemented.
- `core/measures/ged.py` is a stub — global effective dimension logic not yet implemented.
- `tests/` is empty. It's meant to test the correctness of implemented code.
- `core/trainers/statistical_trainer.py` is not yet implemented.
- `core/trainers/hyperparameter_trainer.py` is not yet fully implemented.

## Structure

```
trainability_effective_dim/
├── core/
│   ├── model/models.py               # GQNN: PennyLane QNN wrapped as TorchLayer
│   ├── measures/led.py               # [WIP] Local Effective Dimension computation
│   ├── measures/ged.py               # [WIP] Global Effective Dimension computation
│   ├── trainers/
│   │   ├── abstract_trainer.py       # AbstractTrainer base class
│   │   ├── basic_trainer.py          # BasicTrainer: train + simple/continuous forecast
│   │   └── hyperparameter_trainer.py # [WIP] HyperparameterTrainer: Optuna hyperparam search
│   │   └── statistical_trainer.py    # [WIP] trainer meant to measure the statstics of multiple training runs
│   └── datasets/                     # Preprocessed .pt datasets (Mackey-Glass, NARMA10)
├── references/                       # Background papers on effective dimension
├── results/                          # Figures and hyperparameter search .pkl
├── tests/                            # [WIP]
├── experiment_setup.ipynb            # [WIP] End-to-end pipeline notebook
└── README.md
```

## Dependencies (within `qtsa_expressivity`)

- `src.datagen` — Mackey-Glass / NARMA10 time series generation
- `src.models` — circuit drawing utilities, `gqnn_shape`, `zz_feature_map`
- `projects.expr_train_theory.qfim` — pure-state QFIM calculation and diagnostics.
- `utils.Charts` — plotting helpers

## Notes on how the code works


1. **Data preparation** — Time series (Mackey-Glass, NARMA10) are generated via `src.datagen`. They are chronologically split into training, validation, and testing chunks, scaled using train-only statistics, windowed into input-output pairs using a sliding window, and saved as PyTorch `.pt` `TensorDataset`s.

2. **Model** — The `GQNN` class in `core/model/models.py` defines a variational quantum circuit. Classical inputs are embedded via a configurable feature map (ZZ, IQP, or AngleEmbedding), followed by `StronglyEntanglingLayers` with trainable weights. It offers two execution modes:
   - `net.qlayers`: A `qml.qnn.TorchLayer` module that returns Pauli-Z expectation values for PyTorch training.
   - `net.state_function`: A NumPy-compatible explicit parameter callable that returns the analytical statevector (used for QFIM and effective dimension).
   
3. **Training** — `BasicTrainer` and `HyperparameterTrainer` in `core/trainers/` load the `.pt` datasets, instantiate the model from a config dict (device, layers, qubits, feature map, etc.), and run an Adam-based epoch loop with optional L1/L2 regularization. The `HyperparameterTrainer` additionally integrates Optuna trials with pruning.

4. **Forecasting** — Two evaluation modes in `BasicTrainer`:
   - `test_simple_forecast` — one-step prediction per window (ground-truth inputs at each step).
   - `test_continuous_forecast` — autoregressive: each prediction is fed back as input for the next step, simulating multi-step ahead forecasting.

5. **Orchestration** — The full pipeline (generation → windowing → training → evaluation) is run from `experiment_setup.ipynb`.
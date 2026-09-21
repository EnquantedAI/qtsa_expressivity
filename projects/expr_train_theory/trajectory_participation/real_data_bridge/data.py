from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


PROJECT_DATASET_NAMES = (
    "NARMA10_Chaotic",
    "Mackey_Glass_tau_30",
)
PROJECT_SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class DatasetWindows:
    """Preprocessed time-series windows loaded from the team's dataset files."""

    name: str
    split: str
    inputs: np.ndarray
    targets: np.ndarray
    source_path: Path

    @property
    def n_windows(self) -> int:
        return int(self.inputs.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.inputs.shape[1])


def repository_root() -> Path:
    return Path(__file__).resolve().parents[4]


def project_dataset_dir(repo_root=None) -> Path:
    root = Path(repo_root) if repo_root is not None else repository_root()
    return root / "projects" / "trainability_effective_dim" / "core" / "datasets"


def project_dataset_path(name, split="test", *, repo_root=None) -> Path:
    if name not in PROJECT_DATASET_NAMES:
        raise ValueError(
            f"unsupported dataset {name!r}; expected one of {PROJECT_DATASET_NAMES}"
        )
    if split not in PROJECT_SPLITS:
        raise ValueError(f"unsupported split {split!r}; expected one of {PROJECT_SPLITS}")
    return project_dataset_dir(repo_root) / f"{split}_dataset_{name}.pt"


def load_project_dataset(name, split="test", *, repo_root=None) -> DatasetWindows:
    """Load an existing preprocessed TensorDataset without modifying its owner project."""
    path = project_dataset_path(name, split, repo_root=repo_root)
    if not path.is_file():
        raise FileNotFoundError(path)

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - dependency is part of the repo env
        raise ImportError("PyTorch is required to read the shared .pt datasets") from exc

    dataset = torch.load(path, map_location="cpu", weights_only=False)
    tensors = getattr(dataset, "tensors", None)
    if tensors is None or len(tensors) != 2:
        raise TypeError("expected a TensorDataset containing (inputs, targets)")

    inputs = np.asarray(tensors[0].detach().cpu().numpy(), dtype=float)
    targets = np.asarray(tensors[1].detach().cpu().numpy(), dtype=float).reshape(-1)
    if inputs.ndim != 2 or inputs.shape[0] < 1 or inputs.shape[1] < 1:
        raise ValueError("dataset inputs must be a non-empty 2D array")
    if targets.shape != (inputs.shape[0],):
        raise ValueError("dataset targets must contain one value per input window")
    if not np.all(np.isfinite(inputs)) or not np.all(np.isfinite(targets)):
        raise ValueError("dataset contains non-finite values")

    return DatasetWindows(
        name=name,
        split=split,
        inputs=inputs,
        targets=targets,
        source_path=path,
    )


def select_window_indices(n_total, count, *, mode="even", seed=2026):
    """Choose a reproducible subset of windows without looking at metric values."""
    n_total = int(n_total)
    count = int(count)
    if n_total < 1:
        raise ValueError("n_total must be positive")
    if count < 1 or count > n_total:
        raise ValueError("count must satisfy 1 <= count <= n_total")

    if mode == "even":
        if count == 1:
            return np.asarray([n_total // 2], dtype=int)
        return np.rint(np.linspace(0, n_total - 1, count)).astype(int)
    if mode == "random":
        rng = np.random.default_rng(seed)
        return np.sort(rng.choice(n_total, size=count, replace=False).astype(int))
    raise ValueError("mode must be 'even' or 'random'")


def select_windows(dataset: DatasetWindows, count, *, mode="even", seed=2026):
    indices = select_window_indices(dataset.n_windows, count, mode=mode, seed=seed)
    return [
        {
            "window_index": int(index),
            "inputs": np.asarray(dataset.inputs[index], dtype=float).copy(),
            "target": float(dataset.targets[index]),
        }
        for index in indices
    ]

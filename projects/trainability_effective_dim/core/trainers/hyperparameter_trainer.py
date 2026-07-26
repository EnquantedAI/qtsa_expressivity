import gc
from os import PathLike
from typing import Any, Callable, Mapping, Optional, Union

import optuna
import pennylane as qml
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm.notebook import tqdm

from .abstract_trainer import AbstractTrainer

torch.serialization.add_safe_globals([TensorDataset])

PathType = Union[str, PathLike[str]]
Config = Mapping[str, Any]
ModelFactory = Callable[..., torch.nn.Module]
TrainingMetrics = dict[str, list[float]]


class HyperparameterTrainer(AbstractTrainer):
    """Train QNNs during an Optuna hyperparameter-optimization trial.

    This trainer creates a model from a sampled configuration, records epoch-level
    validation losses in Optuna, and prunes unpromising trials.
    """

    def __init__(
        self,
        training_path: PathType,
        validating_path: PathType,
        criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        testing_path: Optional[PathType] = None,
    ) -> None:
        """Initialize the trainer and load serialized datasets.

        Args:
            training_path: Path to the serialized training TensorDataset.
            validating_path: Path to the serialized validation TensorDataset.
            criterion: Loss function comparing predictions and target values.
            testing_path: Optional path to the serialized testing TensorDataset.
        """
        super().__init__(
            training_path=training_path,
            validating_path=validating_path,
            testing_path=testing_path,
            criterion=criterion,
        )

    def train_model(
        self,
        trial: optuna.trial.Trial,
        config: Config,
        model_class: ModelFactory,
    ) -> tuple[torch.nn.Module, TrainingMetrics]:
        """Train one model for an Optuna trial.

        After every epoch, the method reports the validation loss to Optuna. It
        raises ``optuna.TrialPruned`` when the configured pruning strategy stops
        the current trial.

        Args:
            trial: Active Optuna trial used for metric reporting and pruning.
            config: Nested dictionary containing training and model settings.
            model_class: Callable that builds the QNN PyTorch module.

        Returns:
            A tuple containing the trained QNN and per-epoch training and
            validation loss histories.

        Raises:
            optuna.TrialPruned: If Optuna determines the trial should stop early.
        """
        training_config = config["training_config"]
        model_config = config["model_config"]

        dev = qml.device(model_config["sim"], wires=model_config["n_qubits"])

        net = model_class(
            n_layers=model_config["n_layers"],
            n_qubits=model_config["n_qubits"],
            dev=dev,
            interface=model_config["interface"],
            diff_method=model_config["dif_method"],
            fm_style=model_config["fm_style"],
            meas=model_config["meas"],
        )

        device = training_config["device"]
        net = net.to(device)

        optimizer = Adam(
            net.parameters(),
            lr=training_config["optimizer"]["lr"],
            weight_decay=training_config["optimizer"]["weight_decay"],
        )

        trainloader = DataLoader(
            self.trainset,
            batch_size=int(training_config["batch_size"]),
            shuffle=True,
            num_workers=training_config["number_of_training_workers"],
            pin_memory=False if training_config["device"] == "cpu" else True,
        )
        valloader = DataLoader(
            self.valset,
            batch_size=int(training_config["batch_size"]),
            shuffle=False,
            num_workers=training_config["number_of_validating_workers"],
            pin_memory=False if training_config["device"] == "cpu" else True,
        )

        training_metrics = {"training_loss": [], "validating_loss": []}

        for epoch in tqdm(range(training_config["epochs"]), desc="Epochs"):
            net.train()
            epoch_steps = 0
            train_loss_sum = 0
            for inputs, targets in trainloader:
                inputs = inputs.to(device).squeeze(0)
                targets = targets.to(device).view(-1)

                optimizer.zero_grad()

                outputs = net(inputs).view(-1)

                loss = self.criterion(outputs, targets)

                if training_config["regularization"]["type"] == "l1":
                    penality = sum(p.abs().sum() for p in net.parameters())
                    loss = loss + training_config["regularization"]["lambda"] * penality

                elif training_config["regularization"]["type"] == "l2":
                    penality = sum((p**2).sum() for p in net.parameters())
                    loss = loss + training_config["regularization"]["lambda"] * penality

                loss.backward()
                optimizer.step()

                train_loss_sum += loss.item()
                epoch_steps += 1

            net.eval()
            val_loss_sum = 0.0
            val_steps = 0

            for inputs, targets in valloader:
                with torch.no_grad():
                    inputs = inputs.to(device).squeeze(0)
                    targets = targets.to(device).view(-1)

                    outputs = net(inputs).view(-1)

                    loss = self.criterion(outputs, targets)

                    if training_config["regularization"]["type"] == "l1":
                        penality = sum(p.abs().sum() for p in net.parameters())
                        loss = (
                            loss
                            + training_config["regularization"]["lambda"] * penality
                        )

                    elif training_config["regularization"]["type"] == "l2":
                        penality = sum((p**2).sum() for p in net.parameters())
                        loss = (
                            loss
                            + training_config["regularization"]["lambda"] * penality
                        )

                    val_loss_sum += loss.item()
                    val_steps += 1

            training_metrics["training_loss"].append(train_loss_sum / epoch_steps)
            training_metrics["validating_loss"].append(val_loss_sum / val_steps)

            trial.report(training_metrics["validating_loss"][-1], epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

            gc.collect()

        del trainloader
        del valloader

        return net, training_metrics

    def test_model(
        self,
        config: Config,
        model_class: ModelFactory,
    ) -> None:
        """Evaluate a model with a future testing implementation.

        Args:
            config: Configuration required for model construction or evaluation.
            model_class: Callable that builds the QNN PyTorch module.

        Returns:
            None. The testing workflow is not yet implemented.
        """
        pass
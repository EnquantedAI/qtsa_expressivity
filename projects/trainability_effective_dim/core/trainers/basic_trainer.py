from os import PathLike
from typing import Any, Callable, Mapping, Optional, Union

import pennylane as qml
import torch
from pennylane import numpy as np
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm.notebook import tqdm

from .abstract_trainer import AbstractTrainer

torch.serialization.add_safe_globals([TensorDataset])

PathType = Union[str, PathLike[str]]
Config = Mapping[str, Any]
ModelFactory = Callable[..., torch.nn.Module]
TrainingMetrics = dict[str, list[float]]
ForecastResult = tuple[np.ndarray, np.ndarray, np.ndarray]


class BasicTrainer(AbstractTrainer):
    """Train and evaluate a QNN for time-series forecasting.

    The trainer loads serialized TensorDataset objects through AbstractTrainer,
    trains a model with Adam, and supports one-step and autoregressive forecasts.
    """

    def __init__(
            self,
            training_path: PathType,
            validating_path: PathType,
            criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
            testing_path: Optional[PathType] = None,
    ) -> None:
        """Initialize the trainer and load the configured datasets.

        Args:
            training_path: Path to the serialized training TensorDataset.
            validating_path: Path to the serialized validation TensorDataset.
            criterion: Loss function comparing predictions and targets.
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
            config: Config,
            model_class: ModelFactory,
    ) -> tuple[torch.nn.Module, TrainingMetrics]:
        """Instantiate and train a QNN using the supplied configuration.

        Args:
            config: Nested configuration containing ``training_config`` and
                ``model_config`` dictionaries.
            model_class: Callable that creates the QNN PyTorch module.

        Returns:
            A tuple containing the trained model and a dictionary with per-epoch
            training and validation losses.
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
                    penality = sum((p ** 2).sum() for p in net.parameters())
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
                        penality = sum((p ** 2).sum() for p in net.parameters())
                        loss = (
                                loss
                                + training_config["regularization"]["lambda"] * penality
                        )

                    val_loss_sum += loss.item()
                    val_steps += 1

            training_metrics["training_loss"].append(train_loss_sum / epoch_steps)
            training_metrics["validating_loss"].append(val_loss_sum / val_steps)

        return net, training_metrics

    def test_model(
            self,
            config: Config,
            model_class: ModelFactory,
    ) -> None:
        """Evaluate a model using a future project-specific test procedure.

        Args:
            config: Configuration required to construct or evaluate the model.
            model_class: Callable that creates the QNN PyTorch module.

        Returns:
            None. This method is currently not implemented.
        """
        pass

    def test_simple_forecast(
            self,
            model_instance: torch.nn.Module,
            data: Any,
    ) -> ForecastResult:
        """Generate one-step forecasts using ground-truth input windows.

        Each validation window is independently passed to the model. Therefore,
        predictions do not affect the inputs used for subsequent forecasts.

        Args:
            model_instance: Trained QNN PyTorch module in evaluation mode.
            data: Reserved argument for externally provided evaluation data.
                The current implementation uses ``self.valset``.

        Returns:
            A tuple containing the first input window, one-step forecasts, and
            corresponding ground-truth target values.
        """
        testloader = DataLoader(
            self.valset,
            batch_size=1,
            shuffle=False,
        )

        forecasts = []
        ground_truth = []
        initial_data = None

        for inputs, target in testloader:
            with torch.no_grad():
                pred = model_instance(inputs).view(-1)

                if initial_data is None:
                    initial_data = inputs.squeeze(0).cpu().numpy().copy()

                forecasts.append(pred.cpu().numpy())
                ground_truth.append(target.view(-1).cpu().numpy())

        forecasts = np.concatenate(forecasts)
        ground_truth = np.concatenate(ground_truth)

        return initial_data, forecasts, ground_truth

    def test_continuous_forecast(
            self,
            model_instance: torch.nn.Module,
            data: Any,
    ) -> ForecastResult:
        """Generate autoregressive multi-step forecasts.

        The first validation window initializes the rolling input. Each prediction
        is appended to that input and used to produce the following prediction.

        Args:
            model_instance: Trained QNN PyTorch module in evaluation mode.
            data: Reserved argument for externally provided evaluation data.
                The current implementation uses ``self.valset``.

        Returns:
            A tuple containing the initial input window, autoregressive forecasts,
            and corresponding ground-truth target values.
        """
        testloader = DataLoader(
            self.valset,
            batch_size=1,
            shuffle=False,
        )

        forecasts = []
        initial_data = []
        ground_truth = []
        rolling_input = None

        for inputs, target in testloader:
            with torch.no_grad():
                if rolling_input is None:
                    rolling_input = inputs
                    initial_data = inputs.squeeze(0).cpu().numpy().copy()

                pred = model_instance(rolling_input).view(-1)

                rolling_input = torch.cat(
                    (rolling_input[:, 1:], pred.unsqueeze(1)),
                    dim=1,
                )

                forecasts.append(pred.cpu().numpy())
                ground_truth.append(target.view(-1).cpu().numpy())

        forecasts = np.concatenate(forecasts)
        ground_truth = np.concatenate(ground_truth)

        return initial_data, forecasts, ground_truth
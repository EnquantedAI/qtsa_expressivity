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
            testing_path: PathType,
            criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
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
            model_class: Any,
    ) -> tuple[Any, TrainingMetrics]:
        training_config = config["training_config"]
        model_config = config["model_config"]

        torch_device = torch.device(training_config["device"])

        quantum_device = qml.device(
            model_config["quantum_device"],
            wires=model_config["n_qubits"],
        )

        net = model_class(
            n_layers=model_config["n_layers"],
            n_qubits=model_config["n_qubits"],
            quantum_device=quantum_device,
            interface=model_config["interface"],
            diff_method=model_config["diff_method"],
            fm_style=model_config["fm_style"],
            meas=model_config["meas"],
        )

        qlayer = net.qlayers.to(torch_device)

        optimizer = Adam(
            qlayer.parameters(),
            lr=training_config["optimizer"]["lr"],
            weight_decay=training_config["optimizer"]["weight_decay"],
        )

        trainloader = DataLoader(
            self.trainset,
            batch_size=int(training_config["batch_size"]),
            shuffle=True,
            num_workers=int(training_config["number_of_training_workers"]),
            pin_memory=torch_device.type != "cpu",
        )

        valloader = DataLoader(
            self.valset,
            batch_size=int(training_config["batch_size"]),
            shuffle=False,
            num_workers=int(training_config["number_of_validating_workers"]),
            pin_memory=torch_device.type != "cpu",
        )

        regularization_type = training_config["regularization"]["type"]
        regularization_lambda = training_config["regularization"]["lambda"]

        training_metrics = {
            "training_loss": [],
            "validating_loss": [],
        }

        for _ in tqdm(range(int(training_config["epochs"])), desc="Epochs"):
            qlayer.train()
            train_loss_sum = 0.0
            train_steps = 0

            for inputs, targets in trainloader:
                inputs = inputs.to(
                    device=torch_device,
                    dtype=qlayer.weights.dtype,
                )
                targets = targets.to(
                    device=torch_device,
                    dtype=qlayer.weights.dtype,
                ).view(-1)

                if inputs.shape[0] != 1:
                    raise ValueError(
                        "GQNN currently requires batch_size=1 because its custom "
                        "ZZ feature map does not support batched inputs."
                    )

                inputs = inputs.squeeze(0)

                optimizer.zero_grad()

                outputs = qlayer(inputs).view(-1)
                loss = self.criterion(outputs, targets)

                if regularization_type == "l1":
                    penalty = sum(parameter.abs().sum() for parameter in qlayer.parameters())
                    loss = loss + regularization_lambda * penalty
                elif regularization_type == "l2":
                    penalty = sum((parameter ** 2).sum() for parameter in qlayer.parameters())
                    loss = loss + regularization_lambda * penalty

                loss.backward()
                optimizer.step()

                train_loss_sum += loss.item()
                train_steps += 1

            qlayer.eval()
            validation_loss_sum = 0.0
            validation_steps = 0

            with torch.no_grad():
                for inputs, targets in valloader:
                    inputs = inputs.to(
                        device=torch_device,
                        dtype=qlayer.weights.dtype,
                    )
                    targets = targets.to(
                        device=torch_device,
                        dtype=qlayer.weights.dtype,
                    ).view(-1)

                    if inputs.shape[0] != 1:
                        raise ValueError(
                            "GQNN currently requires batch_size=1 because its custom "
                            "ZZ feature map does not support batched inputs."
                        )

                    inputs = inputs.squeeze(0)

                    outputs = qlayer(inputs).view(-1)
                    loss = self.criterion(outputs, targets)

                    if regularization_type == "l1":
                        penalty = sum(parameter.abs().sum() for parameter in qlayer.parameters())
                        loss = loss + regularization_lambda * penalty
                    elif regularization_type == "l2":
                        penalty = sum((parameter ** 2).sum() for parameter in qlayer.parameters())
                        loss = loss + regularization_lambda * penalty

                    validation_loss_sum += loss.item()
                    validation_steps += 1

            if train_steps == 0:
                raise ValueError("The training dataset is empty.")
            if validation_steps == 0:
                raise ValueError("The validation dataset is empty.")

            training_metrics["training_loss"].append(train_loss_sum / train_steps)
            training_metrics["validating_loss"].append(
                validation_loss_sum / validation_steps
            )

        return net, training_metrics

    def test_model(
            self,
            config: Config,
            model_class: Any,
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
            model_instance: Any,
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
            self.testset,
            batch_size=1,
            shuffle=False,
        )

        forecasts = []
        ground_truth = []
        initial_data = None

        for inputs, target in testloader:
            with torch.no_grad():
                pred = model_instance.qlayers(inputs.squeeze(0)).view(-1)

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
            self.testset,
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

                pred = model_instance.qlayers(rolling_input.squeeze(0)).view(-1)

                rolling_input = torch.cat(
                    (rolling_input[:, 1:], pred.unsqueeze(1)),
                    dim=1,
                )

                forecasts.append(pred.cpu().numpy())
                ground_truth.append(target.view(-1).cpu().numpy())

        forecasts = np.concatenate(forecasts)
        ground_truth = np.concatenate(ground_truth)

        return initial_data, forecasts, ground_truth
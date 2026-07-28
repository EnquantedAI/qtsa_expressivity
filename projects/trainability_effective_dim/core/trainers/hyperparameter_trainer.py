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
            model_class: Any,
    ) -> tuple[Any, TrainingMetrics]:
        """Train one GQNN for an Optuna trial."""
        training_config = config["training_config"]
        model_config = config["model_config"]

        torch_device = torch.device(training_config["device"])

        quantum_device = qml.device(
            model_config["quantum_device"],
            wires=int(model_config["n_qubits"]),
        )

        net = model_class(
            n_layers=int(model_config["n_layers"]),
            n_qubits=int(model_config["n_qubits"]),
            quantum_device=quantum_device,
            interface=model_config["interface"],
            diff_method=model_config["diff_method"],
            fm_style=model_config["fm_style"],
            meas=model_config["meas"],
        )

        qlayer = net.qlayers.to(torch_device)

        optimizer = Adam(
            qlayer.parameters(),
            lr=float(training_config["optimizer"]["lr"]),
            weight_decay=float(training_config["optimizer"]["weight_decay"]),
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

        for epoch in tqdm(
                range(int(training_config["epochs"])),
                desc=f"Trial {trial.number}",
        ):
            qlayer.train()
            train_loss_sum = 0.0
            train_steps = 0

            for inputs, targets in trainloader:
                if inputs.shape[0] != 1:
                    raise ValueError(
                        "GQNN currently requires batch_size=1 because its custom "
                        "ZZ feature map is not implemented for batched inputs."
                    )

                inputs = inputs.to(
                    device=torch_device,
                    dtype=qlayer.weights.dtype,
                ).squeeze(0)

                targets = targets.to(
                    device=torch_device,
                    dtype=qlayer.weights.dtype,
                ).view(-1)

                optimizer.zero_grad()

                outputs = qlayer(inputs).view(-1)
                loss = self.criterion(outputs, targets)

                if regularization_type == "l1":
                    penalty = sum(
                        parameter.abs().sum()
                        for parameter in qlayer.parameters()
                    )
                    loss = loss + float(regularization_lambda) * penalty

                elif regularization_type == "l2":
                    penalty = sum(
                        (parameter ** 2).sum()
                        for parameter in qlayer.parameters()
                    )
                    loss = loss + float(regularization_lambda) * penalty

                loss.backward()
                optimizer.step()

                train_loss_sum += loss.item()
                train_steps += 1

            qlayer.eval()
            validation_loss_sum = 0.0
            validation_steps = 0

            with torch.no_grad():
                for inputs, targets in valloader:
                    if inputs.shape[0] != 1:
                        raise ValueError(
                            "GQNN currently requires batch_size=1 because its custom "
                            "ZZ feature map is not implemented for batched inputs."
                        )

                    inputs = inputs.to(
                        device=torch_device,
                        dtype=qlayer.weights.dtype,
                    ).squeeze(0)

                    targets = targets.to(
                        device=torch_device,
                        dtype=qlayer.weights.dtype,
                    ).view(-1)

                    outputs = qlayer(inputs).view(-1)
                    loss = self.criterion(outputs, targets)

                    if regularization_type == "l1":
                        penalty = sum(
                            parameter.abs().sum()
                            for parameter in qlayer.parameters()
                        )
                        loss = loss + float(regularization_lambda) * penalty

                    elif regularization_type == "l2":
                        penalty = sum(
                            (parameter ** 2).sum()
                            for parameter in qlayer.parameters()
                        )
                        loss = loss + float(regularization_lambda) * penalty

                    validation_loss_sum += loss.item()
                    validation_steps += 1

            if train_steps == 0:
                raise ValueError("The training dataset is empty.")
            if validation_steps == 0:
                raise ValueError("The validation dataset is empty.")

            mean_train_loss = train_loss_sum / train_steps
            mean_validation_loss = validation_loss_sum / validation_steps

            training_metrics["training_loss"].append(mean_train_loss)
            training_metrics["validating_loss"].append(mean_validation_loss)

            trial.report(mean_validation_loss, epoch)

            if trial.should_prune():
                del trainloader
                del valloader
                gc.collect()
                raise optuna.TrialPruned()

            gc.collect()

        del trainloader
        del valloader
        gc.collect()

        return net, training_metrics

    def test_model(
        self,
        config: Config,
        model_class: Any,
    ) -> None:
        """Evaluate a model with a future testing implementation.

        Args:
            config: Configuration required for model construction or evaluation.
            model_class: Callable that builds the QNN PyTorch module.

        Returns:
            None. The testing workflow is not yet implemented.
        """
        pass
import pennylane as qml
import torch
from pennylane import numpy as np
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm.notebook import tqdm

from .abstract_trainer import AbstractTrainer

torch.serialization.add_safe_globals([TensorDataset])


class BasicTrainer(AbstractTrainer):
    def __init__(
            self, training_path, validating_path, criterion, testing_path=None
    ) -> None:
        super().__init__(
            training_path=training_path,
            validating_path=validating_path,
            testing_path=testing_path,
            criterion=criterion,
        )

    def train_model(self, config, model_class):
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

    def test_model(self, config, model_class):
        pass

    def test_simple_forecast(self, model_instance, data):
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

    def test_continuous_forecast(self, model_instance, data):
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

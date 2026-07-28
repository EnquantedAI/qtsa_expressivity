from abc import ABC, abstractmethod
from os import PathLike
from typing import Any, Callable, Optional, Union

import torch


PathType = Union[str, PathLike[str]]


class AbstractTrainer(ABC):
    """Abstract base class for trainers using serialized PyTorch datasets.

    Attributes:
        trainset: Dataset loaded from ``training_path``.
        valset: Dataset loaded from ``validating_path``.
        testset: Dataset loaded from ``testing_path``, when provided.
        criterion: Loss function used to evaluate model predictions.
    """

    def __init__(
        self,
        training_path: PathType,
        validating_path: PathType,
        testing_path: PathType,
        criterion: Callable[..., torch.Tensor],
    ) -> None:
        """Load training, validation, and optionally test datasets.

        Args:
            training_path: Path to the serialized training dataset.
            validating_path: Path to the serialized validation dataset.
            criterion: Callable loss function that compares predictions with targets.
            testing_path: Optional path to the serialized test dataset.
        """
        self.trainset = torch.load(training_path)
        self.valset = torch.load(validating_path)
        self.testset = torch.load(testing_path)

        self.criterion = criterion

    @abstractmethod
    def train_model(self, config: dict[str, Any], model: Any) -> Any:
        """Train a model using a supplied training configuration.

        Args:
            config: Mapping containing model, optimizer, and training settings.
            model: PyTorch module to train.

        Returns:
            Implementation-specific training result, such as the trained model,
            loss history, or training metrics.
        """
        pass

    @abstractmethod
    def test_model(self, model: Any) -> Any:
        """Evaluate a trained model on the configured evaluation dataset.

        Args:
            model: Trained PyTorch module to evaluate.

        Returns:
            Implementation-specific evaluation result, such as predictions,
            loss values, or forecasting metrics.
        """
        pass
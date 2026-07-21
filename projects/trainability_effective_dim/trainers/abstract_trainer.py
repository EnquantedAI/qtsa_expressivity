from abc import ABC, abstractmethod
import torch


class AbstractTrainer(ABC):
    def __init__(self, training_path, validating_path, criterion, testing_path=None):
        self.trainset = torch.load(training_path)
        self.valset = torch.load(validating_path)
        if testing_path:
            self.testset = torch.load(testing_path)

        self.criterion = criterion

    @abstractmethod
    def train_model(self, config, model):
        pass

    @abstractmethod
    def test_model(self, model):
        pass

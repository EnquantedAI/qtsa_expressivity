import torch

class UniformParameterSpace:
    def __init__(self, low, high, shape, dtype=torch.float64):
        self.low = low
        self.high = high
        self.shape = shape
        self.dtype = dtype

    def sample(self):
        return torch.empty(self.shape, dtype=self.dtype).uniform_(
            self.low,
            self.high,
        )

class EpsilonBallParameterSpace:
    def __init__(self, center, epsilon, generator=None):
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")

        self.center = center.detach().clone()
        self.epsilon = epsilon
        self.generator = generator
        self.dimension = self.center.numel()

    def sample(self):
        direction = torch.randn(
            self.dimension,
            dtype=self.center.dtype,
            device=self.center.device,
            generator=self.generator,
        )

        direction = direction / torch.linalg.vector_norm(direction)

        radius = self.epsilon * torch.rand(
            (),
            dtype=self.center.dtype,
            device=self.center.device,
            generator=self.generator,
        ).pow(1.0 / self.dimension)

        return (self.center.reshape(-1) + radius * direction).reshape_as(self.center)

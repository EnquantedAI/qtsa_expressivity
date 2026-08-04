from dataclasses import dataclass

@dataclass(slots=True)
class SpectrumResult:
    lambda_max: float
    lambda_min: float
    rank: int
    condition_number: float

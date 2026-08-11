import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
import torch

from projects.trainability_effective_dim.core.measures.ged import estimate_global_effective_dimension


@dataclass
class GEDDepthSweepResult:
    depths: list[int]
    means: list[float]
    standard_errors: list[float]
    samples: dict[int, list[float]]
    theoretical_dataset_size: int


def sweep_ged_by_depth(
    model_factory,
    inputs,
    depths,
    theoretical_dataset_size=1_000,
    n_theta=100,
    repetitions=5,
    normalize = True,
    seed=0,
    min_probability=1e-12,
):
    if theoretical_dataset_size <= 1:
        raise ValueError("theoretical_dataset_size must exceed 1")

    kappa = (
        theoretical_dataset_size
        / (2.0 * math.pi * math.log(theoretical_dataset_size))
    )

    if kappa <= 1.0:
        raise ValueError(
            "theoretical_dataset_size is too small: kappa must exceed 1"
        )

    depths = list(depths)

    if not depths:
        raise ValueError("depths must not be empty")

    samples = {}

    for depth in depths:
        depth_results = []

        for repetition in range(repetitions):
            torch.manual_seed(seed + 10_000 * depth + repetition)

            net = model_factory(depth)

            ged = estimate_global_effective_dimension(
                net=net,
                inputs=inputs,
                normalize=normalize,
                n_theta=n_theta,
                min_probability=min_probability,
                array_of_theoretical_number_of_data_samples=[
                    theoretical_dataset_size
                ],
            )

            depth_results.append(ged[0])

        samples[depth] = depth_results

    means = [
        float(torch.tensor(samples[depth], dtype=torch.float64).mean())
        for depth in depths
    ]

    standard_errors = [
        float(
            torch.tensor(samples[depth], dtype=torch.float64).std(
                unbiased=repetitions > 1
            )
            / math.sqrt(repetitions)
        )
        for depth in depths
    ]

    return GEDDepthSweepResult(
        depths=depths,
        means=means,
        standard_errors=standard_errors,
        samples=samples,
        theoretical_dataset_size=theoretical_dataset_size,
    )


def plot_ged_depth_sweep(result, output_path=None):
    figure, axis = plt.subplots(figsize=(7, 4))

    line = axis.plot(
        result.depths,
        result.means,
        marker="o",
        label="GED",
    )

    color = line[0].get_color()

    lower_bound = [
        mean - standard_error
        for mean, standard_error in zip(
            result.means,
            result.standard_errors,
        )
    ]

    upper_bound = [
        mean + standard_error
        for mean, standard_error in zip(
            result.means,
            result.standard_errors,
        )
    ]

    axis.fill_between(
        result.depths,
        lower_bound,
        upper_bound,
        color=color,
        alpha=0.2,
        label="± 1 standard error",
    )

    axis.set_xlabel("Circuit depth $L$")
    axis.set_ylabel("Normalized global effective dimension")
    axis.set_title(
        "Normalized GED as a function of circuit depth "
        f"(n = {result.theoretical_dataset_size})"
    )
    axis.set_xticks(result.depths)
    axis.grid(True, alpha=0.3)
    axis.legend()

    figure.tight_layout()

    if output_path is not None:
        figure.savefig(output_path, dpi=300)

    return figure, axis


def is_non_decreasing_with_tolerance(values, tolerance=0.05):
    values = torch.tensor(values, dtype=torch.float64)
    differences = values[1:] - values[:-1]

    return bool(torch.all(differences >= -tolerance))
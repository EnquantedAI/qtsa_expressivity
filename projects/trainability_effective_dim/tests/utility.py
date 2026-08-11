import matplotlib.pyplot as plt


def get_parameter_counts(model_factory, depths):
    """
    Return the number of trainable parameters for each circuit depth.

    Args:
        model_factory: Callable that creates a model for a given depth.
        depths: Iterable of circuit-depth values.

    Returns:
        list[int]: Parameter count corresponding to every depth.
    """
    return [
        model_factory(depth).parameter_count
        for depth in depths
    ]


def unnormalize_effective_dimension(
    normalized_values,
    normalized_errors,
    parameter_counts,
):
    """
    Convert parameter-count-normalized effective dimensions to raw values.

    Args:
        normalized_values: Effective dimensions divided by parameter count.
        normalized_errors: Standard errors of normalized values.
        parameter_counts: Parameter count for every evaluated depth.

    Returns:
        tuple[list[float], list[float]]:
            Raw effective dimensions and corresponding standard errors.
    """
    values = [
        value * parameter_count
        for value, parameter_count in zip(
            normalized_values,
            parameter_counts,
        )
    ]

    errors = [
        error * parameter_count
        for error, parameter_count in zip(
            normalized_errors,
            parameter_counts,
        )
    ]

    return values, errors


def normalize_effective_dimension(
    values,
    errors,
    parameter_counts,
):
    """
    Normalize effective dimensions by the parameter count at each depth.

    Args:
        values: Raw effective-dimension estimates.
        errors: Standard errors of raw effective-dimension estimates.
        parameter_counts: Parameter count for every evaluated depth.

    Returns:
        tuple[list[float], list[float]]:
            Parameter-count-normalized effective dimensions and errors.
    """
    normalized_values = [
        value / parameter_count
        for value, parameter_count in zip(
            values,
            parameter_counts,
        )
    ]

    normalized_errors = [
        error / parameter_count
        for error, parameter_count in zip(
            errors,
            parameter_counts,
        )
    ]

    return normalized_values, normalized_errors


def plot_effective_dimension_depth_sweep(
    led_result,
    ged_result,
    model_factory,
    output_path=None,
):
    """
    Plot raw and normalized LED/GED estimates against circuit depth.

    The left subplot shows raw effective dimensions. The right subplot shows
    parameter-count-normalized values. Shaded bands represent ±1 standard
    error across repeated Monte Carlo runs.

    Args:
        led_result: Result object returned by `sweep_led_by_depth`.
        ged_result: Result object returned by `sweep_ged_by_depth`.
        model_factory: Callable that creates a model for a given depth.
        output_path: Optional path used to save the generated figure.

    Returns:
        tuple:
            Matplotlib figure and axes objects.
    """
    def plot_with_uncertainty_band(axis, depths, means, errors, label, marker):
        line = axis.plot(
            depths,
            means,
            marker=marker,
            label=label,
        )

        color = line[0].get_color()

        lower_bound = [
            mean - error
            for mean, error in zip(means, errors)
        ]

        upper_bound = [
            mean + error
            for mean, error in zip(means, errors)
        ]

        axis.fill_between(
            depths,
            lower_bound,
            upper_bound,
            color=color,
            alpha=0.2,
        )

    depths = led_result.depths

    if depths != ged_result.depths:
        raise ValueError("LED and GED sweeps must use identical depth values.")

    parameter_counts = get_parameter_counts(
        model_factory=model_factory,
        depths=depths,
    )

    led_raw, led_raw_errors = unnormalize_effective_dimension(
        normalized_values=led_result.means,
        normalized_errors=led_result.standard_errors,
        parameter_counts=parameter_counts,
    )

    ged_raw, ged_raw_errors = unnormalize_effective_dimension(
        normalized_values=ged_result.means,
        normalized_errors=ged_result.standard_errors,
        parameter_counts=parameter_counts,
    )

    figure, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(15, 5),
        sharex=True,
    )

    plot_with_uncertainty_band(
        axis=axes[0],
        depths=depths,
        means=led_raw,
        errors=led_raw_errors,
        label="LED",
        marker="o",
    )

    plot_with_uncertainty_band(
        axis=axes[0],
        depths=depths,
        means=ged_raw,
        errors=ged_raw_errors,
        label="GED",
        marker="s",
    )

    axes[0].set_title("Un-normalized effective dimension")
    axes[0].set_xlabel("Circuit depth $L$")
    axes[0].set_ylabel("Effective dimension")
    axes[0].set_xticks(depths)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    plot_with_uncertainty_band(
        axis=axes[1],
        depths=depths,
        means=led_result.means,
        errors=led_result.standard_errors,
        label="LED",
        marker="o",
    )

    plot_with_uncertainty_band(
        axis=axes[1],
        depths=depths,
        means=ged_result.means,
        errors=ged_result.standard_errors,
        label="GED",
        marker="s",
    )

    axes[1].set_title("Parameter-count-normalized effective dimension")
    axes[1].set_xlabel("Circuit depth $L$")
    axes[1].set_ylabel("Effective dimension / parameter count")
    axes[1].set_xticks(depths)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    figure.suptitle(
        "Local and global effective dimension versus circuit depth",
        fontsize=14,
    )

    figure.tight_layout()

    if output_path is not None:
        figure.savefig(output_path, dpi=300)

    return figure, axes

def print_effective_dimension_depth_sweep(
    led_result,
    ged_result,
    model_factory,
):
    """
    Print raw and normalized LED/GED values for every circuit depth.

    Args:
        led_result: Result object returned by `sweep_led_by_depth`.
        ged_result: Result object returned by `sweep_ged_by_depth`.
        model_factory: Callable that creates a model for a given depth.
    """
    depths = led_result.depths

    if depths != ged_result.depths:
        raise ValueError("LED and GED sweeps must use identical depth values.")

    parameter_counts = get_parameter_counts(
        model_factory=model_factory,
        depths=depths,
    )

    led_raw, led_raw_errors = unnormalize_effective_dimension(
        normalized_values=led_result.means,
        normalized_errors=led_result.standard_errors,
        parameter_counts=parameter_counts,
    )

    ged_raw, ged_raw_errors = unnormalize_effective_dimension(
        normalized_values=ged_result.means,
        normalized_errors=ged_result.standard_errors,
        parameter_counts=parameter_counts,
    )

    for values in zip(
        depths,
        parameter_counts,
        led_raw,
        led_raw_errors,
        ged_raw,
        ged_raw_errors,
        led_result.means,
        led_result.standard_errors,
        ged_result.means,
        ged_result.standard_errors,
    ):
        (
            depth,
            parameter_count,
            led_value,
            led_error,
            ged_value,
            ged_error,
            led_normalized,
            led_normalized_error,
            ged_normalized,
            ged_normalized_error,
        ) = values

        print(
            f"L={depth:2d} | "
            f"d={parameter_count:3d} | "
            f"LED={led_value:.4f} ± {led_error:.4f} | "
            f"GED={ged_value:.4f} ± {ged_error:.4f} | "
            f"LED/d={led_normalized:.4f} ± {led_normalized_error:.4f} | "
            f"GED/d={ged_normalized:.4f} ± {ged_normalized_error:.4f}"
        )
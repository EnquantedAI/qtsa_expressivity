from pathlib import Path

from .study import duplication_study, sampling_density_study, save_results


def main():
    density = sampling_density_study()
    duplication = duplication_study()

    for row in density:
        print(
            f"m={row['sample_count']:>2}: "
            f"equal={row['d_tp_equal']:.6f}, "
            f"weighted={row['d_tp_trapezoidal']:.6f}"
        )

    print("duplication:", duplication)
    save_results(Path(__file__).parent / "results", density, duplication)


if __name__ == "__main__":
    main()

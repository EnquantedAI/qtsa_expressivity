from pathlib import Path

from .study import run_real_data_study, save_results


def main():
    rows, summary, growth_rows, growth_summary, metadata = run_real_data_study()
    output_dir = Path(__file__).resolve().parent / "results"
    save_results(output_dir, rows, summary, growth_rows, growth_summary, metadata)
    print(f"saved {len(rows)} observations to {output_dir}")
    print(f"datasets: {', '.join(metadata['datasets'])}; split={metadata['split']}")
    print("weights are untrained; this is a data/architecture diagnostic, not a forecasting benchmark")


if __name__ == "__main__":
    main()

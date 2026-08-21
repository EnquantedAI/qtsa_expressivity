from pathlib import Path

from .suite import run_validation_suite, save_report


def main():
    report = run_validation_suite()
    output_dir = Path(__file__).resolve().parent / "results"
    save_report(output_dir, report)

    summary = report["summary"]
    print(f"hard checks: {summary['hard_passed']}/{summary['hard_checks']} passed")
    print(f"diagnostics: {summary['diagnostics']}")
    for row in report["checks"]:
        label = row["status"].upper()
        print(f"{label:4s}  {row['name']}")


if __name__ == "__main__":
    main()

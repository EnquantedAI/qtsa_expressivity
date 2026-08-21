import tempfile
import unittest
from pathlib import Path

from projects.expr_train_theory.trajectory_participation.validation_suite.suite import (
    boundary_checks,
    run_validation_suite,
    sampling_checks,
    save_report,
)


class ValidationSuiteTests(unittest.TestCase):
    def test_boundary_hard_checks_pass(self):
        checks = boundary_checks()
        hard = [row for row in checks if row["kind"] == "hard"]
        self.assertTrue(hard)
        self.assertTrue(all(row["status"] == "pass" for row in hard))

    def test_sampling_contains_weighted_duplicate_check(self):
        checks = sampling_checks()
        row = next(r for r in checks if r["name"] == "weighted duplicate correction")
        self.assertEqual(row["status"], "pass")

    def test_full_suite_has_no_failed_hard_checks(self):
        report = run_validation_suite(seed=123)
        self.assertEqual(report["summary"]["hard_failed"], 0)
        self.assertGreater(report["summary"]["diagnostics"], 0)

    def test_report_files_are_written(self):
        report = run_validation_suite(seed=123)
        with tempfile.TemporaryDirectory() as tmp:
            save_report(tmp, report)
            self.assertTrue((Path(tmp) / "validation_report.json").is_file())
            self.assertTrue((Path(tmp) / "validation_report.md").is_file())


if __name__ == "__main__":
    unittest.main()

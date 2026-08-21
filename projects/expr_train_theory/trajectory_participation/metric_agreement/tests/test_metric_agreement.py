import tempfile
import unittest

from projects.expr_train_theory.trajectory_participation.metric_agreement.agreement import (
    build_agreement_report,
    compare_profile,
    save_report,
    summarize_report,
)
from projects.expr_train_theory.trajectory_participation.metric_profile.profile import (
    MetricProfile,
    default_profiles,
)


class MetricAgreementTests(unittest.TestCase):
    def test_default_cases_have_expected_patterns(self):
        rows = {row.name: row for row in build_agreement_report()}
        self.assertEqual(rows["redundant_ry"].visibility_pattern, "dTP+QFIM+CFIM+QNTK")
        self.assertEqual(rows["phase_only"].visibility_pattern, "dTP+QFIM")
        self.assertEqual(rows["mixed_ry_rz"].visibility_pattern, "dTP+QFIM+CFIM+QNTK")

    def test_phase_only_separates_state_and_output_metrics(self):
        rows = {row.name: row for row in build_agreement_report()}
        row = rows["phase_only"]
        self.assertTrue(row.state_space_agreement)
        self.assertTrue(row.output_space_agreement)
        self.assertFalse(row.all_active_agreement)
        self.assertEqual(row.qfim_cfim_rank_gap, 1)
        self.assertEqual(row.qfim_qntk_rank_gap, 1)

    def test_all_inactive_profile(self):
        profile = MetricProfile(
            name="inactive",
            parameter_count=1,
            snapshot_count=1,
            d_tp=1.0,
            trajectory_rank=1,
            qfim_rank=0,
            cfim_rank=0,
            qntk_rank=0,
            qfim_trace=0.0,
            cfim_trace=0.0,
            qntk_trace=0.0,
            qntk_effective_rank=0.0,
        )
        row = compare_profile(profile)
        self.assertEqual(row.visibility_pattern, "none")
        self.assertTrue(row.all_active_agreement)

    def test_summary_counts_patterns(self):
        rows = build_agreement_report(default_profiles())
        summary = summarize_report(rows)
        self.assertEqual(summary["case_count"], 3)
        self.assertEqual(summary["patterns"]["dTP+QFIM"], 1)
        self.assertEqual(summary["patterns"]["dTP+QFIM+CFIM+QNTK"], 2)

    def test_report_can_be_saved(self):
        rows = build_agreement_report()
        with tempfile.TemporaryDirectory() as tmp:
            csv_path, json_path = save_report(tmp, rows)
            self.assertTrue(csv_path.exists())
            self.assertTrue(json_path.exists())


if __name__ == "__main__":
    unittest.main()

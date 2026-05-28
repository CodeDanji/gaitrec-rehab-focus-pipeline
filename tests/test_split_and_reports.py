import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gait_rehab.modeling import split_by_subject
from gait_rehab.reporting import (
    FORBIDDEN_REPORT_TERMS,
    build_rehab_focus_report,
    generate_final_analysis_report,
)


class SplitAndReportTests(unittest.TestCase):
    def test_subject_split_does_not_leak_subjects(self):
        df = pd.DataFrame(
            {
                "subject_id": ["s1", "s1", "s2", "s3", "s4", "s5"],
                "trial_id": ["t1", "t2", "t1", "t1", "t1", "t1"],
                "label": ["Healthy", "Healthy", "Hip", "Knee", "Ankle", "Calcaneus"],
            }
        )

        train_df, test_df = split_by_subject(df, test_size=0.4, random_state=7)

        train_subjects = set(train_df["subject_id"])
        test_subjects = set(test_df["subject_id"])
        self.assertTrue(train_subjects.isdisjoint(test_subjects))
        self.assertEqual(len(train_df) + len(test_df), len(df))

    def test_rehab_focus_report_has_evidence_candidates_and_no_forbidden_terms(self):
        report = build_rehab_focus_report(
            subject_id="S001",
            trial_id="T003",
            predicted_label="Ankle",
            evidence=[
                ("push_off_index", -1.42, "affected side push-off reduction"),
                ("cop_ml_range_aff", 1.18, "larger medial-lateral COP movement"),
                ("vgrf_peak_asym", -0.93, "lower affected-side vertical loading"),
            ],
            candidates=[
                "ankle push-off function",
                "weight-bearing avoidance strategy",
            ],
        )

        for feature_name, _, _ in [
            ("push_off_index", -1.42, ""),
            ("cop_ml_range_aff", 1.18, ""),
            ("vgrf_peak_asym", -0.93, ""),
        ]:
            self.assertIn(feature_name, report)

        self.assertIn("ankle push-off function", report)
        self.assertIn("weight-bearing avoidance strategy", report)

        lowered = report.lower()
        for term in FORBIDDEN_REPORT_TERMS:
            self.assertNotIn(term.lower(), lowered)

    def test_final_report_does_not_require_optional_tabulate_dependency(self):
        metrics = pd.DataFrame(
            [
                {
                    "model": "dummy",
                    "balanced_accuracy": 0.2,
                    "macro_f1": 0.1,
                    "support": 5,
                    "used_sklearn": False,
                }
            ]
        )
        group_summary = pd.DataFrame(
            [
                {
                    "label": "Ankle",
                    "feature": "push_off_index",
                    "mean": 1.2,
                    "std": 0.3,
                    "n": 4,
                }
            ]
        )
        subject_counts = pd.DataFrame(
            [{"label": "Ankle", "subject_count": 2, "trial_count": 4}]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "final_analysis_report.md"
            generate_final_analysis_report(output_path, metrics, group_summary, subject_counts)

            self.assertTrue(output_path.exists())
            self.assertIn("Final Analysis Report", output_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()

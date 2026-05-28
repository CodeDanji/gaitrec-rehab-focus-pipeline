import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gait_rehab.pipeline import ProjectConfig, run_demo_pipeline


class PipelineOutputContractTests(unittest.TestCase):
    def test_demo_pipeline_writes_full_pipeline_output_contract(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir) / "results"

            run_demo_pipeline(ProjectConfig(output_root=output_root, random_state=5, test_size=0.25))

            required_paths = [
                "tables/gaitrec_features.csv",
                "tables/model_metrics.csv",
                "tables/group_feature_summary.csv",
                "tables/permutation_importance.csv",
                "tables/logistic_coefficients.csv",
                "figures/workflow.svg",
                "figures/model_metrics.svg",
                "figures/confusion_matrix.svg",
                "figures/permutation_importance.svg",
                "figures/group_mean_vgrf_curve.svg",
                "figures/group_ap_impulse_comparison.svg",
                "figures/group_cop_comparison.svg",
                "reports/final_analysis_report.md",
                "reports/example_subject_rehab_focus.md",
                "reports/siat_reference_note.md",
            ]
            for relative_path in required_paths:
                self.assertTrue((output_root / relative_path).exists(), relative_path)

            report = (output_root / "reports" / "final_analysis_report.md").read_text(encoding="utf-8")
            for section in [
                "## Available/Unavailable Features",
                "## Model Interpretation",
                "## SIAT Reference Note",
            ]:
                self.assertIn(section, report)


if __name__ == "__main__":
    unittest.main()

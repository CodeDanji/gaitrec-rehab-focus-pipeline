import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gait_rehab.modeling import available_feature_columns
from scripts.inspect_siat import inspect_siat_root


class SiatReferenceTests(unittest.TestCase):
    def test_siat_inspection_writes_inventory_candidates_and_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            siat_root = root / "siat"
            output_root = root / "inspection"
            siat_root.mkdir()
            pd.DataFrame(
                {
                    "gait_phase": [0, 10, 20],
                    "EMG_TA": [0.1, 0.2, 0.3],
                    "ankle_torque": [1.0, 1.1, 1.2],
                }
            ).to_csv(siat_root / "walking_trial_emg_torque.csv", index=False)

            inspect_siat_root(siat_root, output_root)

            inventory = pd.read_csv(output_root / "tables" / "siat_file_inventory.csv")
            candidates = pd.read_csv(output_root / "tables" / "siat_column_candidates.csv")
            report = (output_root / "reports" / "siat_structure_report.md").read_text(encoding="utf-8")

            self.assertEqual(inventory.iloc[0]["relative_path"], "walking_trial_emg_torque.csv")
            self.assertIn("gait_phase", set(candidates["candidate_type"]))
            self.assertIn("emg", set(candidates["candidate_type"]))
            self.assertIn("joint_torque", set(candidates["candidate_type"]))
            self.assertIn("not merged into the GaitRec classifier", report)

    def test_empty_siat_root_still_writes_empty_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir) / "inspection"

            inspect_siat_root(Path(tmpdir) / "missing", output_root)

            self.assertTrue((output_root / "tables" / "siat_file_inventory.csv").exists())
            self.assertTrue((output_root / "tables" / "siat_column_candidates.csv").exists())
            report = (output_root / "reports" / "siat_structure_report.md").read_text(encoding="utf-8")
            self.assertIn("No SIAT files were found", report)

    def test_siat_columns_are_not_model_features(self):
        frame = pd.DataFrame(
            {
                "vgrf_peak_aff": [1.0],
                "EMG_TA": [0.2],
                "ankle_torque": [1.1],
                "siat_reference_score": [0.5],
            }
        )

        self.assertEqual(available_feature_columns(frame), ["vgrf_peak_aff"])


if __name__ == "__main__":
    unittest.main()

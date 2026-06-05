import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

class SiatReferenceTests(unittest.TestCase):
    def test_discover_siat_pairs_matches_wak_data_and_label_files(self):
        from gait_rehab.siat import discover_siat_pairs

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_dir = root / "Sub01" / "Data"
            label_dir = root / "Sub01" / "Labels"
            data_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            pd.DataFrame({"Time": [0.0], "sEMG: soleus": [0.1]}).to_csv(
                data_dir / "Sub01_WAK_Data.csv",
                index=False,
            )
            pd.DataFrame({"Time": [0.0], "Status": [1], "Group": [1]}).to_csv(
                label_dir / "Sub01_WAK_Label.csv",
                index=False,
            )

            pairs = discover_siat_pairs(root)

        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs.iloc[0]["subject_id"], "Sub01")
        self.assertEqual(pairs.iloc[0]["task"], "WAK")
        self.assertTrue(str(pairs.iloc[0]["data_path"]).endswith("Sub01_WAK_Data.csv"))
        self.assertTrue(str(pairs.iloc[0]["label_path"]).endswith("Sub01_WAK_Label.csv"))

    def test_validate_siat_data_schema_accepts_named_wak_blocks(self):
        from gait_rehab.siat import validate_siat_data_schema

        frame = pd.DataFrame(
            {
                "Time": [0.0],
                **{f"Kinematic: joint {index} angle": [0.0] for index in range(8)},
                **{f"Kinetic: joint {index} torque": [0.0] for index in range(8)},
                **{f"sEMG: muscle {index}": [0.0] for index in range(9)},
            }
        )

        schema = validate_siat_data_schema(frame, Path("Sub01_WAK_Data.csv"))

        self.assertEqual(schema.time_column, "Time")
        self.assertEqual(len(schema.kinematic_columns), 8)
        self.assertEqual(len(schema.kinetic_columns), 8)
        self.assertEqual(len(schema.emg_columns), 9)
        self.assertFalse(schema.schema_inferred_from_position)

    def test_validate_wak_label_schema_requires_time_status_group(self):
        from gait_rehab.siat_labels import validate_wak_label_schema

        labels = pd.DataFrame({"Time": [0.0], "Status": [1], "Group": [1]})

        schema = validate_wak_label_schema(labels, Path("Sub01_WAK_Label.csv"))

        self.assertEqual(schema.time_column, "Time")
        self.assertEqual(schema.status_column, "Status")
        self.assertEqual(schema.group_column, "Group")

    def test_validate_wak_label_schema_rejects_missing_status_column(self):
        from gait_rehab.siat_labels import validate_wak_label_schema

        labels = pd.DataFrame({"Time": [0.0], "label": ["HS"]})

        with self.assertRaisesRegex(ValueError, "Time, Status, Group"):
            validate_wak_label_schema(labels, Path("Sub01_WAK_Label.csv"))

    def test_join_wak_data_and_labels_reports_quality_and_drops_invalid_status(self):
        from gait_rehab.siat_labels import join_wak_data_and_labels

        data = pd.DataFrame(
            {
                "Time": [0.0, 0.1, 0.2],
                "sEMG: soleus": [0.1, 0.2, 0.3],
                "Kinetic: left ankle flexion torque": [1.0, 1.1, 1.2],
            }
        )
        labels = pd.DataFrame(
            {
                "Time": [0.0, 0.10004, 0.2],
                "Status": [1, "NaN", 5],
                "Group": [1, 1, 1],
            }
        )

        joined, quality = join_wak_data_and_labels(
            data,
            labels,
            subject_id="Sub01",
            trial_id="Sub01_WAK",
            time_tolerance_sec=0.0001,
        )

        self.assertEqual(len(joined), 2)
        self.assertEqual(joined["phase_interval"].tolist(), ["HS-MSF", "MWF-HS"])
        self.assertEqual(int(quality.iloc[0]["invalid_status_rows"]), 1)
        self.assertLessEqual(float(quality.iloc[0]["max_time_diff_sec"]), 0.0001)

    def test_map_wak_status_to_phase_intervals_and_functional_phases(self):
        from gait_rehab.siat_labels import map_wak_status

        mapped = [map_wak_status(value) for value in [1, 2, 3, 4, 5]]

        self.assertEqual(
            [item.phase_interval for item in mapped],
            ["HS-MSF", "MSF-MSE", "MSE-TO", "TO-MWF", "MWF-HS"],
        )
        self.assertEqual(mapped[0].functional_phase, "loading_response")
        self.assertEqual(mapped[3].functional_phase, "push_off_to_swing_transition")

    def test_compute_wak_window_quality_counts_constant_status_group_windows(self):
        from gait_rehab.siat_atlas import compute_wak_window_quality

        samples = pd.DataFrame(
            {
                "subject_id": ["Sub01"] * 6,
                "trial_id": ["Sub01_WAK"] * 6,
                "phase_interval": ["HS-MSF", "HS-MSF", "HS-MSF", "MSF-MSE", "MSF-MSE", "MSF-MSE"],
                "Group": [1, 1, 1, 1, 1, 1],
            }
        )

        quality = compute_wak_window_quality(samples, window_size=2, overlap=1)

        self.assertEqual(int(quality.iloc[0]["potential_windows"]), 5)
        self.assertEqual(int(quality.iloc[0]["accepted_windows"]), 4)
        self.assertEqual(int(quality.iloc[0]["dropped_windows"]), 1)

    def test_siat_wak_atlas_requires_phase_subject_coverage_threshold(self):
        from gait_rehab.siat_atlas import validate_wak_atlas_coverage

        coverage = pd.DataFrame(
            {
                "phase_interval": ["HS-MSF", "MSF-MSE"],
                "subject_count": [2, 1],
                "valid_status_rate": [0.90, 0.82],
            }
        )

        with self.assertRaisesRegex(ValueError, "minimum subject coverage"):
            validate_wak_atlas_coverage(coverage, min_subjects_per_phase=2, min_valid_status_rate=0.85)

    def test_siat_wak_atlas_uses_subject_level_aggregation_not_row_level_average(self):
        from gait_rehab.siat_atlas import build_siat_wak_reference_atlas

        samples = pd.DataFrame(
            {
                "subject_id": ["Sub01"] * 10 + ["Sub02"] * 2,
                "trial_id": ["t1"] * 12,
                "task": ["WAK"] * 12,
                "phase_interval": ["HS-MSF"] * 12,
                "functional_phase": ["loading_response"] * 12,
                "sEMG: soleus": [10.0] * 10 + [0.0] * 2,
                "Kinetic: left ankle flexion torque": [2.0] * 10 + [0.0] * 2,
            }
        )

        atlas = build_siat_wak_reference_atlas(samples)
        emg = atlas["siat_wak_emg_phase_summary"]
        row = emg.loc[emg["channel"].eq("sEMG: soleus")].iloc[0]

        self.assertAlmostEqual(row["mean"], 5.0)
        self.assertEqual(row["subject_count"], 2)

    def test_siat_wak_atlas_reports_peak_timing_and_emg_torque_lag(self):
        from gait_rehab.siat_atlas import build_siat_wak_reference_atlas

        samples = pd.DataFrame(
            {
                "subject_id": ["Sub01"] * 5,
                "trial_id": ["t1"] * 5,
                "task": ["WAK"] * 5,
                "Time": [0.0, 0.1, 0.2, 0.3, 0.4],
                "phase_interval": ["HS-MSF", "MSF-MSE", "MSE-TO", "TO-MWF", "MWF-HS"],
                "functional_phase": ["loading_response", "mid_stance_control", "terminal_stance", "push_off_to_swing_transition", "swing_recovery"],
                "sEMG: lateral gastrocnemius": [0.1, 0.2, 0.4, 0.9, 0.3],
                "Kinetic: left ankle flexion torque": [0.2, 0.3, 0.5, 1.2, 0.4],
            }
        )

        atlas = build_siat_wak_reference_atlas(samples)
        peaks = atlas["siat_wak_peak_timing"]

        self.assertIn("peak_time", peaks.columns)
        self.assertIn("peak_phase_interval", peaks.columns)
        self.assertIn("siat_wak_emg_torque_lag", atlas)
        self.assertFalse(atlas["siat_wak_emg_torque_lag"].empty)

    def test_functional_domain_reporting_requires_provenance(self):
        from gait_rehab.reporting import generate_functional_interpretation_summary

        with self.assertRaisesRegex(ValueError, "Missing GaitRec provenance"):
            generate_functional_interpretation_summary(
                siat_atlas={},
                gaitrec_results={"feature_importance": {"vgrf_peak_aff": 0.3}},
                provenance={}
            )

    def test_functional_domain_reporting_contains_interpretation_only_guardrails(self):
        from gait_rehab.reporting import generate_functional_interpretation_summary

        report = generate_functional_interpretation_summary(
            siat_atlas={},
            gaitrec_results={"feature_importance": {"vgrf_peak_aff": 0.3}},
            provenance={"source_branch": "main", "source_commit": "abc", "run_id": "123"}
        )

        self.assertIn("본 프로젝트는 특정 근육의 약화나 통증 원인을 확정하지 않습니다", report)
        self.assertIn("SIAT Reference는 정상군 기준을 제시할 뿐, 진단이나 처방을 위한 용도가 아닙니다", report)
        
        forbidden = ["진단 모델", "질환 여부 판단", "근육 약화 확진"]
        for f in forbidden:
            self.assertNotIn(f, report)

if __name__ == '__main__':
    unittest.main()







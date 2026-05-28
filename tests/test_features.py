import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gait_rehab.features import (
    asymmetry,
    braking_impulse,
    cop_path_length,
    impulse,
    loading_rate,
    peak,
    propulsion_impulse,
    signal_range,
)


class FeatureFormulaTests(unittest.TestCase):
    def test_signal_summary_formulas_match_synthetic_values(self):
        signal = np.array([0.0, 1.0, 3.0, 2.0])

        self.assertEqual(peak(signal), 3.0)
        self.assertEqual(signal_range(signal), 3.0)
        self.assertTrue(math.isclose(impulse(signal), 6.0))
        self.assertTrue(math.isclose(asymmetry(3.0, 1.0), 1.0))

    def test_loading_and_ap_impulses_use_expected_gait_regions(self):
        vgrf = np.array([0.0, 2.0, 5.0, 4.0, 3.0])
        ap_grf = np.array([-2.0, -1.0, 0.0, 3.0, 2.0])

        self.assertTrue(math.isclose(loading_rate(vgrf), 5.0))
        self.assertTrue(math.isclose(braking_impulse(ap_grf), 3.0))
        self.assertTrue(math.isclose(propulsion_impulse(ap_grf), 5.0))

    def test_cop_path_length_sums_stepwise_distances(self):
        cop_ap = np.array([0.0, 3.0, 3.0])
        cop_ml = np.array([0.0, 4.0, 8.0])

        self.assertTrue(math.isclose(cop_path_length(cop_ap, cop_ml), 9.0))

    def test_extracts_rows_from_gaitrec_session_trial_signal_format_with_vertical_only(self):
        from gait_rehab.features import extract_gait_features

        metadata = pd.DataFrame(
            [
                {
                    "SUBJECT_ID": 510,
                    "SESSION_ID": 413,
                    "CLASS_LABEL": "K",
                    "AFFECTED_SIDE": "L",
                    "SPEED": 2,
                    "AGE": 47,
                    "SEX": 1,
                    "HEIGHT": 170,
                    "BODY_MASS": 70.0,
                    "SHOD_CONDITION": 1,
                }
            ]
        )
        signals = {
            "vgrf_left": pd.DataFrame(
                [
                    {
                        "SUBJECT_ID": 510,
                        "SESSION_ID": 413,
                        "TRIAL_ID": 1,
                        "F_V_PRO_1": 0.0,
                        "F_V_PRO_2": 1.0,
                        "F_V_PRO_3": 2.0,
                    }
                ]
            ),
            "vgrf_right": pd.DataFrame(
                [
                    {
                        "SUBJECT_ID": 510,
                        "SESSION_ID": 413,
                        "TRIAL_ID": 1,
                        "F_V_PRO_1": 0.0,
                        "F_V_PRO_2": 1.5,
                        "F_V_PRO_3": 3.0,
                    }
                ]
            ),
        }

        features = extract_gait_features(metadata, signals)

        self.assertEqual(len(features), 1)
        row = features.iloc[0]
        self.assertEqual(row["subject_id"], "510")
        self.assertEqual(row["session_id"], "413")
        self.assertEqual(row["trial_id"], "413_1")
        self.assertEqual(row["label"], "Knee")
        self.assertEqual(row["affected_side"], "left")
        self.assertEqual(row["vgrf_peak_aff"], 2.0)
        self.assertEqual(row["vgrf_peak_unaff"], 3.0)
        self.assertTrue(pd.isna(row["push_off_index"]))
        self.assertTrue(pd.isna(row["cop_path_length_aff"]))


if __name__ == "__main__":
    unittest.main()

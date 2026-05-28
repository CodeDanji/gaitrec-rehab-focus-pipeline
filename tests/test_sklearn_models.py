import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gait_rehab.demo_data import make_demo_gaitrec
from gait_rehab.features import extract_gait_features
from gait_rehab.modeling import available_feature_columns, split_by_subject, train_models


class SklearnModelPathTests(unittest.TestCase):
    def test_sklearn_models_are_default_when_dependency_is_available(self):
        try:
            import sklearn  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("scikit-learn is not installed in this Python environment")

        metadata, signals = make_demo_gaitrec(subjects_per_label=3, trials_per_subject=1, random_state=11)
        features = extract_gait_features(metadata, signals)
        train_df, test_df = split_by_subject(features, test_size=0.25, random_state=3)

        bundle = train_models(train_df, random_state=3)

        self.assertTrue(bundle.used_sklearn)
        self.assertEqual(set(bundle.models), {"dummy", "logistic_regression", "random_forest"})
        self.assertTrue(set(train_df["subject_id"]).isdisjoint(set(test_df["subject_id"])))

    def test_model_feature_columns_exclude_all_null_optional_features(self):
        features = pd.DataFrame(
            {
                "vgrf_peak_aff": [1.0, 1.2],
                "push_off_index": [np.nan, np.nan],
                "cop_path_length_aff": [np.nan, np.nan],
                "sex": ["M", "F"],
                "shoe_condition": [None, None],
            }
        )

        self.assertEqual(available_feature_columns(features), ["vgrf_peak_aff", "sex"])


if __name__ == "__main__":
    unittest.main()

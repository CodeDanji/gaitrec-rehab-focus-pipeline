import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_smoke_subset import build_smoke_subset


class SmokeSubsetTests(unittest.TestCase):
    def test_subset_preserves_selected_keys_across_signal_files_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source"
            output = root / "smoke"
            source.mkdir()
            labels = ["Healthy", "Hip", "Knee", "Ankle", "Calcaneus"]
            metadata_rows = []
            signal_rows = []
            for index, label in enumerate(labels):
                subject_id = f"S{index}"
                session_id = f"SE{index}"
                metadata_rows.append(
                    {
                        "SUBJECT_ID": subject_id,
                        "SESSION_ID": session_id,
                        "CLASS_LABEL": label,
                        "AFFECTED_SIDE": "L",
                    }
                )
                for trial_id in ["T1", "T2"]:
                    signal_rows.append(
                        {
                            "SUBJECT_ID": subject_id,
                            "SESSION_ID": session_id,
                            "TRIAL_ID": trial_id,
                            "sample_1": float(index),
                            "sample_2": float(index + 1),
                        }
                    )

            pd.DataFrame(metadata_rows).to_csv(source / "GRF_metadata.csv", index=False)
            for filename in [
                "GRF_F_V_PRO_left.csv",
                "GRF_F_V_PRO_right.csv",
                "GRF_F_AP_PRO_left.csv",
                "GRF_F_AP_PRO_right.csv",
                "GRF_COP_AP_PRO_left.csv",
                "GRF_COP_AP_PRO_right.csv",
                "GRF_COP_ML_PRO_left.csv",
                "GRF_COP_ML_PRO_right.csv",
            ]:
                pd.DataFrame(signal_rows).to_csv(source / filename, index=False)

            build_smoke_subset(
                input_root=source,
                output_root=output,
                max_bytes=1_000_000,
                seed=7,
                min_subjects_per_label=1,
                min_trials_per_label=1,
                include_ml_grf=False,
            )

            manifest = json.loads((output / "smoke_sampling_manifest.json").read_text(encoding="utf-8"))
            self.assertLessEqual(manifest["output_size_bytes"], 1_000_000)
            self.assertEqual(set(manifest["selected_subject_count_by_label"]), set(labels))

            left_keys = _keys(pd.read_csv(output / "GRF_F_V_PRO_left.csv"))
            for filename in [
                "GRF_F_V_PRO_right.csv",
                "GRF_F_AP_PRO_left.csv",
                "GRF_F_AP_PRO_right.csv",
                "GRF_COP_AP_PRO_left.csv",
                "GRF_COP_AP_PRO_right.csv",
                "GRF_COP_ML_PRO_left.csv",
                "GRF_COP_ML_PRO_right.csv",
            ]:
                self.assertEqual(left_keys, _keys(pd.read_csv(output / filename)))


def _keys(df: pd.DataFrame) -> set[tuple[str, str, str]]:
    return set(zip(df["SUBJECT_ID"].astype(str), df["SESSION_ID"].astype(str), df["TRIAL_ID"].astype(str)))


if __name__ == "__main__":
    unittest.main()

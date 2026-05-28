import sys
import tempfile
import unittest
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gait_rehab.data import load_gaitrec_processed_signals


class GaitRecDataLoaderTests(unittest.TestCase):
    def test_loads_vertical_only_subset_without_requiring_all_processed_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "GRF_F_V_PRO_left.csv").write_text(
                "SUBJECT_ID,SESSION_ID,TRIAL_ID,F_V_PRO_1\n1,10,1,0.1\n",
                encoding="utf-8",
            )
            (root / "GRF_F_V_PRO_right.csv").write_text(
                "SUBJECT_ID,SESSION_ID,TRIAL_ID,F_V_PRO_1\n1,10,1,0.2\n",
                encoding="utf-8",
            )

            signals = load_gaitrec_processed_signals(root)

            self.assertEqual(set(signals), {"vgrf_left", "vgrf_right"})

    def test_loads_signal_files_from_manifest_roles(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "left_signal.csv").write_text(
                "SUBJECT_ID,SESSION_ID,TRIAL_ID,s1\n1,10,1,0.1\n",
                encoding="utf-8",
            )
            (root / "right_signal.csv").write_text(
                "SUBJECT_ID,SESSION_ID,TRIAL_ID,s1\n1,10,1,0.2\n",
                encoding="utf-8",
            )
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "dataset": "gaitrec",
                        "version": "test",
                        "source_collection_id": 1,
                        "source_collection_url": "https://example.test",
                        "files": [
                            {
                                "role": "vgrf_left",
                                "required_for": ["smoke-source"],
                                "article_id": 1,
                                "file_id": 1,
                                "filename": "left_signal.csv",
                                "size_bytes": 0,
                                "download_url": "https://example.test/left",
                                "target_path": "left_signal.csv",
                                "sha256": None,
                            },
                            {
                                "role": "vgrf_right",
                                "required_for": ["smoke-source"],
                                "article_id": 1,
                                "file_id": 2,
                                "filename": "right_signal.csv",
                                "size_bytes": 0,
                                "download_url": "https://example.test/right",
                                "target_path": "right_signal.csv",
                                "sha256": None,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            signals = load_gaitrec_processed_signals(root, manifest_path=manifest_path)

            self.assertEqual(set(signals), {"vgrf_left", "vgrf_right"})
            self.assertEqual(float(signals["vgrf_right"].iloc[0]["s1"]), 0.2)


if __name__ == "__main__":
    unittest.main()

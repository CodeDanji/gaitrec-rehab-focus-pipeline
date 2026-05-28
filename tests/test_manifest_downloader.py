import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gait_rehab.manifest import (
    ManifestError,
    download_manifest_files,
    load_manifest,
    select_manifest_files,
)


class ManifestDownloaderTests(unittest.TestCase):
    def test_project_manifest_has_under_500mb_vertical_subset(self):
        manifest_path = Path(__file__).resolve().parents[1] / "config" / "gaitrec_processed_manifest.json"
        manifest = load_manifest(manifest_path)

        selected = select_manifest_files(manifest, "vertical-500mb")

        self.assertEqual([item["role"] for item in selected], ["metadata", "vgrf_left", "vgrf_right"])
        self.assertLessEqual(sum(int(item["size_bytes"]) for item in selected), 500_000_000)

    def test_manifest_schema_and_set_filtering(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "dataset": "gaitrec",
                        "version": "processed-v1",
                        "source_collection_id": 4788012,
                        "source_collection_url": "https://example.test/collection",
                        "files": [
                            {
                                "role": "metadata",
                                "required_for": ["smoke-source", "full"],
                                "article_id": 1,
                                "file_id": 10,
                                "filename": "GRF_metadata.csv",
                                "size_bytes": 12,
                                "download_url": "https://example.test/meta",
                                "target_path": "GRF_metadata.csv",
                                "sha256": None,
                            },
                            {
                                "role": "ml_grf_left",
                                "required_for": ["smoke-v2", "full"],
                                "article_id": 2,
                                "file_id": 20,
                                "filename": "GRF_F_ML_PRO_left.csv",
                                "size_bytes": 34,
                                "download_url": "https://example.test/ml",
                                "target_path": "GRF_F_ML_PRO_left.csv",
                                "sha256": None,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            manifest = load_manifest(manifest_path)
            selected = select_manifest_files(manifest, "smoke-source")

            self.assertEqual([item["role"] for item in selected], ["metadata"])

    def test_existing_file_size_mismatch_fails_without_overwrite(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "dataset": "gaitrec",
                        "version": "processed-v1",
                        "source_collection_id": 4788012,
                        "source_collection_url": "https://example.test/collection",
                        "files": [
                            {
                                "role": "metadata",
                                "required_for": ["smoke-source"],
                                "article_id": 1,
                                "file_id": 10,
                                "filename": "GRF_metadata.csv",
                                "size_bytes": 10,
                                "download_url": "https://example.test/meta",
                                "target_path": "GRF_metadata.csv",
                                "sha256": None,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            output_root = root / "out"
            output_root.mkdir()
            (output_root / "GRF_metadata.csv").write_text("short", encoding="utf-8")

            with self.assertRaisesRegex(ManifestError, "size mismatch"):
                download_manifest_files(
                    manifest_path=manifest_path,
                    dataset="gaitrec",
                    set_name="smoke-source",
                    output_root=output_root,
                    overwrite=False,
                    dry_run=False,
                )


if __name__ == "__main__":
    unittest.main()

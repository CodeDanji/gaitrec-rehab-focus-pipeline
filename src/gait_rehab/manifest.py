from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
from pathlib import Path
from typing import Any


REQUIRED_MANIFEST_FIELDS = {
    "dataset",
    "version",
    "source_collection_id",
    "source_collection_url",
    "files",
}

REQUIRED_FILE_FIELDS = {
    "role",
    "required_for",
    "article_id",
    "file_id",
    "filename",
    "size_bytes",
    "download_url",
    "target_path",
    "sha256",
}


class ManifestError(ValueError):
    pass


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: dict[str, Any]) -> None:
    missing = sorted(REQUIRED_MANIFEST_FIELDS - set(manifest))
    if missing:
        raise ManifestError(f"Manifest is missing required fields: {missing}")
    if not isinstance(manifest["files"], list) or not manifest["files"]:
        raise ManifestError("Manifest must contain a non-empty files list")

    seen_roles: set[str] = set()
    for index, item in enumerate(manifest["files"]):
        missing_item = sorted(REQUIRED_FILE_FIELDS - set(item))
        if missing_item:
            raise ManifestError(f"Manifest file entry {index} is missing required fields: {missing_item}")
        if not isinstance(item["required_for"], list) or not item["required_for"]:
            raise ManifestError(f"Manifest file entry {index} must declare required_for sets")
        if item["role"] in seen_roles:
            raise ManifestError(f"Duplicate manifest role: {item['role']}")
        seen_roles.add(str(item["role"]))


def select_manifest_files(manifest: dict[str, Any], set_name: str) -> list[dict[str, Any]]:
    return [item for item in manifest["files"] if set_name in item["required_for"]]


def download_manifest_files(
    manifest_path: Path,
    dataset: str,
    set_name: str,
    output_root: Path,
    overwrite: bool = False,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    manifest = load_manifest(manifest_path)
    if manifest["dataset"] != dataset:
        raise ManifestError(f"Manifest dataset {manifest['dataset']!r} does not match requested dataset {dataset!r}")

    selected = select_manifest_files(manifest, set_name)
    if not selected:
        raise ManifestError(f"No files in manifest are required for set {set_name!r}")

    output_root = Path(output_root)
    results: list[dict[str, Any]] = []
    for item in selected:
        target = output_root / item["target_path"]
        expected_size = int(item["size_bytes"])
        planned_status = _planned_status(target, expected_size)
        results.append(
            {
                "role": item["role"],
                "filename": item["filename"],
                "target_path": str(target),
                "size_bytes": expected_size,
                "status": planned_status,
            }
        )
        if dry_run:
            continue
        _download_one(item, target, expected_size, overwrite=overwrite)
        results[-1]["status"] = "skipped" if planned_status == "skip" else "downloaded"
    return results


def _planned_status(target: Path, expected_size: int) -> str:
    if not target.exists():
        return "download"
    return "skip" if target.stat().st_size == expected_size else "size-mismatch"


def _download_one(item: dict[str, Any], target: Path, expected_size: int, overwrite: bool) -> None:
    if target.exists():
        actual_size = target.stat().st_size
        if actual_size == expected_size:
            return
        if not overwrite:
            raise ManifestError(
                f"Existing file size mismatch for {target}: expected {expected_size}, found {actual_size}"
            )

    target.parent.mkdir(parents=True, exist_ok=True)
    url = str(item["download_url"])
    try:
        if url.startswith("file://"):
            shutil.copyfile(Path(urllib.request.url2pathname(url.removeprefix("file://"))), target)
        else:
            urllib.request.urlretrieve(url, target)
    except Exception as exc:  # pragma: no cover - network failures depend on environment
        raise ManifestError(f"Failed to download {item['filename']} from {url}: {exc}") from exc

    actual_size = target.stat().st_size
    if actual_size != expected_size:
        raise ManifestError(f"Downloaded file size mismatch for {target}: expected {expected_size}, found {actual_size}")

    expected_hash = item.get("sha256")
    if expected_hash:
        actual_hash = _sha256(target)
        if actual_hash.lower() != str(expected_hash).lower():
            raise ManifestError(f"Downloaded file sha256 mismatch for {target}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

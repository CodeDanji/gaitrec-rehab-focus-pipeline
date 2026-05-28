from __future__ import annotations

from pathlib import Path

import pandas as pd

from gait_rehab.features import SIGNAL_KEYS, validate_feature_input
from gait_rehab.manifest import load_manifest


SIGNAL_FILE_PATTERNS = {
    "vgrf_left": [["vgrf", "left"], ["vertical", "left"], ["f_v", "pro", "left"], ["grf", "z", "left"]],
    "vgrf_right": [["vgrf", "right"], ["vertical", "right"], ["f_v", "pro", "right"], ["grf", "z", "right"]],
    "ap_grf_left": [["ap", "grf", "left"], ["anterior", "left"], ["grf", "x", "left"]],
    "ap_grf_right": [["ap", "grf", "right"], ["anterior", "right"], ["grf", "x", "right"]],
    "ml_grf_left": [["ml", "grf", "left"], ["medio", "left"], ["grf", "y", "left"]],
    "ml_grf_right": [["ml", "grf", "right"], ["medio", "right"], ["grf", "y", "right"]],
    "cop_ap_left": [["cop", "ap", "left"], ["cop", "x", "left"]],
    "cop_ap_right": [["cop", "ap", "right"], ["cop", "x", "right"]],
    "cop_ml_left": [["cop", "ml", "left"], ["cop", "y", "left"]],
    "cop_ml_right": [["cop", "ml", "right"], ["cop", "y", "right"]],
}


def load_gaitrec_metadata(gaitrec_root: Path) -> pd.DataFrame:
    root = Path(gaitrec_root)
    if not root.exists():
        raise FileNotFoundError(f"GaitRec root does not exist: {root}")
    candidates = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".csv", ".tsv", ".xlsx"} and "metadata" in path.name.lower()
    ]
    if not candidates:
        candidates = [
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in {".csv", ".tsv", ".xlsx"} and "meta" in path.name.lower()
        ]
    if not candidates:
        raise FileNotFoundError("Could not find a metadata CSV/TSV/XLSX file under the GaitRec root")
    return _read_table(candidates[0])


def load_gaitrec_processed_signals(gaitrec_root: Path, manifest_path: Path | None = None) -> dict[str, pd.DataFrame]:
    root = Path(gaitrec_root)
    if manifest_path is not None:
        return _load_gaitrec_processed_signals_from_manifest(root, Path(manifest_path))

    files = [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in {".csv", ".tsv"}]
    signals: dict[str, pd.DataFrame] = {}

    for key in SIGNAL_KEYS:
        path = _find_signal_file(files, SIGNAL_FILE_PATTERNS[key])
        if path is None:
            if key in {"vgrf_left", "vgrf_right"}:
                raise FileNotFoundError(f"Could not locate processed signal file for {key}")
            continue
        signals[key] = _read_table(path)

    return signals


def _load_gaitrec_processed_signals_from_manifest(root: Path, manifest_path: Path) -> dict[str, pd.DataFrame]:
    manifest = load_manifest(manifest_path)
    role_to_path = {
        str(item["role"]): root / str(item["target_path"])
        for item in manifest["files"]
        if str(item["role"]) in SIGNAL_KEYS
    }
    missing_required_roles = [key for key in ["vgrf_left", "vgrf_right"] if key not in role_to_path]
    if missing_required_roles:
        raise FileNotFoundError(f"Manifest is missing required GaitRec signal roles: {missing_required_roles}")

    signals: dict[str, pd.DataFrame] = {}
    for key in SIGNAL_KEYS:
        path = role_to_path.get(key)
        if path is None:
            continue
        if not path.exists():
            if key in {"vgrf_left", "vgrf_right"}:
                raise FileNotFoundError(f"Manifest target for {key} does not exist: {path}")
            continue
        signals[key] = _read_table(path)
    return signals


def validate_gaitrec_inputs(metadata: pd.DataFrame, signals: dict[str, pd.DataFrame]) -> None:
    validate_feature_input(metadata, signals)


def _find_signal_file(files: list[Path], pattern_groups: list[list[str]]) -> Path | None:
    scored: list[tuple[int, Path]] = []
    for path in files:
        name = path.stem.lower().replace("-", "_")
        for pattern in pattern_groups:
            if all(token in name for token in pattern):
                scored.append((len(pattern), path))
    if not scored:
        return None
    scored.sort(key=lambda item: (-item[0], len(str(item[1]))))
    return scored[0][1]


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix == ".xlsx":
        return pd.read_excel(path)
    return pd.read_csv(path)

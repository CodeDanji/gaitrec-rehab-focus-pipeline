from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gait_rehab.features import normalize_label


SMOKE_SOURCE_FILES = {
    "metadata": "GRF_metadata.csv",
    "vgrf_left": "GRF_F_V_PRO_left.csv",
    "vgrf_right": "GRF_F_V_PRO_right.csv",
    "ap_grf_left": "GRF_F_AP_PRO_left.csv",
    "ap_grf_right": "GRF_F_AP_PRO_right.csv",
    "cop_ap_left": "GRF_COP_AP_PRO_left.csv",
    "cop_ap_right": "GRF_COP_AP_PRO_right.csv",
    "cop_ml_left": "GRF_COP_ML_PRO_left.csv",
    "cop_ml_right": "GRF_COP_ML_PRO_right.csv",
}

ML_GRF_FILES = {
    "ml_grf_left": "GRF_F_ML_PRO_left.csv",
    "ml_grf_right": "GRF_F_ML_PRO_right.csv",
}

REQUIRED_LABELS = ["Healthy", "Hip", "Knee", "Ankle", "Calcaneus"]
KEY_COLUMNS = ["SUBJECT_ID", "SESSION_ID", "TRIAL_ID"]


def build_smoke_subset(
    input_root: Path,
    output_root: Path,
    max_bytes: int = 250_000_000,
    seed: int = 42,
    min_subjects_per_label: int = 10,
    min_trials_per_label: int = 30,
    include_ml_grf: bool = False,
) -> dict[str, object]:
    input_root = Path(input_root)
    output_root = Path(output_root)
    files = dict(SMOKE_SOURCE_FILES)
    if include_ml_grf:
        files.update(ML_GRF_FILES)

    _assert_source_files(input_root, files)
    metadata = pd.read_csv(input_root / files["metadata"])
    labeled_metadata = _metadata_with_labels(metadata)

    selected_subjects = _select_subjects_by_label(
        labeled_metadata,
        seed=seed,
        subjects_per_label=min_subjects_per_label,
    )
    selected_metadata = _filter_metadata(metadata, labeled_metadata, selected_subjects)
    signal_frames = {role: pd.read_csv(input_root / filename) for role, filename in files.items() if role != "metadata"}
    selected_keys = _keys_for_selected_metadata(signal_frames["vgrf_left"], selected_metadata)
    selected_signal_frames = _filter_signals(signal_frames, selected_keys)
    _assert_trial_coverage_by_label(selected_metadata, selected_keys, min_trials_per_label)
    _assert_key_consistency(selected_signal_frames)

    output_root.mkdir(parents=True, exist_ok=True)
    written_files = _write_subset_files(output_root, files, selected_metadata, selected_signal_frames)
    output_size = sum(path.stat().st_size for path in written_files)
    if output_size > max_bytes:
        raise ValueError(
            f"Smoke subset is {output_size} bytes, above max_bytes={max_bytes}. "
            "Reduce min_subjects_per_label or include fewer signal files."
        )

    manifest = {
        "seed": seed,
        "max_bytes": max_bytes,
        "include_ml_grf": include_ml_grf,
        "selected_key_count": len(selected_keys),
        "selected_subject_count_by_label": {
            label: len(subjects) for label, subjects in selected_subjects.items()
        },
        "output_size_bytes": output_size,
        "source_files": [
            {"role": role, "filename": filename, "size_bytes": (input_root / filename).stat().st_size}
            for role, filename in files.items()
        ],
    }
    manifest_path = output_root / "smoke_sampling_manifest.json"
    manifest["output_size_bytes"] = _write_manifest_with_total_size(manifest_path, manifest, output_size)
    return manifest


def _assert_source_files(input_root: Path, files: dict[str, str]) -> None:
    missing = [filename for filename in files.values() if not (input_root / filename).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required source files for smoke subset: {missing}")


def _metadata_with_labels(metadata: pd.DataFrame) -> pd.DataFrame:
    df = metadata.copy()
    if "SUBJECT_ID" not in df.columns:
        raise ValueError("Metadata must contain SUBJECT_ID")
    label_col = _first_present(df, ["CLASS_LABEL", "label", "LABEL", "GROUP"])
    if label_col is None:
        raise ValueError("Metadata must contain a label-like column")
    df["_label"] = df[label_col].map(normalize_label)
    df["_subject_id"] = df["SUBJECT_ID"].astype(str)
    if "SESSION_ID" in df.columns:
        df["_session_id"] = df["SESSION_ID"].astype(str)
    else:
        df["_session_id"] = ""
    missing_labels = sorted(set(REQUIRED_LABELS) - set(df["_label"]))
    if missing_labels:
        raise ValueError(f"Metadata is missing required smoke labels: {missing_labels}")
    return df


def _select_subjects_by_label(
    metadata: pd.DataFrame,
    seed: int,
    subjects_per_label: int,
) -> dict[str, list[str]]:
    selected: dict[str, list[str]] = {}
    for label in REQUIRED_LABELS:
        subjects = sorted(metadata.loc[metadata["_label"].eq(label), "_subject_id"].unique())
        if len(subjects) < subjects_per_label:
            raise ValueError(
                f"Label {label} has {len(subjects)} subjects, below min_subjects_per_label={subjects_per_label}"
            )
        sampled = pd.Series(subjects).sample(n=subjects_per_label, random_state=seed).sort_values().tolist()
        selected[label] = [str(subject) for subject in sampled]
    return selected


def _filter_metadata(
    original_metadata: pd.DataFrame,
    labeled_metadata: pd.DataFrame,
    selected_subjects: dict[str, list[str]],
) -> pd.DataFrame:
    selected = {subject for subjects in selected_subjects.values() for subject in subjects}
    return original_metadata.loc[labeled_metadata["_subject_id"].isin(selected)].reset_index(drop=True)


def _keys_for_selected_metadata(signal_df: pd.DataFrame, selected_metadata: pd.DataFrame) -> set[tuple[str, str, str]]:
    selected_subjects = set(selected_metadata["SUBJECT_ID"].astype(str))
    selected_sessions = (
        set(selected_metadata["SESSION_ID"].astype(str)) if "SESSION_ID" in selected_metadata.columns else set()
    )
    signal = _require_key_columns(signal_df)
    mask = signal["SUBJECT_ID"].astype(str).isin(selected_subjects)
    if selected_sessions:
        mask = mask & signal["SESSION_ID"].astype(str).isin(selected_sessions)
    return _key_set(signal.loc[mask])


def _filter_signals(
    signal_frames: dict[str, pd.DataFrame],
    selected_keys: set[tuple[str, str, str]],
) -> dict[str, pd.DataFrame]:
    filtered: dict[str, pd.DataFrame] = {}
    for role, frame in signal_frames.items():
        signal = _require_key_columns(frame)
        mask = signal.apply(
            lambda row: (
                str(row["SUBJECT_ID"]),
                str(row["SESSION_ID"]),
                str(row["TRIAL_ID"]),
            )
            in selected_keys,
            axis=1,
        )
        filtered[role] = frame.loc[mask].reset_index(drop=True)
    return filtered


def _assert_trial_coverage_by_label(
    metadata: pd.DataFrame,
    selected_keys: set[tuple[str, str, str]],
    min_trials_per_label: int,
) -> None:
    labeled = _metadata_with_labels(metadata)
    counts = {label: 0 for label in REQUIRED_LABELS}
    metadata_sessions = {}
    for _, row in labeled.iterrows():
        session_id = str(row["SESSION_ID"]) if "SESSION_ID" in metadata.columns else ""
        metadata_sessions[(str(row["SUBJECT_ID"]), session_id)] = str(row["_label"])
    for subject_id, session_id, _trial_id in selected_keys:
        label = metadata_sessions.get((subject_id, session_id)) or metadata_sessions.get((subject_id, ""))
        if label in counts:
            counts[str(label)] += 1
    low = {label: count for label, count in counts.items() if count < min_trials_per_label}
    if low:
        raise ValueError(f"Selected keys do not satisfy min_trials_per_label={min_trials_per_label}: {low}")


def _assert_key_consistency(signal_frames: dict[str, pd.DataFrame]) -> None:
    expected: set[tuple[str, str, str]] | None = None
    for role, frame in signal_frames.items():
        current = _key_set(_require_key_columns(frame))
        if expected is None:
            expected = current
        elif current != expected:
            raise ValueError(f"Signal key mismatch for {role}")


def _write_subset_files(
    output_root: Path,
    files: dict[str, str],
    metadata: pd.DataFrame,
    signal_frames: dict[str, pd.DataFrame],
) -> list[Path]:
    written: list[Path] = []
    metadata_path = output_root / files["metadata"]
    metadata.to_csv(metadata_path, index=False)
    written.append(metadata_path)
    for role, frame in signal_frames.items():
        target = output_root / files[role]
        frame.to_csv(target, index=False)
        written.append(target)
    return written


def _write_manifest_with_total_size(path: Path, manifest: dict[str, object], data_size: int) -> int:
    total = data_size
    while True:
        manifest["output_size_bytes"] = total
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        next_total = data_size + path.stat().st_size
        if next_total == total:
            return total
        total = next_total


def _key_set(df: pd.DataFrame) -> set[tuple[str, str, str]]:
    return set(zip(df["SUBJECT_ID"].astype(str), df["SESSION_ID"].astype(str), df["TRIAL_ID"].astype(str)))


def _require_key_columns(df: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in KEY_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Signal file is missing key columns: {missing}")
    return df


def _first_present(df: pd.DataFrame, names: list[str]) -> str | None:
    lower = {str(column).lower(): column for column in df.columns}
    for name in names:
        if name.lower() in lower:
            return str(lower[name.lower()])
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a key-consistent GaitRec smoke subset.")
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--max-bytes", type=int, default=250_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-subjects-per-label", type=int, default=10)
    parser.add_argument("--min-trials-per-label", type=int, default=30)
    parser.add_argument("--include-ml-grf", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_smoke_subset(
        input_root=args.input_root,
        output_root=args.output_root,
        max_bytes=args.max_bytes,
        seed=args.seed,
        min_subjects_per_label=args.min_subjects_per_label,
        min_trials_per_label=args.min_trials_per_label,
        include_ml_grf=args.include_ml_grf,
    )
    print(
        f"Wrote {manifest['selected_key_count']} selected keys to {args.output_root} "
        f"({manifest['output_size_bytes']} bytes)"
    )


if __name__ == "__main__":
    main()

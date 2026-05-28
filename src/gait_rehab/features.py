from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


METADATA_ALIASES = {
    "subject_id": ["subject_id", "subject", "participant_id", "patient_id", "id"],
    "session_id": ["session_id", "session", "recording_id", "measurement_id"],
    "trial_id": ["trial_id", "trial", "trial_number", "recording_id", "measurement_id"],
    "label": ["label", "class_label", "impairment", "impairment_label", "group", "diagnosis_group"],
    "affected_side": ["affected_side", "affected", "side", "affected_limb"],
    "walking_speed": ["walking_speed", "speed", "velocity"],
    "age": ["age"],
    "sex": ["sex", "gender"],
    "height": ["height", "height_cm", "body_height"],
    "weight": ["weight", "weight_kg", "body_weight", "body_mass"],
    "shoe_condition": ["shoe_condition", "shod_condition", "shoe", "shoes", "footwear"],
}

SIGNAL_KEYS = [
    "vgrf_left",
    "vgrf_right",
    "ap_grf_left",
    "ap_grf_right",
    "ml_grf_left",
    "ml_grf_right",
    "cop_ap_left",
    "cop_ap_right",
    "cop_ml_left",
    "cop_ml_right",
]

FEATURE_COLUMNS = [
    "vgrf_peak_aff",
    "vgrf_peak_unaff",
    "vgrf_peak_asym",
    "loading_rate_asym",
    "ap_braking_impulse_asym",
    "ap_propulsion_impulse_asym",
    "push_off_index",
    "cop_ap_range_aff",
    "cop_ml_range_aff",
    "cop_path_length_aff",
    "cop_ap_range_asym",
    "cop_ml_range_asym",
]


@dataclass(frozen=True)
class TrialSignals:
    subject_id: str
    trial_id: str
    label: str
    affected_side: str
    walking_speed: float | None
    age: float | None
    sex: str | None
    height: float | None
    weight: float | None
    shoe_condition: str | None
    vgrf: dict[str, np.ndarray]
    ap_grf: dict[str, np.ndarray]
    ml_grf: dict[str, np.ndarray]
    cop_ap: dict[str, np.ndarray]
    cop_ml: dict[str, np.ndarray]
    session_id: str | None = None


def clean_signal(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float).reshape(-1)
    return arr[~np.isnan(arr)]


def peak(values: Iterable[float]) -> float:
    arr = clean_signal(values)
    if arr.size == 0:
        return float("nan")
    return float(np.nanmax(arr))


def signal_range(values: Iterable[float]) -> float:
    arr = clean_signal(values)
    if arr.size == 0:
        return float("nan")
    return float(np.nanmax(arr) - np.nanmin(arr))


def impulse(values: Iterable[float], dt: float = 1.0) -> float:
    arr = clean_signal(values)
    if arr.size == 0:
        return float("nan")
    return float(np.nansum(arr) * dt)


def braking_impulse(values: Iterable[float], dt: float = 1.0) -> float:
    arr = clean_signal(values)
    if arr.size == 0:
        return float("nan")
    return float(np.nansum(np.abs(arr[arr < 0])) * dt)


def propulsion_impulse(values: Iterable[float], dt: float = 1.0) -> float:
    arr = clean_signal(values)
    if arr.size == 0:
        return float("nan")
    return float(np.nansum(arr[arr > 0]) * dt)


def loading_rate(values: Iterable[float]) -> float:
    arr = clean_signal(values)
    if arr.size < 2:
        return float("nan")
    return float(np.nanmax(arr) - arr[0])


def asymmetry(affected_value: float, unaffected_value: float) -> float:
    if pd.isna(affected_value) or pd.isna(unaffected_value):
        return float("nan")
    denom = (abs(affected_value) + abs(unaffected_value)) / 2.0
    if denom == 0:
        return 0.0
    return float((affected_value - unaffected_value) / denom)


def cop_path_length(cop_ap: Iterable[float], cop_ml: Iterable[float]) -> float:
    ap = clean_signal(cop_ap)
    ml = clean_signal(cop_ml)
    n = min(ap.size, ml.size)
    if n < 2:
        return 0.0
    ap = ap[:n]
    ml = ml[:n]
    return float(np.nansum(np.sqrt(np.diff(ap) ** 2 + np.diff(ml) ** 2)))


def normalize_metadata_columns(metadata: pd.DataFrame) -> pd.DataFrame:
    df = metadata.copy()
    lower_to_original = {str(col).strip().lower(): col for col in df.columns}
    rename: dict[str, str] = {}

    for canonical, aliases in METADATA_ALIASES.items():
        for alias in aliases:
            if alias.lower() in lower_to_original:
                rename[lower_to_original[alias.lower()]] = canonical
                break

    df = df.rename(columns=rename)
    if "trial_id" not in df.columns and "session_id" not in df.columns:
        df["trial_id"] = df.groupby("subject_id").cumcount().astype(str)

    if "affected_side" in df.columns:
        df["affected_side"] = df["affected_side"].map(_normalize_side).fillna("left")
    else:
        df["affected_side"] = "left"

    if "label" in df.columns:
        df["label"] = df["label"].map(normalize_label)

    return df


def normalize_label(value: object) -> str:
    text = str(value).strip()
    lowered = text.lower()
    if lowered in {"healthy", "control", "healthy control", "hc"}:
        return "Healthy"
    if lowered in {"h", "hip"}:
        return "Hip"
    if lowered in {"k", "knee"}:
        return "Knee"
    if lowered in {"a", "ankle"}:
        return "Ankle"
    if lowered in {"c", "calcaneus"}:
        return "Calcaneus"
    for label in ["Hip", "Knee", "Ankle", "Calcaneus"]:
        if label.lower() in lowered:
            return label
    return text or "Unknown"


def _normalize_side(value: object) -> str:
    text = str(value).strip().lower()
    if text in {"l", "left", "left side", "affected left", "1"}:
        return "left"
    if text in {"r", "right", "right side", "affected right", "2"}:
        return "right"
    return text if text in {"left", "right"} else "left"


def validate_feature_input(metadata: pd.DataFrame, signals: dict[str, pd.DataFrame]) -> None:
    normalized = normalize_metadata_columns(metadata)
    missing_columns = [
        col for col in ["subject_id", "label", "affected_side"] if col not in normalized.columns
    ]
    if missing_columns:
        raise ValueError(f"Metadata is missing required columns: {missing_columns}")

    missing_signals = [key for key in ["vgrf_left", "vgrf_right"] if key not in signals]
    if missing_signals:
        raise ValueError(f"Signal dictionary is missing required keys: {missing_signals}")


def extract_gait_features(metadata: pd.DataFrame, signals: dict[str, pd.DataFrame]) -> pd.DataFrame:
    validate_feature_input(metadata, signals)
    metadata = normalize_metadata_columns(metadata)
    rows: list[dict[str, object]] = []

    for trial in iterate_trials(metadata, signals):
        aff = trial.affected_side
        unaff = "right" if aff == "left" else "left"

        vgrf_peak_aff = peak(trial.vgrf[aff])
        vgrf_peak_unaff = peak(trial.vgrf[unaff])
        loading_aff = loading_rate(trial.vgrf[aff])
        loading_unaff = loading_rate(trial.vgrf[unaff])
        brake_aff = braking_impulse(trial.ap_grf[aff])
        brake_unaff = braking_impulse(trial.ap_grf[unaff])
        prop_aff = propulsion_impulse(trial.ap_grf[aff])
        prop_unaff = propulsion_impulse(trial.ap_grf[unaff])
        cop_ap_range_aff = signal_range(trial.cop_ap[aff])
        cop_ml_range_aff = signal_range(trial.cop_ml[aff])
        cop_ap_range_unaff = signal_range(trial.cop_ap[unaff])
        cop_ml_range_unaff = signal_range(trial.cop_ml[unaff])
        cop_path_aff = (
            cop_path_length(trial.cop_ap[aff], trial.cop_ml[aff])
            if trial.cop_ap[aff].size and trial.cop_ml[aff].size
            else float("nan")
        )

        rows.append(
            {
                "subject_id": trial.subject_id,
                "session_id": trial.session_id,
                "trial_id": trial.trial_id,
                "label": trial.label,
                "affected_side": aff,
                "walking_speed": trial.walking_speed,
                "age": trial.age,
                "sex": trial.sex,
                "height": trial.height,
                "weight": trial.weight,
                "shoe_condition": trial.shoe_condition,
                "vgrf_peak_aff": vgrf_peak_aff,
                "vgrf_peak_unaff": vgrf_peak_unaff,
                "vgrf_peak_asym": asymmetry(vgrf_peak_aff, vgrf_peak_unaff),
                "loading_rate_asym": asymmetry(loading_aff, loading_unaff),
                "ap_braking_impulse_asym": asymmetry(brake_aff, brake_unaff),
                "ap_propulsion_impulse_asym": asymmetry(prop_aff, prop_unaff),
                "push_off_index": prop_aff,
                "cop_ap_range_aff": cop_ap_range_aff,
                "cop_ml_range_aff": cop_ml_range_aff,
                "cop_path_length_aff": cop_path_aff,
                "cop_ap_range_asym": asymmetry(cop_ap_range_aff, cop_ap_range_unaff),
                "cop_ml_range_asym": asymmetry(cop_ml_range_aff, cop_ml_range_unaff),
            }
        )

    return pd.DataFrame(rows)


def iterate_trials(metadata: pd.DataFrame, signals: dict[str, pd.DataFrame]) -> Iterable[TrialSignals]:
    normalized_metadata = normalize_metadata_columns(metadata)
    normalized_signals = {key: _normalize_key_columns(value) for key, value in signals.items()}
    left_vgrf = normalized_signals["vgrf_left"]
    right_vgrf = normalized_signals["vgrf_right"]
    right_lookup = _build_signal_lookup(right_vgrf)
    optional_lookups = {
        key: _build_signal_lookup(value)
        for key, value in normalized_signals.items()
        if key not in {"vgrf_left", "vgrf_right"}
    }
    metadata_lookup = _build_metadata_lookup(normalized_metadata)

    for _, left_row in left_vgrf.iterrows():
        subject_id = _key_str(left_row["subject_id"])
        session_id = _key_str(left_row.get("session_id")) if "session_id" in left_row.index else None
        raw_trial_id = _key_str(left_row.get("trial_id")) if "trial_id" in left_row.index else "0"
        trial_id = _compose_trial_id(session_id, raw_trial_id)
        metadata_row = _lookup_metadata_row(metadata_lookup, subject_id, session_id, raw_trial_id)
        right_row = _lookup_signal_row(right_lookup, subject_id, session_id, raw_trial_id)

        yield TrialSignals(
            subject_id=subject_id,
            trial_id=trial_id,
            session_id=session_id,
            label=str(metadata_row["label"]),
            affected_side=_normalize_side(metadata_row["affected_side"]),
            walking_speed=_optional_float(metadata_row.get("walking_speed")),
            age=_optional_float(metadata_row.get("age")),
            sex=_optional_str(metadata_row.get("sex")),
            height=_optional_float(metadata_row.get("height")),
            weight=_optional_float(metadata_row.get("weight")),
            shoe_condition=_optional_str(metadata_row.get("shoe_condition")),
            vgrf={
                "left": row_signal(left_row),
                "right": row_signal(right_row),
            },
            ap_grf={
                "left": get_optional_trial_signal(optional_lookups, "ap_grf_left", subject_id, session_id, raw_trial_id),
                "right": get_optional_trial_signal(optional_lookups, "ap_grf_right", subject_id, session_id, raw_trial_id),
            },
            ml_grf={
                "left": get_optional_trial_signal(optional_lookups, "ml_grf_left", subject_id, session_id, raw_trial_id),
                "right": get_optional_trial_signal(optional_lookups, "ml_grf_right", subject_id, session_id, raw_trial_id),
            },
            cop_ap={
                "left": get_optional_trial_signal(optional_lookups, "cop_ap_left", subject_id, session_id, raw_trial_id),
                "right": get_optional_trial_signal(optional_lookups, "cop_ap_right", subject_id, session_id, raw_trial_id),
            },
            cop_ml={
                "left": get_optional_trial_signal(optional_lookups, "cop_ml_left", subject_id, session_id, raw_trial_id),
                "right": get_optional_trial_signal(optional_lookups, "cop_ml_right", subject_id, session_id, raw_trial_id),
            },
        )


def get_trial_signal(signal_df: pd.DataFrame, subject_id: str, trial_id: str, session_id: str | None = None) -> np.ndarray:
    df = _normalize_key_columns(signal_df)
    mask = df["subject_id"].map(_key_str).eq(subject_id)
    if "trial_id" in df.columns:
        mask = mask & df["trial_id"].map(_key_str).eq(trial_id)
    if session_id is not None and "session_id" in df.columns:
        mask = mask & df["session_id"].map(_key_str).eq(session_id)
    matched = df.loc[mask]
    if matched.empty:
        raise KeyError(f"Missing signal for subject={subject_id}, trial={trial_id}")

    value_cols = [col for col in ["value", "signal", "force", "cop"] if col in matched.columns]
    if value_cols:
        sort_cols = [col for col in ["time", "sample", "percent_gait_cycle"] if col in matched.columns]
        if sort_cols:
            matched = matched.sort_values(sort_cols[0])
        return clean_signal(matched[value_cols[0]].to_numpy())

    return row_signal(matched.iloc[0])


def get_optional_trial_signal(
    signal_lookups: dict[str, dict[tuple[str, str | None, str], pd.Series]],
    signal_key: str,
    subject_id: str,
    session_id: str | None,
    raw_trial_id: str,
) -> np.ndarray:
    if signal_key not in signal_lookups:
        return np.asarray([], dtype=float)
    try:
        return row_signal(_lookup_signal_row(signal_lookups[signal_key], subject_id, session_id, raw_trial_id))
    except KeyError:
        return np.asarray([], dtype=float)


def row_signal(row: pd.Series) -> np.ndarray:
    ignored = {"subject_id", "session_id", "trial_id", "side", "label", "affected_side"}
    numeric_cols = [col for col in row.index if col not in ignored and pd.api.types.is_number(row[col])]
    if not numeric_cols:
        return np.asarray([], dtype=float)
    return clean_signal(row[numeric_cols].to_numpy())


def _normalize_key_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    lower_to_original = {str(col).strip().lower(): col for col in normalized.columns}
    rename: dict[str, str] = {}
    for canonical, aliases in {
        "subject_id": METADATA_ALIASES["subject_id"],
        "session_id": METADATA_ALIASES["session_id"],
        "trial_id": METADATA_ALIASES["trial_id"],
    }.items():
        for alias in aliases:
            if alias.lower() in lower_to_original:
                rename[lower_to_original[alias.lower()]] = canonical
                break
    normalized = normalized.rename(columns=rename)
    if "subject_id" not in normalized.columns:
        raise ValueError("Signal frame is missing a subject_id-like column")
    return normalized


def _find_metadata_row(
    metadata: pd.DataFrame,
    subject_id: str,
    session_id: str | None,
    raw_trial_id: str,
) -> pd.Series:
    mask = metadata["subject_id"].map(_key_str).eq(subject_id)
    if session_id is not None and "session_id" in metadata.columns:
        mask = mask & metadata["session_id"].map(_key_str).eq(session_id)
    elif "trial_id" in metadata.columns:
        mask = mask & metadata["trial_id"].map(_key_str).eq(raw_trial_id)
    matched = metadata.loc[mask]
    if matched.empty:
        raise KeyError(f"Missing metadata for subject={subject_id}, session={session_id}, trial={raw_trial_id}")
    return matched.iloc[0]


def _build_metadata_lookup(metadata: pd.DataFrame) -> dict[tuple[str, str | None, str], pd.Series]:
    lookup: dict[tuple[str, str | None, str], pd.Series] = {}
    for _, row in metadata.iterrows():
        subject_id = _key_str(row["subject_id"])
        session_id = _key_str(row.get("session_id")) if "session_id" in row.index else None
        trial_id = _key_str(row.get("trial_id")) if "trial_id" in row.index else ""
        lookup[(subject_id, session_id, trial_id)] = row
        if session_id:
            lookup[(subject_id, session_id, "")] = row
    return lookup


def _lookup_metadata_row(
    lookup: dict[tuple[str, str | None, str], pd.Series],
    subject_id: str,
    session_id: str | None,
    raw_trial_id: str,
) -> pd.Series:
    for key in [
        (subject_id, session_id, raw_trial_id),
        (subject_id, session_id, ""),
        (subject_id, None, raw_trial_id),
    ]:
        if key in lookup:
            return lookup[key]
    raise KeyError(f"Missing metadata for subject={subject_id}, session={session_id}, trial={raw_trial_id}")


def _find_signal_row(
    signal_df: pd.DataFrame,
    subject_id: str,
    session_id: str | None,
    raw_trial_id: str,
) -> pd.Series:
    mask = signal_df["subject_id"].map(_key_str).eq(subject_id)
    if session_id is not None and "session_id" in signal_df.columns:
        mask = mask & signal_df["session_id"].map(_key_str).eq(session_id)
    if "trial_id" in signal_df.columns:
        mask = mask & signal_df["trial_id"].map(_key_str).eq(raw_trial_id)
    matched = signal_df.loc[mask]
    if matched.empty:
        raise KeyError(f"Missing signal for subject={subject_id}, session={session_id}, trial={raw_trial_id}")
    return matched.iloc[0]


def _build_signal_lookup(signal_df: pd.DataFrame) -> dict[tuple[str, str | None, str], pd.Series]:
    lookup: dict[tuple[str, str | None, str], pd.Series] = {}
    for _, row in signal_df.iterrows():
        subject_id = _key_str(row["subject_id"])
        session_id = _key_str(row.get("session_id")) if "session_id" in row.index else None
        trial_id = _key_str(row.get("trial_id")) if "trial_id" in row.index else "0"
        lookup[(subject_id, session_id, trial_id)] = row
    return lookup


def _lookup_signal_row(
    lookup: dict[tuple[str, str | None, str], pd.Series],
    subject_id: str,
    session_id: str | None,
    raw_trial_id: str,
) -> pd.Series:
    key = (subject_id, session_id, raw_trial_id)
    if key in lookup:
        return lookup[key]
    fallback = (subject_id, None, raw_trial_id)
    if fallback in lookup:
        return lookup[fallback]
    raise KeyError(f"Missing signal for subject={subject_id}, session={session_id}, trial={raw_trial_id}")


def _compose_trial_id(session_id: str | None, raw_trial_id: str) -> str:
    return f"{session_id}_{raw_trial_id}" if session_id else raw_trial_id


def _key_str(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _optional_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_str(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    return str(value)

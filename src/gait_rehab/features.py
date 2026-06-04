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
    "body_weight": ["weight", "weight_kg", "body_weight", "body_weight_n"],
    "body_mass": ["body_mass", "mass", "mass_kg"],
    "shoe_condition": ["shoe_condition", "shod_condition", "shoe", "shoes", "footwear"],
    "shoe_size": ["shoe_size", "size"],
    "orthopedic_insole": ["orthopedic_insole", "insole", "orthopedic_shoe"],
    "session_type": ["session_type", "type"],
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

FEATURE_COLUMNS = []


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
    body_weight: float | None
    body_mass: float | None
    shoe_condition: str | None
    shoe_size: float | None
    orthopedic_insole: str | None
    session_type: str | None
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
    """Computes the integral (area under curve) over the normalized frames.
    Note: dt is 1.0 frame by default, so this is a normalized area, not true physical impulse (N*s).
    """
    arr = clean_signal(values)
    if arr.size == 0:
        return float("nan")
    return float(np.nansum(arr) * dt)


def braking_impulse(values: Iterable[float], dt: float = 1.0) -> float:
    """Computes the braking integral over the normalized frames."""
    arr = clean_signal(values)
    if arr.size == 0:
        return float("nan")
    return float(np.nansum(np.abs(arr[arr < 0])) * dt)


def propulsion_impulse(values: Iterable[float], dt: float = 1.0) -> float:
    """Computes the propulsion integral over the normalized frames."""
    arr = clean_signal(values)
    if arr.size == 0:
        return float("nan")
    return float(np.nansum(arr[arr > 0]) * dt)


def loading_rate(values: Iterable[float]) -> float:
    """Computes the maximum derivative in the early stance phase (first 20%)."""
    arr = clean_signal(values)
    if arr.size < 2:
        return float("nan")
    n_early = max(2, int(arr.size * 0.2))
    early_arr = arr[:n_early]
    # Max difference between consecutive frames in early stance
    return float(np.nanmax(np.diff(early_arr)))


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
        df["affected_side"] = df["affected_side"].map(_normalize_side)
    else:
        df["affected_side"] = "unknown"

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
    if text in {"0.0", "0", "l", "left", "left side", "affected left"}:
        return "left"
    if text in {"1.0", "1", "r", "right", "right side", "affected right"}:
        return "right"
    if text in {"2.0", "2", "b", "both", "both sides"}:
        return "both"
    if text in {"left", "right", "both"}:
        return text
    return "unknown"


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
    import warnings
    validate_feature_input(metadata, signals)
    metadata = normalize_metadata_columns(metadata)
    rows: list[dict[str, object]] = []

    def sym_mag(l: float, r: float) -> float:
        if pd.isna(l) or pd.isna(r):
            return float("nan")
        denom = (abs(l) + abs(r)) / 2.0
        return float(abs(l - r) / denom) if denom > 0 else 0.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for trial in iterate_trials(metadata, signals):
            aff = trial.affected_side
        
            # Base features per side
            vgrf_peak_l = peak(trial.vgrf.get("left", []))
            vgrf_peak_r = peak(trial.vgrf.get("right", []))
            loading_l = loading_rate(trial.vgrf.get("left", []))
            loading_r = loading_rate(trial.vgrf.get("right", []))
            brake_l = braking_impulse(trial.ap_grf.get("left", []))
            brake_r = braking_impulse(trial.ap_grf.get("right", []))
            prop_l = propulsion_impulse(trial.ap_grf.get("left", []))
            prop_r = propulsion_impulse(trial.ap_grf.get("right", []))
            cop_ap_l = signal_range(trial.cop_ap.get("left", []))
            cop_ap_r = signal_range(trial.cop_ap.get("right", []))
            cop_ml_l = signal_range(trial.cop_ml.get("left", []))
            cop_ml_r = signal_range(trial.cop_ml.get("right", []))
        
            cop_path_l = cop_path_length(trial.cop_ap.get("left", []), trial.cop_ml.get("left", [])) if trial.cop_ap.get("left", np.array([])).size else float("nan")
            cop_path_r = cop_path_length(trial.cop_ap.get("right", []), trial.cop_ml.get("right", [])) if trial.cop_ap.get("right", np.array([])).size else float("nan")

            # Side-neutral features
            vgrf_peak_mean = np.nanmean([vgrf_peak_l, vgrf_peak_r])
            vgrf_peak_max = np.nanmax([vgrf_peak_l, vgrf_peak_r])
            vgrf_peak_diff = abs(vgrf_peak_l - vgrf_peak_r)
        
            loading_mean = np.nanmean([loading_l, loading_r])
            loading_max = np.nanmax([loading_l, loading_r])
            loading_sym = sym_mag(loading_l, loading_r)
        
            brake_mean = np.nanmean([brake_l, brake_r])
            brake_sym = sym_mag(brake_l, brake_r)
        
            prop_mean = np.nanmean([prop_l, prop_r])
            prop_sym = sym_mag(prop_l, prop_r)
        
            cop_ap_mean = np.nanmean([cop_ap_l, cop_ap_r])
            cop_ap_sym = sym_mag(cop_ap_l, cop_ap_r)
        
            cop_ml_mean = np.nanmean([cop_ml_l, cop_ml_r])
            cop_ml_sym = sym_mag(cop_ml_l, cop_ml_r)
        
            cop_path_mean = np.nanmean([cop_path_l, cop_path_r])

            # Affected-side features (only if left or right is clearly defined)
            if aff in ["left", "right"]:
                unaff = "right" if aff == "left" else "left"
                vgrf_peak_aff = vgrf_peak_l if aff == "left" else vgrf_peak_r
                vgrf_peak_unaff = vgrf_peak_r if aff == "left" else vgrf_peak_l
                loading_aff = loading_l if aff == "left" else loading_r
                loading_unaff = loading_r if aff == "left" else loading_l
                brake_aff = brake_l if aff == "left" else brake_r
                brake_unaff = brake_r if aff == "left" else brake_l
                prop_aff = prop_l if aff == "left" else prop_r
                prop_unaff = prop_r if aff == "left" else prop_l
                cop_ap_aff = cop_ap_l if aff == "left" else cop_ap_r
                cop_ap_unaff = cop_ap_r if aff == "left" else cop_ap_l
                cop_ml_aff = cop_ml_l if aff == "left" else cop_ml_r
                cop_ml_unaff = cop_ml_r if aff == "left" else cop_ml_l
                cop_path_aff = cop_path_l if aff == "left" else cop_path_r
            else:
                vgrf_peak_aff = vgrf_peak_unaff = float("nan")
                loading_aff = loading_unaff = float("nan")
                brake_aff = brake_unaff = float("nan")
                prop_aff = prop_unaff = float("nan")
                cop_ap_aff = cop_ap_unaff = float("nan")
                cop_ml_aff = cop_ml_unaff = float("nan")
                cop_path_aff = float("nan")

            # BMI computation (use mass if available, else convert weight from Newtons to kg)
            bmi = float("nan")
            body_mass_kg = trial.body_mass
            if body_mass_kg is None and trial.body_weight is not None:
                body_mass_kg = trial.body_weight / 9.81
            if body_mass_kg and trial.height and trial.height > 0:
                bmi = float(body_mass_kg / ((trial.height / 100.0) ** 2))

            # The GaitRec PRO data is already amplitude-normalized to multiples of body weight.
            # Do NOT divide by bw_norm again, which would cause double normalization.

            def pad_or_truncate(arr, length=101):
                res = np.full(length, np.nan)
                n = min(len(arr), length)
                if n > 0:
                    res[:n] = arr[:n]
                return res

            vgrf_l_norm = pad_or_truncate(trial.vgrf.get("left", []))
            vgrf_r_norm = pad_or_truncate(trial.vgrf.get("right", []))

            row_dict = {
                    "subject_id": trial.subject_id,
                    "session_id": trial.session_id,
                    "trial_id": trial.trial_id,
                    "label": trial.label,
                    "affected_side": aff,
                    "walking_speed": trial.walking_speed,
                    "age": trial.age,
                    "sex": trial.sex,
                    "height": trial.height,
                    "body_weight": trial.body_weight,
                    "body_mass": trial.body_mass,
                    "bmi": bmi,
                    "shoe_condition": trial.shoe_condition,
                    "shoe_size": trial.shoe_size,
                    "orthopedic_insole": trial.orthopedic_insole,
                    "session_type": trial.session_type,
                
                    # Side-neutral
                    "vgrf_peak_mean": float(vgrf_peak_mean),
                    "vgrf_peak_max": float(vgrf_peak_max),
                    "vgrf_peak_diff": float(vgrf_peak_diff),
                    "loading_rate_mean": float(loading_mean),
                    "loading_rate_max": float(loading_max),
                    "loading_rate_sym": float(loading_sym),
                    "ap_braking_impulse_mean": float(brake_mean),
                    "ap_braking_impulse_sym": float(brake_sym),
                    "ap_propulsion_impulse_mean": float(prop_mean),
                    "ap_propulsion_impulse_sym": float(prop_sym),
                    "push_off_index": float(prop_mean),
                    "cop_ap_range_mean": float(cop_ap_mean),
                    "cop_ap_range_sym": float(cop_ap_sym),
                    "cop_ml_range_mean": float(cop_ml_mean),
                    "cop_ml_range_sym": float(cop_ml_sym),
                    "cop_path_length_mean": float(cop_path_mean),

                    # Affected-side
                    "vgrf_peak_aff": vgrf_peak_aff,
                    "vgrf_peak_unaff": vgrf_peak_unaff,
                    "vgrf_peak_asym": asymmetry(vgrf_peak_aff, vgrf_peak_unaff),
                    "loading_rate_asym": asymmetry(loading_aff, loading_unaff),
                    "ap_braking_impulse_asym": asymmetry(brake_aff, brake_unaff),
                    "ap_propulsion_impulse_asym": asymmetry(prop_aff, prop_unaff),
                    "cop_ap_range_aff": cop_ap_aff,
                    "cop_ml_range_aff": cop_ml_aff,
                    "cop_path_length_aff": cop_path_aff,
                    "cop_ap_range_asym": asymmetry(cop_ap_aff, cop_ap_unaff),
                    "cop_ml_range_asym": asymmetry(cop_ml_aff, cop_ml_unaff),
                }

            for i in range(101):
                row_dict[f"vgrf_left_{i}"] = float(vgrf_l_norm[i])
                row_dict[f"vgrf_right_{i}"] = float(vgrf_r_norm[i])

            rows.append(row_dict)

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
        try:
            subject_id = _key_str(left_row["subject_id"])
            session_id = _key_str(left_row.get("session_id")) if "session_id" in left_row.index else None
            raw_trial_id = _key_str(left_row.get("trial_id")) if "trial_id" in left_row.index else "0"
            trial_id = _compose_trial_id(session_id, raw_trial_id)
            metadata_row = _lookup_metadata_row(metadata_lookup, subject_id, session_id, raw_trial_id)
            right_row = _lookup_signal_row(right_lookup, subject_id, session_id, raw_trial_id)
        except KeyError:
            continue

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
            body_weight=_optional_float(metadata_row.get("body_weight")),
            body_mass=_optional_float(metadata_row.get("body_mass")),
            shoe_condition=_optional_str(metadata_row.get("shoe_condition")),
            shoe_size=_optional_float(metadata_row.get("shoe_size")),
            orthopedic_insole=_optional_str(metadata_row.get("orthopedic_insole")),
            session_type=_optional_str(metadata_row.get("session_type")),
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

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from gait_rehab.features import FEATURE_COLUMNS


FORBIDDEN_REPORT_TERMS = [
    "diagnosis",
    "prescription",
    "cause confirmed",
    "specific muscle weakness confirmed",
    "진단",
    "처방",
    "원인 확정",
    "특정 근육 약화 확정",
]

FEATURE_INTERPRETATIONS = {
    "push_off_index": "affected-side push-off capacity",
    "ap_propulsion_impulse_asym": "left-right propulsion balance",
    "ap_braking_impulse_asym": "braking/loading response balance",
    "vgrf_peak_asym": "vertical weight-bearing symmetry",
    "loading_rate_asym": "early stance loading strategy",
    "cop_ml_range_aff": "medio-lateral pressure control",
    "cop_ap_range_aff": "anterior-posterior rollover pattern",
    "cop_path_length_aff": "pressure-path stability during stance",
    "walking_speed": "overall gait speed context",
}

FEATURE_CANDIDATES = {
    "push_off_index": "ankle push-off function",
    "ap_propulsion_impulse_asym": "propulsive force symmetry",
    "ap_braking_impulse_asym": "braking and loading response control",
    "vgrf_peak_asym": "weight-bearing avoidance strategy",
    "loading_rate_asym": "early stance load acceptance",
    "cop_ml_range_aff": "balance and frontal-plane foot control",
    "cop_ap_range_aff": "foot rollover and progression",
    "cop_path_length_aff": "stance pressure-path stability",
    "walking_speed": "speed-normalized functional assessment",
}


def summarize_groups(features: pd.DataFrame) -> pd.DataFrame:
    numeric_features = [col for col in FEATURE_COLUMNS + ["walking_speed", "age", "height", "weight"] if col in features.columns]
    rows = []
    for label, group in features.groupby("label", dropna=False):
        for feature in numeric_features:
            values = pd.to_numeric(group[feature], errors="coerce").dropna()
            if values.empty:
                continue
            std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            rows.append(
                {
                    "label": label,
                    "feature": feature,
                    "n": int(values.size),
                    "mean": float(values.mean()),
                    "std": std,
                    "ci95_low": float(values.mean() - 1.96 * std / np.sqrt(values.size)) if values.size else np.nan,
                    "ci95_high": float(values.mean() + 1.96 * std / np.sqrt(values.size)) if values.size else np.nan,
                }
            )
    return pd.DataFrame(rows)


def subject_trial_counts(features: pd.DataFrame) -> pd.DataFrame:
    return (
        features.groupby("label")
        .agg(subject_count=("subject_id", "nunique"), trial_count=("trial_id", "count"))
        .reset_index()
        .sort_values("label")
    )


def build_rehab_focus_report(
    subject_id: str,
    trial_id: str,
    predicted_label: str,
    evidence: list[tuple[str, float, str]],
    candidates: list[str],
) -> str:
    lines = [
        f"# Rehab Focus Candidate Report: {subject_id} / {trial_id}",
        "",
        f"Model pattern group: {predicted_label}",
        "",
        "This report highlights gait-function patterns that can guide a follow-up rehabilitation assessment.",
        "It does not determine a clinical condition or assign a treatment.",
        "",
        "## Evidence Features",
    ]
    for feature, score, explanation in evidence:
        lines.append(f"- `{feature}`: z={score:.2f}; {explanation}.")

    lines.extend(["", "## Priority Check Candidates"])
    for candidate in candidates:
        lines.append(f"- {candidate}")

    lines.extend(
        [
            "",
            "## Interpretation Guardrail",
            "- Use these outputs as screening cues for human assessment.",
            "- Confirm with clinical examination, pain report, range of motion, and strength testing.",
        ]
    )
    report = "\n".join(lines)
    _assert_no_forbidden_terms(report)
    return report


def generate_example_rehab_report(
    features: pd.DataFrame,
    output_path: Path,
    predicted_labels: pd.Series | None = None,
) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row = _select_example_row(features)
    predicted_label = str(predicted_labels.loc[row.name]) if predicted_labels is not None and row.name in predicted_labels.index else str(row["label"])
    evidence = rank_evidence_features(row, features)
    candidates = candidates_from_evidence(evidence)
    report = build_rehab_focus_report(
        subject_id=str(row["subject_id"]),
        trial_id=str(row["trial_id"]),
        predicted_label=predicted_label,
        evidence=evidence,
        candidates=candidates,
    )
    output_path.write_text(report, encoding="utf-8")
    return report


def rank_evidence_features(row: pd.Series, features: pd.DataFrame, top_n: int = 3) -> list[tuple[str, float, str]]:
    candidates = [col for col in FEATURE_COLUMNS + ["walking_speed"] if col in features.columns]
    zscores: list[tuple[str, float, str]] = []
    for feature in candidates:
        values = pd.to_numeric(features[feature], errors="coerce")
        mean = values.mean()
        std = values.std()
        if pd.isna(row.get(feature)) or pd.isna(mean) or pd.isna(std) or std == 0:
            continue
        z = float((row[feature] - mean) / std)
        interpretation = FEATURE_INTERPRETATIONS.get(feature, "notable gait feature")
        zscores.append((feature, z, interpretation))
    zscores.sort(key=lambda item: abs(item[1]), reverse=True)
    return zscores[:top_n]


def candidates_from_evidence(evidence: list[tuple[str, float, str]], min_candidates: int = 2) -> list[str]:
    candidates = []
    for feature, _, _ in evidence:
        candidate = FEATURE_CANDIDATES.get(feature)
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    fallback = [
        "weight-bearing avoidance strategy",
        "ankle push-off function",
        "balance and frontal-plane foot control",
    ]
    for item in fallback:
        if len(candidates) >= min_candidates:
            break
        if item not in candidates:
            candidates.append(item)
    return candidates


def generate_final_analysis_report(
    output_path: Path,
    metrics: pd.DataFrame,
    group_summary: pd.DataFrame,
    subject_counts: pd.DataFrame,
    model_notes: str = "",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    best_row = metrics.sort_values(["macro_f1", "balanced_accuracy"], ascending=False).iloc[0]
    available_features = (
        group_summary.loc[group_summary["n"].gt(0), "feature"].drop_duplicates().tolist()
        if not group_summary.empty
        else []
    )
    available_gait_features = [feature for feature in FEATURE_COLUMNS if feature in available_features]
    missing_gait_features = [feature for feature in FEATURE_COLUMNS if feature not in available_features]
    gait_group_summary = group_summary[group_summary["feature"].isin(FEATURE_COLUMNS)] if not group_summary.empty else pd.DataFrame()
    top_group_features = (
        gait_group_summary.assign(abs_mean=gait_group_summary["mean"].abs())
        .sort_values("abs_mean", ascending=False)
        .groupby("label")
        .head(3)
        if not gait_group_summary.empty
        else pd.DataFrame()
    )

    lines = [
        "# Final Analysis Report",
        "",
        "## Problem Definition",
        "This project uses processed GRF/COP gait features to quantify impairment-group-related gait function patterns and suggest rehabilitation assessment priorities.",
        "",
        "## Data",
        "The main analysis unit is subject_id + trial_id + affected_side. Train/test separation is performed by subject_id to avoid trial leakage.",
        "",
        "### Label Counts",
        _markdown_table(subject_counts) if not subject_counts.empty else "No label counts available.",
        "",
        "## Methods",
        "Feature extraction summarizes the gait signals available in the provided data subset.",
        "Models include a most-frequent baseline plus interpretable and nonlinear classifiers when scikit-learn is available. Offline fallback models are used only to keep the demo runnable.",
        "",
        "## Available/Unavailable Features",
        f"Available gait features in this run: {', '.join(available_gait_features) if available_gait_features else 'none'}.",
        f"Unavailable because the corresponding AP/COP files were not included: {', '.join(missing_gait_features) if missing_gait_features else 'none'}.",
        "",
        "## Results",
        _markdown_table(metrics),
        "",
        f"Best model by macro F1: `{best_row['model']}` with macro_f1={best_row['macro_f1']:.3f} and balanced_accuracy={best_row['balanced_accuracy']:.3f}.",
        "",
        "### Group Feature Highlights",
        _markdown_table(top_group_features[["label", "feature", "mean", "std", "n"]])
        if not top_group_features.empty
        else "No group feature highlights available.",
        "",
        "## Model Interpretation",
        f"`used_sklearn` in this run: {bool(best_row.get('used_sklearn', False))}. Feature importance and coefficient tables should be read as screening support for gait-function patterns, not clinical conclusions.",
        "",
        "## SIAT Reference Note",
        "SIAT-LLMD is kept separate from the GaitRec classifier and can only provide auxiliary EMG/torque timing context when inspected sample files are available.",
        "",
        "## Limits",
        "GaitRec does not include EMG, so muscle-level mechanisms are not inferred from this dataset. SIAT-LLMD is used only as healthy-reference context for EMG/torque timing.",
        "",
        "## Future Direction",
        "Add real processed GaitRec files, compare speed-normalized features, and use SIAT walking samples for a limited EMG/torque reference figure.",
    ]
    if model_notes:
        lines.extend(["", "## Runtime Notes", model_notes])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _select_example_row(features: pd.DataFrame) -> pd.Series:
    non_healthy = features[features["label"].astype(str) != "Healthy"]
    if not non_healthy.empty:
        return non_healthy.iloc[0]
    return features.iloc[0]


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    printable = df.copy()
    for col in printable.columns:
        if pd.api.types.is_float_dtype(printable[col]):
            printable[col] = printable[col].map(lambda value: f"{value:.3f}")
        else:
            printable[col] = printable[col].astype(str)

    headers = [str(col) for col in printable.columns]
    rows = printable.astype(str).values.tolist()
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    header_line = "| " + " | ".join(header.ljust(widths[index]) for index, header in enumerate(headers)) + " |"
    sep_line = "| " + " | ".join("-" * widths[index] for index in range(len(headers))) + " |"
    body = [
        "| " + " | ".join(row[index].ljust(widths[index]) for index in range(len(headers))) + " |"
        for row in rows
    ]
    return "\n".join([header_line, sep_line, *body])


def _assert_no_forbidden_terms(report: str) -> None:
    lowered = report.lower()
    found = [term for term in FORBIDDEN_REPORT_TERMS if term.lower() in lowered]
    if found:
        raise ValueError(f"Report contains forbidden terms: {found}")

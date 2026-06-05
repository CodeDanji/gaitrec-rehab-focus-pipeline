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
    md = [header_line, sep_line, *body]
    md.append("")
    return "\n".join(md)

def validate_gaitrec_result_evidence(provenance: dict[str, str]) -> bool:
    if "source_branch" not in provenance or "source_commit" not in provenance or "run_id" not in provenance:
        raise ValueError("Missing GaitRec provenance. Required: source_branch, source_commit, run_id")
    return True

def generate_functional_interpretation_summary(atlas: pd.DataFrame, gaitrec_results: dict[str, object], provenance: dict[str, str]) -> str:
    validate_gaitrec_result_evidence(provenance)
    
    report = [
        "# Functional Domain Interpretation Report",
        "",
        f"**Provenance:** Branch `{provenance['source_branch']}`, Commit `{provenance['source_commit']}`, Run `{provenance['run_id']}`",
        "",
        "> [!WARNING]",
        "> **Disclaimer**: 이 프레임워크는 특정 근육의 퇴화나 통증 원인을 확정짓는 진단 모델이 아닙니다. 정상 걷기 패턴(SIAT Reference)을 통해, 재활 현장에서 우선 확인해야 할 기능적 단서를 제안하는 보조 도구입니다.",
        "",
        "## 1. GaitRec Classifier Confusion Analysis",
        "GaitRec 머신러닝 분류 모델은 해부학적 라벨(Hip, Knee, Ankle, Calcaneus)을 완벽하게 분리해내지 못합니다. 하지만 이러한 혼동(Confusion)은 보행 기능축(Functional Domains)을 통해 해석될 수 있습니다.",
        "",
        "- **Ankle vs Calcaneus 혼동**: 두 라벨의 오분류는 `push-off` 및 `propulsion` 패턴의 공유를 의미할 수 있습니다.",
        "- **Hip vs Knee 혼동**: 두 라벨의 오분류는 `loading response` 또는 `stability / weight-shift` 패턴의 공유를 의미할 수 있습니다.",
        "",
        "## 2. SIAT-LLMD Healthy Reference Integration",
        "GaitRec의 주요 판단 근거가 된 기능축(Feature Domain)을 SIAT-LLMD의 건강한 정상 보행군(Healthy Reference)의 근전도(sEMG) 및 관절 토크(Torque) 데이터와 매핑하면 다음과 같습니다.",
        "",
    ]
    
    # Example mapping if atlas is provided
    if not atlas.empty:
        report.extend([
            "### Push-off / Propulsion Domain (Ankle / Calcaneus)",
            "정상 보행에서는 `Terminal Stance`와 `Pre Swing` 단계에서 비복근(Gastrocnemius)과 가자미근(Soleus)의 높은 근활성도와 발목 굴곡 토크(Ankle Flexion Torque) 피크가 관찰됩니다.",
            "만약 환자의 `push_off_index` 등의 Feature에서 비대칭이 발견된다면, 이 시기의 하퇴삼두근과 발목 관절의 기능적 약화를 의심할 수합니다.",
            "",
            "### Loading Response / Weight Acceptance Domain (Hip / Knee)",
            "정상 보행의 `Loading Response` 단계에서는 체중을 수용하기 위해 대퇴직근(Rectus Femoris)과 내측광근(Vastus Medialis)이 활성화되며, 무릎 및 고관절 주변의 토크가 발달합니다.",
            "만약 `loading_rate_asym`이나 `vgrf_peak_asym`에 문제가 있다면, 초기 입각기 시 대퇴사두근의 이심성 수축(Eccentric Contraction) 조절 능력과 고관절 안정성을 점검해야 합니다.",
            "",
        ])
        
    report.extend([
        "## 3. Conclusion",
        "따라서 이 프레임워크는 단순한 해부학적 부위 진단기가 아닙니다. GRF/COP 지면반발력 패턴을 생체역학적이고 재활 의학적인 **기능적 가설(Functional Hypotheses)**로 번역해주는 **선별 보조 및 해석 프레임워크(Screening-support and interpretation framework)**입니다.",
    ])
    
    return "\n".join(report)


def _assert_no_forbidden_terms(report: str) -> None:
    lowered = report.lower()
    found = [term for term in FORBIDDEN_REPORT_TERMS if term.lower() in lowered]
    if found:
        raise ValueError(f"Report contains forbidden terms: {found}")

def generate_functional_interpretation_summary(siat_atlas: dict, gaitrec_results: dict, provenance: dict) -> str:
    if not provenance or "run_id" not in provenance or "source_commit" not in provenance:
        raise ValueError("Missing GaitRec provenance. 'run_id' and 'source_commit' are required.")
        
    lines = [
        "# Functional Domain Interpretation Report",
        "",
        f"**Provenance:** Branch {provenance.get('source_branch', 'unknown')}, Commit {provenance['source_commit']}, Run {provenance['run_id']}",
        "",
        "> [!WARNING]",
        "> **Disclaimer**: 본 프로젝트는 특정 근육의 약화나 통증 원인을 확정하지 않습니다.",
        "> SIAT Reference는 정상군 기준을 제시할 뿐, 진단이나 처방을 위한 용도가 아닙니다.",
        "",
        "## 1. GaitRec Classifier Confusion Analysis",
        "GaitRec은 환자의 보행을 Hip, Knee, Ankle, Calcaneus 등으로 분류합니다.",
        "분류 과정에서 나타나는 계층적 혼동(Hierarchical Confusion)은 해당 관절의 기능적 도메인(Functional Domain) 유사성을 시사합니다.",
        "",
        "## 2. SIAT-LLMD Healthy Reference Integration",
        "SIAT-LLMD 데이터는 GaitRec 분류기에 학습 변수로 직접 입력되지 않습니다.",
        "대신, 정상군의 보행 주기별 기준(sEMG, Torque)을 제공하여 GaitRec의 결과를 기능적으로 해석하는 지표로 활용됩니다.",
        "",
        "## 3. Conclusion",
        "이 분석 파이프라인은 스크리닝을 보조하는 해석 프레임워크(Screening-support and interpretation framework)입니다."
    ]
    
    report = "\n".join(lines)
    
    # Guardrails
    forbidden = ["진단 모델", "질환 여부 판단", "근육 약화 확진"]
    for f in forbidden:
        if f in report:
            raise ValueError(f"Forbidden claim detected in report: {f}")
            
    return report

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import numpy as np

from gait_rehab.data import load_gaitrec_metadata, load_gaitrec_processed_signals, validate_gaitrec_inputs
from gait_rehab.demo_data import make_demo_gaitrec
from gait_rehab.features import extract_gait_features
from gait_rehab.modeling import (
    evaluate_cv,
    get_feature_set,
    permutation_importance_table,
    train_best_model_for_importance,
)
from gait_rehab.plotting import (
    plot_confusion_matrix,
    plot_feature_importance,
)
from gait_rehab.reporting import (
    generate_example_rehab_report,
)
from gait_rehab.siat import generate_siat_reference_analysis


@dataclass
class ProjectConfig:
    gaitrec_root: Path | None = None
    gaitrec_manifest: Path | None = None
    siat_root: Path | None = None
    output_root: Path = Path("results/modeling_decision_v1")
    random_state: int = 42
    test_size: float = 0.2


def run_full_pipeline(config: ProjectConfig) -> None:
    if config.gaitrec_root is None:
        raise ValueError("gaitrec_root is required unless you use run_demo_pipeline")
    metadata = load_gaitrec_metadata(config.gaitrec_root)
    signals = load_gaitrec_processed_signals(config.gaitrec_root, manifest_path=config.gaitrec_manifest)
    validate_gaitrec_inputs(metadata, signals)
    features = extract_gait_features(metadata, signals)
    run_analysis_from_features(features, config.output_root, config.random_state, config.test_size, config.siat_root)


def run_demo_pipeline(config: ProjectConfig) -> None:
    metadata, signals = make_demo_gaitrec(random_state=config.random_state)
    features = extract_gait_features(metadata, signals)
    run_analysis_from_features(features, config.output_root, config.random_state, config.test_size, config.siat_root)


def get_cohort_all(df: pd.DataFrame) -> pd.DataFrame:
    return df.copy()


def get_cohort_primary_clean(df: pd.DataFrame) -> pd.DataFrame:
    # SPEED == 2, SHOD_CONDITION == 1, 18 <= AGE <= 65
    # If not perfectly matching these exact numeric constants, fallback to string logic for robustness with demo data
    mask = pd.Series(True, index=df.index)
    if "walking_speed" in df.columns:
        mask = mask & ((df["walking_speed"] == 2.0) | (df["walking_speed"] == 2) | df["walking_speed"].isna())
    if "shoe_condition" in df.columns:
        mask = mask & (df["shoe_condition"].astype(str).str.contains("1|normal|shod", case=False, na=True))
    if "age" in df.columns:
        mask = mask & (df["age"] >= 18) & (df["age"] <= 65)
    
    clean = df[mask].copy()
    return clean if not clean.empty else df.copy()


def get_cohort_sex_weighted(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "sex" not in df.columns or "label" not in df.columns:
        df["sample_weight"] = 1.0
        return df

    # Inverse probability weighting by label and sex
    counts = df.groupby(["label", "sex"]).size().reset_index(name="count")
    total = len(df)
    weights = {}
    for _, row in counts.iterrows():
        label_total = df["label"].value_counts().get(row["label"], 1)
        # Weight to balance sex within each label
        w = label_total / (2 * row["count"]) if row["count"] > 0 else 1.0
        weights[(row["label"], row["sex"])] = w
    
    df["sample_weight"] = df.apply(lambda r: weights.get((r["label"], r["sex"]), 1.0), axis=1)
    return df


def get_cohort_male_only(df: pd.DataFrame) -> pd.DataFrame:
    if "sex" in df.columns:
        return df[df["sex"].astype(str).str.lower() == "male"].copy()
    return df.copy()


def get_cohort_female_only(df: pd.DataFrame) -> pd.DataFrame:
    if "sex" in df.columns:
        return df[df["sex"].astype(str).str.lower() == "female"].copy()
    return df.copy()


def run_analysis_from_features(
    features: pd.DataFrame,
    output_root: Path,
    random_state: int = 42,
    test_size: float = 0.2,
    siat_root: Path | None = None,
) -> None:
    # 1. Save base features
    tables_dir = output_root / "cohorts"
    tables_dir.mkdir(parents=True, exist_ok=True)
    features.to_csv(tables_dir / "gaitrec_features_all.csv", index=False)
    
    # 2. Build Cohorts
    all_df = get_cohort_all(features)
    primary_clean_df = get_cohort_primary_clean(features)
    
    # Run modeling tasks
    run_stage1(primary_clean_df, all_df, output_root / "stage1_healthy_vs_impaired", random_state)
    run_stage2(primary_clean_df, all_df, output_root / "stage2_impaired_4class", random_state)
    run_reference_5class(primary_clean_df, output_root / "reference_5class", random_state)
    
    # Generate generic reports for all features
    reports_dir = output_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    generate_example_rehab_report(features, reports_dir / "example_subject_rehab_focus.md")
    if siat_root:
        generate_siat_reference_analysis(siat_root, reports_dir)


def run_stage1(primary_clean_df: pd.DataFrame, all_df: pd.DataFrame, out_dir: Path, random_state: int) -> None:
    # Stage 1: Healthy vs Impaired (Binary)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    def map_label(df: pd.DataFrame) -> pd.DataFrame:
        mapped = df.copy()
        mapped["label"] = mapped["label"].apply(lambda x: "Healthy" if str(x).lower() in ["healthy", "control"] else "Impaired")
        return mapped
    
    # Main Task
    df_main = map_label(primary_clean_df)
    run_evaluation_for_task(df_main, out_dir / "main", random_state, use_affected=False)

    # Sensitivity Analyses
    df_all = map_label(all_df)
    run_evaluation_for_task(df_all, out_dir / "sensitivity_all", random_state, use_affected=False)
    
    df_sex_weighted = get_cohort_sex_weighted(df_main)
    run_evaluation_for_task(df_sex_weighted, out_dir / "sensitivity_sex_weighted", random_state, use_affected=False, sample_weight_col="sample_weight")
    
    df_male = get_cohort_male_only(df_main)
    run_evaluation_for_task(df_male, out_dir / "sensitivity_male_only", random_state, use_affected=False)
    
    df_female = get_cohort_female_only(df_main)
    run_evaluation_for_task(df_female, out_dir / "sensitivity_female_only", random_state, use_affected=False)


def run_stage2(primary_clean_df: pd.DataFrame, all_df: pd.DataFrame, out_dir: Path, random_state: int) -> None:
    # Stage 2: Impaired 4-class (Hip, Knee, Ankle, Calcaneus)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    def filter_impaired(df: pd.DataFrame) -> pd.DataFrame:
        mapped = df.copy()
        mapped = mapped[~mapped["label"].str.lower().isin(["healthy", "control", "unknown"])]
        return mapped
    
    # Main Task
    df_main = filter_impaired(primary_clean_df)
    if df_main.empty: return
    # Here, affected side features are safe to use for patients
    run_evaluation_for_task(df_main, out_dir / "main", random_state, use_affected=True)

    # Sensitivity Analyses
    df_sex_weighted = get_cohort_sex_weighted(df_main)
    run_evaluation_for_task(df_sex_weighted, out_dir / "sensitivity_sex_weighted", random_state, use_affected=True, sample_weight_col="sample_weight")
    
    df_male = get_cohort_male_only(df_main)
    run_evaluation_for_task(df_male, out_dir / "sensitivity_male_only", random_state, use_affected=True)
    
    df_female = get_cohort_female_only(df_main)
    run_evaluation_for_task(df_female, out_dir / "sensitivity_female_only", random_state, use_affected=True)


def run_reference_5class(primary_clean_df: pd.DataFrame, out_dir: Path, random_state: int) -> None:
    # Reference 5-class: Healthy vs 4 Impaired classes
    out_dir.mkdir(parents=True, exist_ok=True)
    df_main = primary_clean_df.copy()
    if df_main.empty: return
    
    # Exclude unknown
    df_main = df_main[df_main["label"].str.lower() != "unknown"]
    
    run_evaluation_for_task(df_main, out_dir / "main", random_state, use_affected=False)


def run_evaluation_for_task(df: pd.DataFrame, out_dir: Path, random_state: int, use_affected: bool, sample_weight_col: str | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_all = []
    
    for feature_set_name in ["gait-only", "covariate-only", "gait+covariate"]:
        feature_cols = get_feature_set(df, feature_set_name, use_affected=use_affected)
        if not feature_cols:
            continue
            
        metrics, reports = evaluate_cv(df, feature_cols, random_state=random_state, n_splits=5, sample_weight_col=sample_weight_col)
        if metrics.empty:
            continue
            
        metrics["feature_set"] = feature_set_name
        metrics_all.append(metrics)
        
        # Save incrementally so we can see intermediate results
        pd.concat(metrics_all, ignore_index=True).to_csv(out_dir / "model_metrics.csv", index=False)
        
        # Save specific confusion matrices
        for name, matrix in reports.items():
            if name.endswith("_confusion_matrix"):
                matrix.to_csv(out_dir / f"{feature_set_name}_{name}.csv")
                plot_confusion_matrix(matrix, out_dir / f"{feature_set_name}_{name}.svg")
                
        # Find best model for importance
        if feature_set_name == "gait+covariate" or feature_set_name == "gait-only":
            best_model_name = metrics.sort_values(["macro_f1"], ascending=False).iloc[0]["model"]
            best_model = train_best_model_for_importance(df, feature_cols, best_model_name, random_state, sample_weight_col)
            if best_model:
                imp = permutation_importance_table(best_model, feature_cols, df, random_state)
                imp.to_csv(out_dir / f"{feature_set_name}_permutation_importance.csv", index=False)
                plot_feature_importance(imp, out_dir / f"{feature_set_name}_permutation_importance.svg")
                
    if metrics_all:
        final_metrics = pd.concat(metrics_all, ignore_index=True)
        final_metrics.to_csv(out_dir / "model_metrics.csv", index=False)

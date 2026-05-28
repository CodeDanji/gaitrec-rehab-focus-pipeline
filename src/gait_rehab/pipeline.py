from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from gait_rehab.data import load_gaitrec_metadata, load_gaitrec_processed_signals, validate_gaitrec_inputs
from gait_rehab.demo_data import make_demo_gaitrec
from gait_rehab.features import extract_gait_features
from gait_rehab.modeling import (
    evaluate_models,
    logistic_coefficients_table,
    permutation_importance_table,
    split_by_subject,
    train_models,
)
from gait_rehab.plotting import (
    plot_confusion_matrix,
    plot_feature_importance,
    plot_group_ap_impulses,
    plot_group_cop,

    plot_group_mean_vgrf,
    plot_group_summary,
    plot_metric_bars,
    write_workflow_svg,
)
from gait_rehab.reporting import (
    generate_example_rehab_report,
    generate_final_analysis_report,
    subject_trial_counts,
    summarize_groups,
)
from gait_rehab.siat import generate_siat_reference_analysis


@dataclass
class ProjectConfig:
    gaitrec_root: Path | None = None
    gaitrec_manifest: Path | None = None
    siat_root: Path | None = None
    output_root: Path = Path("results")
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


def run_analysis_from_features(
    features: pd.DataFrame,
    output_root: Path,
    random_state: int = 42,
    test_size: float = 0.2,
    siat_root: Path | None = None,
) -> None:
    tables_dir = output_root / "tables"
    figures_dir = output_root / "figures"
    reports_dir = output_root / "reports"
    for directory in [tables_dir, figures_dir, reports_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    features.to_csv(tables_dir / "gaitrec_features.csv", index=False)
    group_summary = summarize_groups(features)
    counts = subject_trial_counts(features)
    group_summary.to_csv(tables_dir / "group_feature_summary.csv", index=False)
    counts.to_csv(tables_dir / "label_counts.csv", index=False)

    train_df, test_df = split_by_subject(features, test_size=test_size, random_state=random_state)
    model_bundle = train_models(train_df, random_state=random_state)
    metrics, reports = evaluate_models(model_bundle, test_df)
    metrics.to_csv(tables_dir / "model_metrics.csv", index=False)

    for name, report in reports.items():
        report.to_csv(tables_dir / f"{name}.csv")

    best_model = metrics.sort_values(["macro_f1", "balanced_accuracy"], ascending=False).iloc[0]["model"]
    importance = permutation_importance_table(model_bundle, str(best_model), test_df, random_state=random_state)
    importance.to_csv(tables_dir / "permutation_importance.csv", index=False)
    logistic_coefficients_table(model_bundle).to_csv(tables_dir / "logistic_coefficients.csv", index=False)

    write_workflow_svg(figures_dir / "workflow.svg")
    plot_metric_bars(metrics, figures_dir / "model_metrics.svg")
    plot_feature_importance(importance, figures_dir / "permutation_importance.svg")
    confusion = reports.get(f"{best_model}_confusion_matrix")
    if confusion is not None:
        plot_confusion_matrix(confusion, figures_dir / "confusion_matrix.svg")
    summary_feature = "push_off_index"
    if group_summary[group_summary["feature"].eq(summary_feature) & group_summary["n"].gt(0)].empty:
        summary_feature = "vgrf_peak_aff"
    plot_group_summary(group_summary, figures_dir / f"group_{summary_feature}_summary.svg", feature=summary_feature)
    plot_group_mean_vgrf(group_summary, figures_dir / "group_mean_vgrf_curve.svg")
    plot_group_ap_impulses(group_summary, figures_dir / "group_ap_impulse_comparison.svg")
    plot_group_cop(group_summary, figures_dir / "group_cop_comparison.svg")

    generate_example_rehab_report(features, reports_dir / "example_subject_rehab_focus.md")
    generate_final_analysis_report(
        reports_dir / "final_analysis_report.md",
        metrics=metrics,
        group_summary=group_summary,
        subject_counts=counts,
        model_notes=model_bundle.notes,
    )
    generate_siat_reference_analysis(siat_root, output_root)

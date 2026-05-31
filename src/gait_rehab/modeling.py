from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Any
import warnings

import numpy as np
import pandas as pd


NEUTRAL_GAIT_FEATURES = [
    "vgrf_peak_mean",
    "vgrf_peak_max",
    "vgrf_peak_diff",
    "loading_rate_mean",
    "loading_rate_max",
    "loading_rate_sym",
    "ap_braking_impulse_mean",
    "ap_braking_impulse_sym",
    "ap_propulsion_impulse_mean",
    "ap_propulsion_impulse_sym",
    "push_off_index",
    "cop_ap_range_mean",
    "cop_ap_range_sym",
    "cop_ml_range_mean",
    "cop_ml_range_sym",
    "cop_path_length_mean",
]

AFFECTED_GAIT_FEATURES = [
    "vgrf_peak_aff",
    "vgrf_peak_unaff",
    "vgrf_peak_asym",
    "loading_rate_asym",
    "ap_braking_impulse_asym",
    "ap_propulsion_impulse_asym",
    "cop_ap_range_aff",
    "cop_ml_range_aff",
    "cop_path_length_aff",
    "cop_ap_range_asym",
    "cop_ml_range_asym",
]

COVARIATE_NUMERIC = ["age", "height", "body_weight", "body_mass", "bmi", "walking_speed", "shoe_size"]
COVARIATE_CATEGORICAL = ["sex", "shoe_condition"]

VGRF_WAVEFORM_FEATURES = [f"vgrf_left_{i}" for i in range(101)] + [f"vgrf_right_{i}" for i in range(101)]


def get_feature_set(df: pd.DataFrame, set_type: str, use_affected: bool = False) -> list[str]:
    features: list[str] = []
    if set_type in ["gait-only", "gait+covariate"]:
        features.extend(NEUTRAL_GAIT_FEATURES)
        features.extend(VGRF_WAVEFORM_FEATURES)
        if use_affected:
            features.extend(AFFECTED_GAIT_FEATURES)
    if set_type in ["covariate-only", "gait+covariate"]:
        features.extend(COVARIATE_NUMERIC)
        features.extend(COVARIATE_CATEGORICAL)
    
    available = []
    for col in features:
        if col in df.columns:
            if col in COVARIATE_CATEGORICAL:
                if df[col].notna().any():
                    available.append(col)
            elif pd.to_numeric(df[col], errors="coerce").notna().any():
                available.append(col)
    return available


class PredictsLabels(Protocol):
    def predict(self, x: pd.DataFrame) -> np.ndarray:
        ...


@dataclass
class ModelBundle:
    models: dict[str, PredictsLabels]
    feature_columns: list[str]
    used_sklearn: bool
    notes: str = ""


def confusion_matrix_frame(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str] | None = None) -> pd.DataFrame:
    labels = labels or sorted(set(y_true) | set(y_pred))
    matrix = pd.DataFrame(0, index=labels, columns=labels, dtype=int)
    for true, pred in zip(y_true, y_pred):
        if true not in matrix.index:
            matrix.loc[true] = 0
        if pred not in matrix.columns:
            matrix[pred] = 0
        matrix.loc[true, pred] += 1
    matrix.index.name = "true_label"
    matrix.columns.name = "predicted_label"
    return matrix


def classification_report_frame(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str] | None = None) -> pd.DataFrame:
    labels = labels or sorted(set(y_true) | set(y_pred))
    rows = []
    for label in labels:
        tp = int(np.sum((y_true == label) & (y_pred == label)))
        fp = int(np.sum((y_true != label) & (y_pred == label)))
        fn = int(np.sum((y_true == label) & (y_pred != label)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "label": label,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": int(np.sum(y_true == label)),
            }
        )
    return pd.DataFrame(rows)


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str] | None = None) -> float:
    labels = labels or sorted(set(y_true) | set(y_pred))
    recalls = []
    for label in labels:
        mask = y_true == label
        if np.sum(mask) == 0:
            continue
        recalls.append(float(np.mean(y_pred[mask] == label)))
    return float(np.mean(recalls)) if recalls else 0.0


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str] | None = None) -> float:
    report = classification_report_frame(y_true, y_pred, labels=labels)
    return float(report["f1"].mean()) if not report.empty else 0.0


def _build_sklearn_models(feature_columns: list[str], random_state: int) -> dict[str, Any]:
    from sklearn.compose import ColumnTransformer
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.decomposition import PCA

    num_cols = [c for c in NEUTRAL_GAIT_FEATURES + AFFECTED_GAIT_FEATURES + COVARIATE_NUMERIC if c in feature_columns]
    cat_cols = [c for c in COVARIATE_CATEGORICAL if c in feature_columns]
    wave_cols = [c for c in VGRF_WAVEFORM_FEATURES if c in feature_columns]

    def make_preprocessor() -> ColumnTransformer:
        transformers = []
        if num_cols:
            transformers.append(("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_cols))
        if cat_cols:
            transformers.append(("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent", fill_value="unknown")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_cols))
        if wave_cols:
            transformers.append(("wave", Pipeline([("imputer", SimpleImputer(strategy="median")), ("pca", PCA(n_components=0.95, random_state=random_state))]), wave_cols))
        
        return ColumnTransformer(transformers=transformers)

    return {
        "dummy": Pipeline([("preprocess", make_preprocessor()), ("model", DummyClassifier(strategy="most_frequent"))]),
        "logistic_regression": Pipeline([("preprocess", make_preprocessor()), ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state))]),
        "random_forest": Pipeline([("preprocess", make_preprocessor()), ("model", RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=5, class_weight="balanced_subsample", random_state=random_state, n_jobs=-1))]),
    }


def _build_numpy_fallback_models(feature_columns: list[str]) -> dict[str, Any]:
    return {} # Skipping numpy fallback fully for CV to save space, but pipeline uses sklearn


def evaluate_cv(df: pd.DataFrame, feature_columns: list[str], random_state: int = 42, n_splits: int = 5, sample_weight_col: str | None = None) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    if "label" not in df.columns or "subject_id" not in df.columns:
        raise ValueError("evaluate_cv requires label and subject_id columns")

    y = df["label"].astype(str).to_numpy()
    groups = df["subject_id"].astype(str).to_numpy()
    labels = sorted(np.unique(y))
    subjects = np.unique(groups)

    if len(subjects) < n_splits:
        warnings.warn(f"Not enough subjects ({len(subjects)}) for {n_splits} splits. Skipping.")
        return pd.DataFrame(), {}

    try:
        from sklearn.model_selection import StratifiedGroupKFold
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = list(cv.split(df[feature_columns], y, groups))
        used_sklearn = True
    except ModuleNotFoundError:
        warnings.warn("scikit-learn not found. Using a simple custom subject split instead of CV.")
        subjects = np.unique(groups)
        rng = np.random.default_rng(random_state)
        rng.shuffle(subjects)
        n_test = max(1, len(subjects) // n_splits)
        test_subs = set(subjects[:n_test])
        test_mask = df["subject_id"].astype(str).isin(test_subs)
        splits = [(np.where(~test_mask)[0], np.where(test_mask)[0])]
        used_sklearn = False

    rows = []
    reports = {}

    if used_sklearn:
        models = _build_sklearn_models(feature_columns, random_state)
    else:
        models = _build_numpy_fallback_models(feature_columns)

    if not models:
        return pd.DataFrame(), {}

    for model_name, model_template in models.items():
        from sklearn.base import clone
        y_true_all = []
        y_pred_all = []

        for train_idx, test_idx in splits:
            model = clone(model_template) if used_sklearn else model_template
            x_train = df.iloc[train_idx][feature_columns]
            y_train = y[train_idx]
            w_train = df.iloc[train_idx][sample_weight_col].to_numpy() if sample_weight_col and sample_weight_col in df.columns else None
            x_test = df.iloc[test_idx][feature_columns]
            y_test = y[test_idx]

            if w_train is not None and used_sklearn:
                model.fit(x_train, y_train, model__sample_weight=w_train)
            else:
                model.fit(x_train, y_train)
            
            y_pred = np.asarray(model.predict(x_test)).astype(str)
            
            y_true_all.extend(y_test)
            y_pred_all.extend(y_pred)

        y_true_all = np.array(y_true_all)
        y_pred_all = np.array(y_pred_all)
        
        matrix = confusion_matrix_frame(y_true_all, y_pred_all, labels=labels)
        report = classification_report_frame(y_true_all, y_pred_all, labels=labels)
        
        rows.append({
            "model": model_name,
            "balanced_accuracy": balanced_accuracy(y_true_all, y_pred_all, labels=labels),
            "macro_f1": macro_f1(y_true_all, y_pred_all, labels=labels),
            "support": len(y_true_all),
            "used_sklearn": used_sklearn,
        })
        reports[model_name] = report
        reports[f"{model_name}_confusion_matrix"] = matrix

    return pd.DataFrame(rows), reports


def train_best_model_for_importance(df: pd.DataFrame, feature_columns: list[str], best_model_name: str, random_state: int = 42, sample_weight_col: str | None = None) -> Any:
    models = _build_sklearn_models(feature_columns, random_state)
    model = models.get(best_model_name)
    if not model:
        return None
    
    x = df[feature_columns]
    y = df["label"].astype(str).to_numpy()
    w = df[sample_weight_col].to_numpy() if sample_weight_col and sample_weight_col in df.columns else None
    
    if w is not None:
        model.fit(x, y, model__sample_weight=w)
    else:
        model.fit(x, y)
    return model


def permutation_importance_table(
    model: Any,
    feature_columns: list[str],
    test_df: pd.DataFrame,
    random_state: int = 42,
    repeats: int = 5,
) -> pd.DataFrame:
    if model is None:
        return pd.DataFrame(columns=["feature", "importance_mean", "importance_std"])

    x = test_df[feature_columns].copy()
    y_true = test_df["label"].astype(str).to_numpy()
    baseline = macro_f1(y_true, np.asarray(model.predict(x)).astype(str))
    rng = np.random.default_rng(random_state)
    rows: list[dict[str, object]] = []

    for feature in feature_columns:
        scores = []
        for _ in range(repeats):
            shuffled = x.copy()
            shuffled[feature] = rng.permutation(shuffled[feature].to_numpy())
            score = macro_f1(y_true, np.asarray(model.predict(shuffled)).astype(str))
            scores.append(baseline - score)
        rows.append(
            {
                "feature": feature,
                "importance_mean": float(np.mean(scores)),
                "importance_std": float(np.std(scores)),
            }
        )

    return pd.DataFrame(rows).sort_values("importance_mean", ascending=False)

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd


NUMERIC_FEATURES = [
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
    "walking_speed",
    "age",
    "height",
    "weight",

]

CATEGORICAL_FEATURES = ["sex", "shoe_condition"]


class PredictsLabels(Protocol):
    def predict(self, x: pd.DataFrame) -> np.ndarray:
        ...


@dataclass
class ModelBundle:
    models: dict[str, PredictsLabels]
    feature_columns: list[str]
    used_sklearn: bool
    notes: str = ""


def available_feature_columns(df: pd.DataFrame) -> list[str]:
    available: list[str] = []
    for col in NUMERIC_FEATURES:
        if col in df.columns and pd.to_numeric(df[col], errors="coerce").notna().any():
            available.append(col)
    for col in CATEGORICAL_FEATURES:
        if col in df.columns and df[col].notna().any():
            available.append(col)
    return available


def split_by_subject(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "subject_id" not in df.columns:
        raise ValueError("split_by_subject requires a subject_id column")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    subjects = np.asarray(sorted(df["subject_id"].astype(str).unique()))
    if subjects.size < 2:
        raise ValueError("At least two subjects are required for subject-level split")

    rng = np.random.default_rng(random_state)
    shuffled = subjects.copy()
    rng.shuffle(shuffled)
    n_test = max(1, int(round(subjects.size * test_size)))
    n_test = min(n_test, subjects.size - 1)
    test_subjects = set(shuffled[:n_test])

    test_mask = df["subject_id"].astype(str).isin(test_subjects)
    return df.loc[~test_mask].reset_index(drop=True), df.loc[test_mask].reset_index(drop=True)


def train_models(train_df: pd.DataFrame, random_state: int = 42) -> ModelBundle:
    feature_columns = available_feature_columns(train_df)
    if "label" not in train_df.columns:
        raise ValueError("train_models requires a label column")
    if not feature_columns:
        raise ValueError("No recognized feature columns are available for modeling")

    try:
        return _train_sklearn_models(train_df, feature_columns, random_state=random_state)
    except ModuleNotFoundError as exc:
        if exc.name not in {"sklearn", "scikit_learn"}:
            raise
        return _train_numpy_fallback_models(train_df, feature_columns)


def evaluate_models(bundle: ModelBundle, test_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    rows: list[dict[str, object]] = []
    reports: dict[str, pd.DataFrame] = {}
    y_true = test_df["label"].astype(str).to_numpy()
    labels = sorted(pd.Series(y_true).dropna().astype(str).unique())

    for model_name, model in bundle.models.items():
        y_pred = np.asarray(model.predict(test_df[bundle.feature_columns])).astype(str)
        matrix = confusion_matrix_frame(y_true, y_pred, labels=labels)
        report = classification_report_frame(y_true, y_pred, labels=labels)
        rows.append(
            {
                "model": model_name,
                "balanced_accuracy": balanced_accuracy(y_true, y_pred, labels=labels),
                "macro_f1": macro_f1(y_true, y_pred, labels=labels),
                "support": len(y_true),
                "used_sklearn": bundle.used_sklearn,
            }
        )
        reports[model_name] = report
        reports[f"{model_name}_confusion_matrix"] = matrix

    return pd.DataFrame(rows), reports


def permutation_importance_table(
    bundle: ModelBundle,
    model_name: str,
    test_df: pd.DataFrame,
    random_state: int = 42,
    repeats: int = 5,
) -> pd.DataFrame:
    if model_name not in bundle.models:
        return pd.DataFrame(columns=["feature", "importance_mean", "importance_std"])

    model = bundle.models[model_name]
    x = test_df[bundle.feature_columns].copy()
    y_true = test_df["label"].astype(str).to_numpy()
    baseline = macro_f1(y_true, np.asarray(model.predict(x)).astype(str))
    rng = np.random.default_rng(random_state)
    rows: list[dict[str, object]] = []

    for feature in bundle.feature_columns:
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


def logistic_coefficients_table(bundle: ModelBundle) -> pd.DataFrame:
    model = bundle.models.get("logistic_regression") or bundle.models.get("softmax_logistic_regression")
    if model is None:
        return pd.DataFrame(columns=["class", "feature", "coefficient"])

    if hasattr(model, "named_steps"):
        estimator = model.named_steps["model"]
        preprocessor = model.named_steps["preprocess"]
        feature_names = list(preprocessor.get_feature_names_out())
        rows = []
        for class_name, coefs in zip(estimator.classes_, estimator.coef_):
            for feature, coefficient in zip(feature_names, coefs):
                rows.append({"class": class_name, "feature": feature, "coefficient": coefficient})
        return pd.DataFrame(rows)

    if hasattr(model, "coefficients_"):
        feature_names = getattr(getattr(model, "preprocessor", None), "output_columns_", bundle.feature_columns)
        rows = []
        for class_name, coefs in zip(model.classes_, model.coefficients_):
            for feature, coefficient in zip(feature_names, coefs):
                rows.append({"class": class_name, "feature": feature, "coefficient": coefficient})
        return pd.DataFrame(rows)

    return pd.DataFrame(columns=["class", "feature", "coefficient"])


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


def _train_sklearn_models(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    random_state: int,
) -> ModelBundle:
    from sklearn.compose import ColumnTransformer
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numeric = [col for col in NUMERIC_FEATURES if col in feature_columns]
    categorical = [col for col in CATEGORICAL_FEATURES if col in feature_columns]
    def make_preprocessor() -> ColumnTransformer:
        return ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    numeric,
                ),
                (
                    "cat",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore")),
                        ]
                    ),
                    categorical,
                ),
            ]
        )

    models = {
        "dummy": Pipeline(
            [
                ("preprocess", make_preprocessor()),
                ("model", DummyClassifier(strategy="most_frequent")),
            ]
        ),
        "logistic_regression": Pipeline(
            [
                ("preprocess", make_preprocessor()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("preprocess", make_preprocessor()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=500,
                        max_depth=8,
                        min_samples_leaf=5,
                        class_weight="balanced_subsample",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }

    x_train = train_df[feature_columns]
    y_train = train_df["label"].astype(str)
    for model in models.values():
        model.fit(x_train, y_train)

    return ModelBundle(models=models, feature_columns=feature_columns, used_sklearn=True)


def _train_numpy_fallback_models(train_df: pd.DataFrame, feature_columns: list[str]) -> ModelBundle:
    x_train = train_df[feature_columns]
    y_train = train_df["label"].astype(str).to_numpy()
    preprocessor = BasicPreprocessor(feature_columns).fit(x_train)

    models: dict[str, PredictsLabels] = {
        "dummy": DummyMostFrequent().fit(x_train, y_train),
        "softmax_logistic_regression": SoftmaxClassifier(preprocessor=preprocessor).fit(x_train, y_train),
        "nearest_centroid_fallback": NearestCentroidClassifier(preprocessor=preprocessor).fit(x_train, y_train),
    }
    return ModelBundle(
        models=models,
        feature_columns=feature_columns,
        used_sklearn=False,
        notes="scikit-learn is not installed; used numpy fallback models for runnable demo output.",
    )


class DummyMostFrequent:
    def fit(self, _x: pd.DataFrame, y: np.ndarray) -> "DummyMostFrequent":
        counts = pd.Series(y).value_counts()
        self.label_ = str(counts.index[0])
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return np.asarray([self.label_] * len(x))


class BasicPreprocessor:
    def __init__(self, feature_columns: list[str]):
        self.feature_columns = feature_columns

    def fit(self, x: pd.DataFrame) -> "BasicPreprocessor":
        self.numeric_ = [col for col in NUMERIC_FEATURES if col in self.feature_columns]
        self.categorical_ = [col for col in CATEGORICAL_FEATURES if col in self.feature_columns]
        self.medians_ = x[self.numeric_].median(numeric_only=True).fillna(0.0) if self.numeric_ else pd.Series(dtype=float)
        filled = x[self.numeric_].fillna(self.medians_) if self.numeric_ else pd.DataFrame(index=x.index)
        self.means_ = filled.mean() if self.numeric_ else pd.Series(dtype=float)
        self.stds_ = filled.std().replace(0, 1.0).fillna(1.0) if self.numeric_ else pd.Series(dtype=float)
        self.categories_ = {
            col: sorted(x[col].fillna("missing").astype(str).unique().tolist()) for col in self.categorical_
        }
        self.output_columns_ = self.numeric_ + [
            f"{col}={category}" for col, categories in self.categories_.items() for category in categories
        ]
        return self

    def transform(self, x: pd.DataFrame) -> np.ndarray:
        arrays = []
        if self.numeric_:
            numeric = x[self.numeric_].fillna(self.medians_)
            arrays.append(((numeric - self.means_) / self.stds_).to_numpy(dtype=float))
        for col, categories in self.categories_.items():
            values = x[col].fillna("missing").astype(str)
            arrays.append(np.column_stack([(values == category).astype(float) for category in categories]))
        if not arrays:
            return np.empty((len(x), 0))
        return np.column_stack(arrays)


class SoftmaxClassifier:
    def __init__(self, preprocessor: BasicPreprocessor, learning_rate: float = 0.2, epochs: int = 800):
        self.preprocessor = preprocessor
        self.learning_rate = learning_rate
        self.epochs = epochs

    def fit(self, x: pd.DataFrame, y: np.ndarray) -> "SoftmaxClassifier":
        x_matrix = self.preprocessor.transform(x)
        x_matrix = np.column_stack([np.ones(len(x_matrix)), x_matrix])
        self.classes_ = np.asarray(sorted(pd.Series(y).unique()))
        y_index = np.asarray([np.where(self.classes_ == value)[0][0] for value in y])
        y_one_hot = np.eye(len(self.classes_))[y_index]
        weights = np.zeros((x_matrix.shape[1], len(self.classes_)))

        class_counts = pd.Series(y).value_counts()
        sample_weight = np.asarray([1.0 / class_counts[value] for value in y], dtype=float)
        sample_weight = sample_weight / sample_weight.mean()

        for _ in range(self.epochs):
            logits = x_matrix @ weights
            logits -= logits.max(axis=1, keepdims=True)
            probs = np.exp(logits)
            probs /= probs.sum(axis=1, keepdims=True)
            gradient = x_matrix.T @ ((probs - y_one_hot) * sample_weight[:, None]) / len(x_matrix)
            weights -= self.learning_rate * gradient

        self.intercept_ = weights[0]
        self.coefficients_ = weights[1:].T
        self.weights_ = weights
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        x_matrix = self.preprocessor.transform(x)
        x_matrix = np.column_stack([np.ones(len(x_matrix)), x_matrix])
        logits = x_matrix @ self.weights_
        return self.classes_[np.argmax(logits, axis=1)]


class NearestCentroidClassifier:
    def __init__(self, preprocessor: BasicPreprocessor):
        self.preprocessor = preprocessor

    def fit(self, x: pd.DataFrame, y: np.ndarray) -> "NearestCentroidClassifier":
        matrix = self.preprocessor.transform(x)
        self.classes_ = np.asarray(sorted(pd.Series(y).unique()))
        self.centroids_ = np.vstack([matrix[y == label].mean(axis=0) for label in self.classes_])
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        matrix = self.preprocessor.transform(x)
        distances = ((matrix[:, None, :] - self.centroids_[None, :, :]) ** 2).sum(axis=2)
        return self.classes_[np.argmin(distances, axis=1)]

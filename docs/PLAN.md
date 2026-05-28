# GRF/COP + EMG 재활 초점 제안 프로젝트 최종 발표급 구현 계획

## Summary
- 목표는 GaitRec processed GRF/COP 데이터로 **근골격계 impairment group별 보행 기능 이상 패턴**을 분석하고, 설명 가능한 모델을 통해 **재활 평가에서 우선 확인할 기능 후보**를 제안하는 것이다.
- 최종 산출물은 재현 가능한 Python 분석 파이프라인, 결과 표, 발표용 그림, 개별 subject 리포트다.
- SIAT-LLMD는 GaitRec 모델에 직접 결합하지 않고, EMG/torque 관계를 설명하는 **보조 생리학 참고 분석**으로만 사용한다.

## Model Technical Spec
- 입력 단위:
  - 기본 분석 단위는 `subject_id + trial_id + affected_side`.
  - 학습/평가는 반드시 `subject_id` 기준 group split을 사용한다.
- 예측 target:
  - `label ∈ {Healthy, Hip, Knee, Ankle, Calcaneus}`
- 입력 feature:
  - GRF feature: `vgrf_peak_aff`, `vgrf_peak_unaff`, `vgrf_peak_asym`, `loading_rate_asym`, `ap_braking_impulse_asym`, `ap_propulsion_impulse_asym`, `push_off_index`
  - COP feature: `cop_ap_range_aff`, `cop_ml_range_aff`, `cop_path_length_aff`, `cop_ap_range_asym`, `cop_ml_range_asym`
  - metadata covariates: `walking_speed`, `age`, `sex`, `height`, `weight`, `shoe_condition`
- 전처리:
  - numeric feature: median imputation + standard scaling
  - categorical feature: most-frequent imputation + one-hot encoding
  - class imbalance: `class_weight="balanced"` 또는 balanced metric 사용
- 모델 구성:
  - Baseline: `DummyClassifier(strategy="most_frequent")`
  - Primary interpretable model: `LogisticRegression(multi_class="multinomial", class_weight="balanced")`
  - Secondary nonlinear model: `RandomForestClassifier(class_weight="balanced_subsample")`
  - Explanation: permutation importance, logistic coefficients, group별 feature summary, confusion matrix
- 평가 지표:
  - `balanced_accuracy`
  - `macro_f1`
  - class별 precision/recall/F1
  - confusion matrix
  - baseline 대비 성능 향상
- 금지:
  - trial-level random split 금지
  - SIAT-LLMD와 GaitRec을 합쳐 하나의 진단 모델로 학습 금지
  - “근육 약화 원인 확정”, “진단”, “처방” 표현 금지

## Model Flow
```text
1. Load GaitRec metadata and processed GRF/COP signals
2. Validate subject/trial/side keys
3. Map affected side and unaffected side
4. Normalize each signal to gait cycle percentage if needed
5. Extract GRF/COP features per trial
6. Merge features with metadata and impairment label
7. Split train/test by subject_id
8. Fit baseline model
9. Fit interpretable logistic regression
10. Fit secondary random forest model
11. Evaluate metrics and confusion matrix
12. Compute feature importance and group-level summaries
13. Generate subject-level rehab focus report
14. Generate presentation figures and final analysis report
15. Run SIAT sample analysis separately for EMG/torque reference figures
```

## Code Shape
```python
# project/src/pipeline.py

from dataclasses import dataclass
from pathlib import Path
import pandas as pd

@dataclass
class ProjectConfig:
    gaitrec_root: Path
    siat_root: Path | None
    output_root: Path
    random_state: int = 42
    test_size: float = 0.2

def run_full_pipeline(config: ProjectConfig) -> None:
    metadata = load_gaitrec_metadata(config.gaitrec_root)
    signals = load_gaitrec_processed_signals(config.gaitrec_root)

    validate_gaitrec_inputs(metadata, signals)

    features = extract_gait_features(metadata, signals)
    features.to_csv(config.output_root / "tables/gaitrec_features.csv", index=False)

    train_df, test_df = split_by_subject(features, test_size=config.test_size)

    model_bundle = train_models(train_df)
    metrics = evaluate_models(model_bundle, test_df)
    metrics.to_csv(config.output_root / "tables/model_metrics.csv", index=False)

    generate_model_figures(model_bundle, test_df, config.output_root / "figures")
    generate_group_summary(features, config.output_root)
    generate_example_rehab_reports(model_bundle, test_df, config.output_root)

    if config.siat_root is not None:
        generate_siat_reference_analysis(config.siat_root, config.output_root)
```

```python
# project/src/features.py

def extract_gait_features(metadata: pd.DataFrame, signals: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []

    for trial in iterate_trials(metadata, signals):
        aff = trial.affected_side
        unaff = "right" if aff == "left" else "left"

        row = {
            "subject_id": trial.subject_id,
            "trial_id": trial.trial_id,
            "label": trial.label,
            "affected_side": aff,
            "walking_speed": trial.walking_speed,
            "age": trial.age,
            "sex": trial.sex,
            "height": trial.height,
            "weight": trial.weight,
            "shoe_condition": trial.shoe_condition,
            "vgrf_peak_aff": peak(trial.vgrf[aff]),
            "vgrf_peak_unaff": peak(trial.vgrf[unaff]),
            "vgrf_peak_asym": asymmetry(peak(trial.vgrf[aff]), peak(trial.vgrf[unaff])),
            "loading_rate_asym": asymmetry(loading_rate(trial.vgrf[aff]), loading_rate(trial.vgrf[unaff])),
            "ap_braking_impulse_asym": asymmetry(braking_impulse(trial.ap_grf[aff]), braking_impulse(trial.ap_grf[unaff])),
            "ap_propulsion_impulse_asym": asymmetry(propulsion_impulse(trial.ap_grf[aff]), propulsion_impulse(trial.ap_grf[unaff])),
            "push_off_index": propulsion_impulse(trial.ap_grf[aff]),
            "cop_ap_range_aff": signal_range(trial.cop_ap[aff]),
            "cop_ml_range_aff": signal_range(trial.cop_ml[aff]),
            "cop_path_length_aff": cop_path_length(trial.cop_ap[aff], trial.cop_ml[aff]),
            "cop_ap_range_asym": asymmetry(signal_range(trial.cop_ap[aff]), signal_range(trial.cop_ap[unaff])),
            "cop_ml_range_asym": asymmetry(signal_range(trial.cop_ml[aff]), signal_range(trial.cop_ml[unaff])),
        }

        rows.append(row)

    return pd.DataFrame(rows)
```

```python
# project/src/modeling.py

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, classification_report
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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

def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), NUMERIC_FEATURES),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), CATEGORICAL_FEATURES),
        ]
    )

def train_models(train_df):
    x_train = train_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_train = train_df["label"]

    preprocessor = build_preprocessor()

    models = {
        "dummy": Pipeline([
            ("preprocess", preprocessor),
            ("model", DummyClassifier(strategy="most_frequent")),
        ]),
        "logistic_regression": Pipeline([
            ("preprocess", preprocessor),
            ("model", LogisticRegression(
                max_iter=2000,
                multi_class="multinomial",
                class_weight="balanced",
                random_state=42,
            )),
        ]),
        "random_forest": Pipeline([
            ("preprocess", preprocessor),
            ("model", RandomForestClassifier(
                n_estimators=500,
                max_depth=8,
                min_samples_leaf=5,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
            )),
        ]),
    }

    for model in models.values():
        model.fit(x_train, y_train)

    return models
```

## Presentation Outputs
- 핵심 표:
  - label별 subject/trial 수
  - label별 주요 feature 평균, 표준편차, confidence interval
  - 모델별 balanced accuracy, macro F1
  - baseline 대비 성능 향상
- 핵심 그림:
  - class별 평균 vertical GRF curve
  - class별 AP braking/propulsion 비교
  - class별 COP AP/ML range 비교
  - confusion matrix
  - permutation importance
  - 개별 subject rehab focus report 예시
  - SIAT gait phase별 EMG/ankle torque 참고 그림
- 최종 메시지:
  - “GRF/COP만으로도 impairment group과 관련된 보행 기능 이상 패턴을 정량화할 수 있다.”
  - “모델은 진단명이 아니라 재활 평가에서 우선 확인할 기능 후보를 제안한다.”
  - “EMG는 GaitRec에 없으므로 SIAT는 생리학적 해석 보조로만 사용한다.”

## Test Plan
- feature 계산 테스트:
  - synthetic signal에서 peak, impulse, asymmetry, COP path length가 예상값과 일치해야 한다.
- 데이터 split 테스트:
  - train/test에 동일 `subject_id`가 동시에 없어야 한다.
- 모델 테스트:
  - logistic regression과 random forest가 dummy baseline보다 높은 balanced accuracy 또는 macro F1을 보여야 한다.
- 리포트 테스트:
  - example subject report는 근거 feature 3개 이상, 재활 확인 후보 2개 이상을 포함해야 한다.
  - 금지 표현인 “진단”, “처방”, “원인 확정”, “특정 근육 약화 확정”이 포함되지 않아야 한다.
- 발표 산출물 테스트:
  - 모든 핵심 표와 그림 파일이 생성되어야 한다.
  - `final_analysis_report.md`가 문제정의, 데이터, 방법, 결과, 한계, future direction을 포함해야 한다.

## Assumptions
- Python, pandas, scikit-learn, matplotlib/seaborn 기반으로 구현한다.
- XGBoost/LightGBM은 선택하지 않는다. 설치 부담을 줄이고 재현성을 높이기 위해 scikit-learn 모델만 사용한다.
- SHAP은 필수 구현에서 제외한다. 대신 logistic coefficient와 permutation importance를 기본 설명 방법으로 사용한다.
- 최종 발표용 품질은 앱보다 분석 재현성, 결과 해석, 한계 방어력을 우선한다.

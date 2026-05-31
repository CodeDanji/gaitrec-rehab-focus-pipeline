# GaitRec Pipeline V2 (순수 보행 역학 중심 모델링)

이번 구현안은 메타데이터(Covariates)에 편향되었던 기존 모델의 한계를 극복하고, "순수하게 걷는 패턴의 모양(Waveform)"을 분석하는 모델로 파이프라인을 고도화(V2)하는 작업입니다.

## Proposed Changes

### 1. `features.py`

#### [MODIFY] features.py
* **수직 지면반발력(VGRF) 파형 추출 및 정규화(Dynamic Normalization)**
  * 기존에는 VGRF의 Peak 값 등 스칼라(Scalar) 수치만 추출했으나, 이제 전체 101개의 포인트(0~100%)로 구성된 VGRF Waveform 데이터를 추출하여 DataFrame 컬럼으로 추가합니다 (`vgrf_left_0` ~ `vgrf_left_100`, `vgrf_right_0` ~ `vgrf_right_100`).
  * 파형 데이터를 추출할 때 모든 VGRF(Newtons) 값을 피험자의 체중(`body_weight`)으로 나누어 **체중 대비 비율(%BW)**로 엄격하게 정규화합니다.
* **관련 함수 수정**:
  * `extract_gait_features` 내부에서 101개의 VGRF 값을 모두 순회하며 컬럼화하는 로직 추가.

### 2. `modeling.py`

#### [MODIFY] modeling.py
* **데이터 누수 및 힌트 메타데이터 제거**
  * `COVARIATE_CATEGORICAL` 리스트에서 `session_type`과 `orthopedic_insole`을 완전히 삭제합니다.
* **PCA (Principal Component Analysis) 파이프라인 적용**
  * Scikit-Learn의 `ColumnTransformer`를 확장하여, 새롭게 추가된 VGRF 파형 컬럼들(총 202개)에 대해서만 `PCA(n_components=0.95)`를 수행하도록 전처리 파이프라인(Preprocessor)을 수정합니다.
  * 기존의 스칼라(수치형) 데이터들과 Categorical 데이터들은 기존과 동일하게 스케일링/원핫인코딩을 거친 후, PCA가 완료된 파형 데이터와 결합(Feature Union)되어 Random Forest 등의 모델에 입력되도록 연결합니다.

## Verification Plan

### Automated Tests
1. `pytest` 또는 `python scripts/run_pipeline.py`를 실행.
2. 실행 시 `features.py`에서 파형 컬럼 202개가 정상 생성되는지, 메모리 초과 에러가 발생하지 않는지 확인.
3. `modeling.py` 파이프라인 통과 시 PCA 컴포넌트 차원 축소가 정상적으로 이루어지는지 로그 확인.

### Manual Verification
1. `model_metrics.csv`를 열어 `gait+covariate` 및 `gait-only` 모델의 Macro F1 Score가 이전 결과(누수가 있던 0.748) 대비 얼마나 변화했는지(순수 역학으로 얼마나 맞추는지) 확인.
2. Feature Importance 결과(`permutation_importance.csv`)에서 `pca_0`, `pca_1` 등의 주성분 피처들이 상위권에 랭크되어, 보행 파형이 실제 예측에 기여하고 있는지 검증.

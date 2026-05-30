# GaitRec 데이터 전처리 및 모델링 결정 문서

작성일: 2026-05-31

## 1. 목적

이 문서는 GaitRec GRF/COP 분석 파이프라인의 다음 구현 문서를 만들기 위한 결정 문서다.

핵심 목표는 단순히 라벨 분류 성능을 높이는 것이 아니라, 다음 세 가지를 분리해서 확인하는 것이다.

- 보행역학 신호 자체가 설명하는 부분
- 성별, 나이, 속도, 신발 조건 같은 covariate가 설명하는 부분
- 두 종류의 정보를 합쳤을 때의 최대 분류 성능

최종 출력은 의료적 확정 진단이 아니라, 재활 평가에서 우선 확인할 기능 후보를 제안하는 screening support로 제한한다.

## 2. 현재 Git 기준 근거

지금까지 Git과 결과물에서 확인한 주요 실험은 다음과 같다.

| 실험 | 주요 입력 | 최고 모델 | macro F1 | 해석 |
| --- | --- | --- | ---: | --- |
| vertical-only subset | metadata + vertical GRF | random forest | 0.465 | 500MB 이하 검증용. AP/COP feature는 unavailable. |
| full GRF/COP + covariate | GRF/COP + age/sex/speed/shoe/height/weight | random forest | 0.478 | 성능은 가장 높지만 covariate 의존성이 큼. |
| no-phys experiment | speed/age/height/weight/sex 제거, shoe_condition 유지 | gradient boosting | 0.369 | covariate를 빼면 성능이 크게 낮아짐. 단, 완전한 gait-only 실험은 아님. |

기존 full run의 permutation importance 상위는 `walking_speed`, `height`, `sex`, `push_off_index`, `weight`, `age`, `shoe_condition` 순서였다. 따라서 기존 최고 성능은 순수 GRF/COP 기반 성능으로 해석하면 안 된다.

## 3. 데이터셋 사실

현재 full feature table 기준:

| 구분 | subject 수 | session 수 | trial row 수 |
| --- | ---: | ---: | ---: |
| 전체 | 2,295 | 8,971 | 75,732 |
| 환자군 전체 | 2,084 | 8,161 | 67,977 |
| Healthy | 211 | 810 | 7,755 |

라벨별 분포:

| label | subject 수 | session 수 | trial row 수 |
| --- | ---: | ---: | ---: |
| Ankle | 627 | 2,587 | 21,386 |
| Knee | 625 | 2,386 | 19,873 |
| Hip | 450 | 1,512 | 12,748 |
| Calcaneus | 382 | 1,676 | 13,970 |
| Healthy | 211 | 810 | 7,755 |

중요한 covariate 불균형:

- `sex`: 전체 trial 기준 male 75.28%, female 24.72%.
- `SPEED`: slow/fast는 사실상 Healthy에만 있고, 환자군은 self-selected 중심이다.
- `SHOD_CONDITION`: Healthy는 barefoot 비율이 높고 orthopedic shoe가 없으며, 환자군에는 orthopedic shoe가 존재한다.
- `age`: Healthy가 환자군보다 젊다. Healthy median은 32세이고, 환자군은 대체로 43-45세다.

이 불균형 때문에 covariate-only 모델도 라벨을 일정 수준 맞출 수 있다. 따라서 성능 수치만으로 보행역학 feature의 유용성을 주장하면 안 된다.

## 4. 결정 1: 병변측 매핑을 먼저 수정한다

원본 metadata에는 `AFFECTED_SIDE`가 존재한다. 현재 로컬 metadata에서 확인한 값 분포는 다음과 같다.

| AFFECTED_SIDE | 의미 | session 수 |
| ---: | --- | ---: |
| 0.0 | left | 3,933 |
| 1.0 | right | 3,811 |
| 2.0 | both | 417 |
| NaN | Healthy 또는 unknown | 810 |

현재 코드의 문제:

- `_normalize_side()`가 `"1"`, `"2"` 같은 문자열만 처리하고 `1.0`, `2.0`, `0.0` float 값을 처리하지 못한다.
- 공식 의미는 `0=left`, `1=right`, `2=both`인데 현재 코드는 `"1"=left`, `"2"=right`처럼 해석한다.
- 알 수 없는 값은 전부 `left`로 fallback된다.
- 그 결과 현재 `results/gaitrec_full/tables/gaitrec_features.csv`의 `affected_side`는 75,732 rows 전부 `left`다.

결정:

- `AFFECTED_SIDE` 원본값과 정규화값을 모두 보존한다.
- 정규화값은 `left`, `right`, `both`, `unknown` 중 하나로 둔다.
- `both`와 `unknown`은 임의로 `left` 또는 `right`로 바꾸지 않는다.
- affected/unaffected feature는 `left` 또는 `right`가 확정된 row에서만 사용한다.
- Healthy와 both-side 케이스까지 포함하는 main model에는 side-neutral gait feature를 사용한다.

다음 구현 문서에서는 feature family를 두 개로 나눈다.

- `side_neutral_gait_features`: left/right 평균, 최대, 차이의 절댓값, symmetry magnitude 등 병변측이 필요 없는 feature.
- `affected_side_gait_features`: `left/right`가 확정된 환자 row에서만 계산하는 affected/unaffected feature.

현재 affected-side 수정 전 결과는 historical baseline으로만 취급한다.

## 5. 결정 2: Cohort 정의

Cohort는 모델의 목적에 따라 명확히 나눈다.

### 5.1 All Cohort

전체 데이터를 사용한다.

용도:

- historical 비교.
- 전체 데이터에 대한 sensitivity analysis.
- 데이터 손실 없이 최대 범위 결과를 확인하는 reference.

Main 해석에는 사용하지 않는다. speed, shoe condition, age, sex confounding이 크기 때문이다.

### 5.2 Primary Clean Cohort

Main 분석의 기본 cohort다.

조건:

- `SPEED == 2`
- `SHOD_CONDITION == 1`
- `18 <= AGE <= 65`

이 조건은 Stage 1과 Stage 2 모두에 적용한다.

예상 규모:

| 대상 | subject 수 | session 수 | trial row 수 |
| --- | ---: | ---: | ---: |
| primary clean 전체 | 2,116 | 7,412 | 62,092 |
| primary clean 환자군 | 1,930 | 7,226 | 60,266 |
| primary clean Healthy | 186 | 186 | 1,826 |

결정:

- Stage 1 main은 primary clean 전체에서 `Healthy` vs `Impaired`를 학습한다.
- Stage 2 main은 primary clean 환자군에서 `Hip/Knee/Ankle/Calcaneus`를 학습한다.

### 5.3 Sex-Balanced Cohort

Sex-balanced cohort는 main cohort가 아니라 confounding 검증용 보조 cohort다.

두 방식 중 구현 문서에서 하나를 지정하되, 기본은 sample weighting으로 둔다.

- `sex matching`: 각 라벨의 sex 비율을 맞추기 위해 일부 subject를 downsample한다.
- `sex sample weighting`: 데이터를 버리지 않고 label-sex 조합별 weight를 조정한다.

기본을 sample weighting으로 두는 이유:

- primary clean 환자군에서 Calcaneus female subject가 42명으로 작다.
- 엄격한 matching은 Ankle, Knee, Hip의 많은 데이터를 버리게 만든다.
- Stage 2의 4-class 모델은 작은 cell에 맞춰 downsample하면 variance가 커질 수 있다.

### 5.4 Male-Only / Female-Only Cohort

Male-only와 female-only는 main 분석이 아니라 sensitivity analysis로만 사용한다.

이유:

- male-only는 데이터가 많지만 결론이 남성 환자군으로 제한된다.
- female-only는 일부 라벨, 특히 Calcaneus/Hip의 subject 수가 작아 main model로 불안정하다.
- 이 분석은 "성별을 제한해도 주요 gait pattern이 유지되는가"를 확인하는 용도다.

## 6. 결정 3: Sensitivity Analysis의 의미

Sensitivity analysis는 전처리 선택을 바꿔도 결론이 유지되는지 확인하는 강건성 검증이다.

예:

- age range를 `18-65`, `20-60`, `25-60`으로 바꿔 비교한다.
- all cohort와 primary clean cohort 결과를 비교한다.
- sex weighting 적용 전후를 비교한다.
- male-only/female-only에서 주요 gait feature importance가 유지되는지 확인한다.
- orthopedic shoe 포함/제외 결과를 비교한다.

Sensitivity analysis는 최종 main model을 대체하지 않는다. main claim이 특정 필터 하나에만 의존하는지 확인하는 장치로 사용한다.

## 7. 결정 4: Feature Set은 세 가지로 고정한다

모든 주요 modeling task는 아래 세 feature set을 함께 보고한다.

### 7.1 Gait-Only

GRF/COP 기반 feature만 사용한다.

포함:

- side-neutral GRF/COP feature.
- left/right가 확정된 환자 분석에서는 affected-side GRF/COP feature.

제외:

- `age`
- `sex`
- `height`
- `body_weight`
- `body_mass`
- `shoe_size`
- `SPEED`
- `SHOD_CONDITION`
- `ORTHOPEDIC_INSOLE`

용도:

- 재활 기능 해석의 중심 모델.
- 보행역학 신호 자체가 라벨을 얼마나 설명하는지 확인.

### 7.2 Covariate-Only

Metadata/covariate만 사용하고 GRF/COP feature는 제외한다.

포함 후보:

- `age`
- `sex`
- `height`
- `body_mass`
- `body_weight`
- `shoe_size`
- `SPEED`
- `SHOD_CONDITION`
- `ORTHOPEDIC_INSOLE`
- `SESSION_TYPE`

용도:

- confounding 정도를 측정하는 control model.
- covariate-only 성능이 높으면 gait+covariate 성능도 조심해서 해석한다.

주의:

- `body_weight`의 현재 값 범위는 kg가 아니라 force 또는 다른 단위일 가능성이 있다.
- 구현 문서에서는 `BODY_MASS`와 `BODY_WEIGHT`를 구분해서 보존한다.

### 7.3 Gait + Covariate

Gait-only와 covariate-only feature를 모두 사용한다.

용도:

- 사용 가능한 모든 정보를 넣었을 때의 최대 분류 성능 확인.
- gait feature가 covariate-only 대비 추가 설명력을 제공하는지 확인.

해석 제한:

- 이 모델의 높은 성능을 곧바로 보행역학 기반 성능으로 해석하지 않는다.
- feature importance는 gait feature와 covariate feature를 분리해서 보고한다.

## 8. 결정 5: Modeling Task

Modeling task는 세 가지로 둔다.

### 8.1 Stage 1: Healthy vs Impaired

목표:

- Healthy와 환자군 전체를 이진 분류한다.

Main:

- primary clean cohort.
- side-neutral gait-only, covariate-only, gait+covariate 비교.

Sensitivity:

- all cohort.
- sex sample weighting.
- male-only.
- female-only.
- age range 변경.

주의:

- Stage 1은 Healthy와 환자군의 speed, shoe, age 차이가 가장 크게 작동하는 task다.
- 따라서 covariate-only baseline을 반드시 함께 보고한다.

### 8.2 Stage 2: Impaired 내부 4-Class 분류

목표:

- 환자군 내부에서 `Hip`, `Knee`, `Ankle`, `Calcaneus`를 분류한다.

Main:

- primary clean 환자군.
- side-neutral gait-only, covariate-only, gait+covariate 비교.
- affected-side feature는 `AFFECTED_SIDE in {left,right}`인 row에서 별도 추가 분석으로 사용한다.

Sensitivity:

- sex sample weighting.
- male-only.
- female-only.
- `both` 제외/포함 정책 비교.

주의:

- Stage 2에서는 Healthy가 빠지므로 speed confounding은 Stage 1보다 약하다.
- 그래도 sex, shoe, age, body size confounding은 남아 있으므로 covariate-only baseline이 필요하다.

### 8.3 Reference: 5-Class 분류

기존 task인 `Healthy`, `Hip`, `Knee`, `Ankle`, `Calcaneus` 5-class 분류는 historical/reference task로 유지한다.

용도:

- 이전 결과와 비교.
- 발표용 전체 confusion matrix 참고.

Main claim은 Stage 1과 Stage 2 결과를 중심으로 작성한다.

## 9. 결정 6: Split과 평가

모든 학습/평가는 subject-level split을 사용한다.

결정:

- 같은 `subject_id`가 train과 test에 동시에 들어가면 실패로 본다.
- 단일 holdout 결과만으로 결론을 내리지 않는다.
- 구현 문서에서는 `StratifiedGroupKFold` 기반 5-fold evaluation을 기본으로 설계한다.
- fold stratification이 라이브러리 버전 문제로 불가능하면 label-stratified subject sampling을 직접 구현한다.

필수 metric:

- balanced accuracy.
- macro F1.
- class별 precision/recall/F1.
- confusion matrix.
- permutation importance.

추가 metric:

- Stage 1에서는 sensitivity/specificity를 함께 보고한다.
- Stage 2에서는 label별 recall을 우선 확인한다.

## 10. 결정 7: 결과 해석과 보고

결과 문장은 anatomical label 확정보다 functional rehab focus를 중심으로 작성한다.

허용하는 해석 축:

- push-off 기능.
- loading asymmetry.
- vertical loading pattern.
- COP AP/ML range.
- COP path length.
- weight-bearing strategy.
- gait stability cue.

금지하는 해석:

- 특정 원인을 확정하는 표현.
- 특정 처치를 지시하는 표현.
- 특정 근육 약화를 확정하는 표현.
- GaitRec 결과를 SIAT EMG/torque 결과와 직접 연결해 같은 subject의 원인처럼 말하는 표현.

SIAT-LLMD는 계속 auxiliary reference로만 둔다. GaitRec classifier 입력에 SIAT feature를 병합하지 않는다.

## 11. 다음 구현 문서가 담아야 할 필수 요구사항

다음 구현 문서는 아래 작업을 decision-complete하게 다뤄야 한다.

1. `AFFECTED_SIDE` 정규화 수정.
2. 원본 metadata 보존 범위:
   - `BODY_MASS`
   - `BODY_WEIGHT`
   - `SHOE_SIZE`
   - `ORTHOPEDIC_INSOLE`
   - `SESSION_TYPE`
   - `SPEED`
   - `SHOD_CONDITION`
3. side-neutral feature family 추가.
4. affected-side feature family 재계산.
5. cohort builder 추가:
   - all
   - primary clean
   - sex-weighted
   - male-only
   - female-only
6. feature set builder 추가:
   - gait-only
   - covariate-only
   - gait+covariate
7. task runner 추가:
   - Stage 1 Healthy vs Impaired
   - Stage 2 impaired 4-class
   - reference 5-class
8. subject-level cross-validation 추가.
9. 결과 저장 구조 추가.

권장 output 구조:

```text
results/modeling_decision_v1/
  cohorts/
  stage1_healthy_vs_impaired/
  stage2_impaired_4class/
  reference_5class/
  reports/
```

## 12. 구현 검증 기준

구현 완료 후 최소한 아래 검증을 통과해야 한다.

- `AFFECTED_SIDE=0.0`은 `left`, `1.0`은 `right`, `2.0`은 `both`, `NaN`은 `unknown`으로 정규화된다.
- feature table의 `affected_side`가 전부 `left`가 아니다.
- primary clean cohort는 `SPEED=2`, `SHOD_CONDITION=1`, `18 <= AGE <= 65`만 포함한다.
- Stage 1 primary clean에는 Healthy와 환자군이 모두 포함된다.
- Stage 2 primary clean에는 Healthy가 포함되지 않는다.
- gait-only feature set에는 covariate가 들어가지 않는다.
- covariate-only feature set에는 GRF/COP feature가 들어가지 않는다.
- 같은 `subject_id`가 train/test 또는 fold 간 evaluation split에 누수되지 않는다.
- report guardrail은 기존처럼 유지된다.

## 13. 기본 결론

다음 구현의 우선순위는 다음 순서로 둔다.

1. 병변측 매핑 수정.
2. side-neutral 및 affected-side feature 재정의.
3. primary clean cohort 생성.
4. gait-only, covariate-only, gait+covariate 비교.
5. Stage 1과 Stage 2 modeling.
6. sensitivity analysis.
7. functional rehab focus 중심 report 생성.

이 순서를 지키지 않으면 모델 성능은 좋아 보여도, 그 성능이 보행역학에서 나온 것인지 데이터 수집 조건과 인구통계 차이에서 나온 것인지 구분하기 어렵다.

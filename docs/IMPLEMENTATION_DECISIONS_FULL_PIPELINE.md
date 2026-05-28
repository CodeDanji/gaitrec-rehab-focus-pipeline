# 전체 파이프라인 구현 결정 문서

작성일: 2026-05-26

## 1. 목적

이 문서는 현재 구현된 **GaitRec vertical-GRF 검증판** 이후에 남아 있는 기능을 어떻게 구현할지 결정하기 위한 기준 문서다. 다음 구현 문서와 실제 구현 작업은 이 문서를 기준으로 진행한다.

최종 목표는 실제 GaitRec/SIAT-LLMD 파일을 기반으로 재현 가능한 GRF/COP + 보조 EMG/torque 분석 파이프라인을 만드는 것이다. 단, 테스트와 동료 인수 검증은 데이터 용량 부담을 줄이기 위해 250MB 미만의 row-sampled 실제 데이터 subset으로 수행한다.

## 2. 현재 구현 상태

현재 repo는 전체 프로젝트 중 **GaitRec processed vertical GRF left/right + metadata**만 실제 데이터로 검증한 상태다.

구현 및 검증 완료:

- GaitRec metadata 로딩.
- processed vertical GRF left/right 로딩.
- GaitRec 실제 key 구조인 `SUBJECT_ID`, `SESSION_ID`, `TRIAL_ID` 지원.
- affected/unaffected side 매핑.
- vertical GRF peak, vertical GRF asymmetry, loading rate asymmetry 계산.
- subject-level train/test split.
- baseline 및 minimal runtime fallback 모델 학습/평가.
- label별 feature summary, confusion matrix, permutation importance 생성.
- final analysis report와 example subject rehab-focus report 생성.
- 금지 표현 guardrail 테스트.

현재 실제 검증 결과:

- feature table: 75,732 rows, 23 columns.
- unique subjects: 2,295.
- labels: `Healthy`, `Hip`, `Knee`, `Ankle`, `Calcaneus`.
- dummy baseline macro F1: 0.086.
- softmax fallback macro F1: 0.327.
- nearest centroid fallback macro F1: 0.274.

## 3. Gap Analysis

현재 구현은 최종 브리프인 `grf-cop-emg-rehab-focus-project.md`와 `PLAN.md` 대비 부분 구현이다.

아직 실제 파일 기반으로 구현/검증되지 않은 항목:

- GaitRec AP GRF left/right 기반 braking impulse, propulsion impulse, push-off index.
- GaitRec ML GRF left/right 기반 보조 안정성 지표.
- GaitRec COP AP/ML left/right 기반 COP range, COP path length, COP asymmetry.
- GaitRec processed GRF/COP 전체 파일 조합에 대한 end-to-end 실행.
- scikit-learn `LogisticRegression` / `RandomForestClassifier` 기본 경로 검증.
- class별 평균 vertical/AP/COP curve와 발표용 비교 figure.
- SIAT-LLMD EMG/torque/gait phase parser.
- SIAT-LLMD 기반 gait phase별 EMG/torque 참고 분석.

현재 있는 SIAT 코드는 placeholder다. `siat_root`를 받아도 실제 SIAT 파일을 파싱하지 않고, note와 placeholder figure만 생성한다.

## 4. 핵심 구현 결정

### 4.1 데이터 취득 방식

데이터 취득은 **Figshare manifest downloader 방식**으로 구현한다.

결정 사항:

- GaitRec 파일별 article id, file id, 파일명, 크기, 역할을 manifest에 기록한다.
- downloader는 manifest를 읽어 필요한 subset 또는 full set을 내려받는다.
- 데이터 파일은 `data/` 아래에 저장하며 git에는 포함하지 않는다.
- 다운로드된 파일의 size check를 수행해 잘못 받은 파일을 조기에 탐지한다.

GaitRec full processed 대상:

- `GRF_metadata.csv`
- `GRF_F_V_PRO_left.csv`
- `GRF_F_V_PRO_right.csv`
- `GRF_F_AP_PRO_left.csv`
- `GRF_F_AP_PRO_right.csv`
- `GRF_F_ML_PRO_left.csv`
- `GRF_F_ML_PRO_right.csv`
- `GRF_COP_AP_PRO_left.csv`
- `GRF_COP_AP_PRO_right.csv`
- `GRF_COP_ML_PRO_left.csv`
- `GRF_COP_ML_PRO_right.csv`

SIAT-LLMD 대상:

- 전체 파일을 repo에 포함하지 않는다.
- manifest에 공식 출처, 파일 역할, 용량, 필수 여부를 기록한다.
- 실제 parser 구현 전에는 SIAT 파일 구조를 먼저 inspect하는 별도 단계가 필요하다.

### 4.2 Smoke 검증 데이터

테스트와 동료 인수 검증용 데이터는 **250MB 미만 row-sampled 실제 데이터 subset**으로 만든다.

결정 사항:

- 공식 CSV 파일을 그대로 모두 쓰지 않고, subject/session/trial key를 유지한 일부 row만 추출한다.
- left/right 및 AP/COP 파일 사이의 key 정합성을 유지한다.
- label별 최소 subject/trial 수를 포함해 모델과 리포트가 실제로 생성되는지 검증한다.
- row-sampled subset은 재현 가능한 seed와 sampling manifest를 기록한다.
- smoke subset도 git에는 포함하지 않는다.

250MB 미만 검증 subset에 포함할 최소 신호:

- metadata.
- vertical GRF left/right.
- AP GRF left/right.
- COP AP left/right.
- COP ML left/right.

ML GRF는 용량과 우선순위에 따라 smoke subset v2에서 포함할 수 있다.

### 4.3 GaitRec feature schema

최종 GaitRec feature table의 분석 단위는 `subject_id + session_id + trial_id + affected_side`다.

필수 identity/covariate columns:

- `subject_id`
- `session_id`
- `trial_id`
- `label`
- `affected_side`
- `walking_speed`
- `age`
- `sex`
- `height`
- `weight`
- `shoe_condition`

필수 GRF/COP feature:

- `vgrf_peak_aff`
- `vgrf_peak_unaff`
- `vgrf_peak_asym`
- `loading_rate_asym`
- `ap_braking_impulse_asym`
- `ap_propulsion_impulse_asym`
- `push_off_index`
- `cop_ap_range_aff`
- `cop_ml_range_aff`
- `cop_path_length_aff`
- `cop_ap_range_asym`
- `cop_ml_range_asym`

파일이 없어서 계산할 수 없는 feature는 0으로 채우지 않고 `NaN`으로 남긴다. final report에는 현재 run에서 available/unavailable feature를 명시한다.

### 4.4 모델링 정책

기본 모델 경로는 scikit-learn 기반으로 고정한다.

결정 사항:

- baseline: `DummyClassifier(strategy="most_frequent")`.
- primary model: `LogisticRegression(multi_class="multinomial", class_weight="balanced")`.
- secondary model: `RandomForestClassifier(class_weight="balanced_subsample")`.
- fallback 모델은 scikit-learn이 없는 최소 환경에서 smoke 실행을 가능하게 하는 보조 경로로 유지한다.
- 발표/최종 결과는 scikit-learn 모델 결과를 우선한다.
- split은 반드시 subject-level split을 사용한다.

평가 지표:

- balanced accuracy.
- macro F1.
- class별 precision/recall/F1.
- confusion matrix.
- baseline 대비 향상.

### 4.5 Report/Figure 정책

결과물은 높은 정확도 과시보다 설명 가능성과 재활적 해석을 우선한다.

필수 표:

- label별 subject/trial count.
- label별 주요 feature mean/std/95% CI.
- 모델별 balanced accuracy/macro F1.
- permutation importance.
- logistic coefficient table.

필수 그림:

- workflow diagram.
- model metrics.
- confusion matrix.
- permutation importance.
- class별 평균 vertical GRF curve.
- class별 AP braking/propulsion 비교.
- class별 COP AP/ML range 또는 COP path 비교.
- example subject rehab-focus report figure 또는 markdown report.

필수 문서:

- `final_analysis_report.md`
- `example_subject_rehab_focus.md`
- `siat_reference_note.md`

### 4.6 SIAT-LLMD 분리 원칙

SIAT-LLMD는 GaitRec 모델 입력으로 병합하지 않는다.

결정 사항:

- SIAT는 auxiliary reference pipeline으로 분리한다.
- output은 EMG/torque/gait phase 참고 그림과 해석 note다.
- SIAT 결과는 GaitRec subject의 원인을 확정하는 데 사용하지 않는다.
- 표현은 “생리학적 배경 참고”, “해석 보조”, “추가 평가 후보” 수준으로 제한한다.

SIAT 구현 목표:

- walking sample 파일 구조 파악.
- gait phase column, EMG channel, joint torque column 식별.
- gait phase별 EMG 평균 curve 생성.
- ankle/knee/hip torque와 관련 EMG timing 비교 figure 생성.
- GaitRec push-off/AP/COP 결과 해석에 참고로 연결하는 note 생성.

## 5. Phase별 구현 로드맵

### Phase 1. Manifest Downloader와 Smoke Subset Builder

목표:

- GaitRec/SIAT 공식 파일 manifest를 만든다.
- 필요한 파일 세트를 명령어로 다운로드할 수 있게 한다.
- 250MB 미만 row-sampled smoke subset을 생성한다.

완료 조건:

- `python scripts/download_data.py --dataset gaitrec --set smoke-source` 형태의 실행 경로가 있다.
- `python scripts/build_smoke_subset.py ...`로 key 정합성이 유지된 subset이 생성된다.
- smoke subset size가 250MB 미만임을 확인하는 검증 출력이 있다.

### Phase 2. GaitRec Full GRF/COP Feature Pipeline

목표:

- AP GRF, ML GRF, COP AP, COP ML 파일을 실제 GaitRec 구조로 로딩한다.
- 모든 PLAN.md feature를 실제 파일 기반으로 계산한다.

완료 조건:

- smoke subset에서 모든 필수 GRF/COP feature가 non-null로 생성된다.
- full processed 데이터에서도 feature extraction이 완료된다.
- 누락 파일이 있을 때는 명확한 unavailable report가 생성된다.

### Phase 3. scikit-learn Model Path 검증

목표:

- scikit-learn 기반 Logistic Regression과 Random Forest를 기본 경로로 검증한다.

완료 조건:

- `requirements.txt` 설치 환경에서 dummy, logistic regression, random forest가 모두 실행된다.
- subject leakage가 없는 split test가 통과한다.
- metrics, classification report, confusion matrix, permutation importance가 생성된다.

### Phase 4. 발표용 Figure/Report 강화

목표:

- 최종 발표에 바로 넣을 수 있는 표/그림/해석 문서를 만든다.

완료 조건:

- class별 평균 vertical GRF curve figure가 생성된다.
- AP braking/propulsion 비교 figure가 생성된다.
- COP range/path 비교 figure가 생성된다.
- final report가 문제정의, 데이터, 방법, 결과, 한계, future direction을 포함한다.
- report가 금지 표현을 포함하지 않는다.

### Phase 5. SIAT-LLMD Auxiliary Reference Pipeline

목표:

- SIAT-LLMD를 별도 reference analysis로 구현한다.

완료 조건:

- SIAT sample 파일 parser가 있다.
- gait phase별 EMG/torque 참고 figure가 생성된다.
- GaitRec 모델 입력에 SIAT feature가 들어가지 않는다.
- final report에서 SIAT는 “보조 생리학 참고”로만 표현된다.

## 6. Acceptance Criteria

동료가 repo를 받은 뒤 다음을 수행할 수 있어야 한다.

Smoke 검증:

- manifest 기반으로 필요한 source file을 받을 수 있다.
- 250MB 미만 row-sampled smoke subset을 만들 수 있다.
- smoke subset으로 전체 GaitRec GRF/COP feature extraction, modeling, report generation을 실행할 수 있다.
- 결과물은 `results/` 아래에 생성되며 git에 포함되지 않는다.

Full 데이터 실행:

- 동료가 GaitRec processed full set을 받은 뒤 같은 CLI로 full run을 수행할 수 있다.
- full run은 같은 feature schema와 output schema를 사용한다.
- 메모리 부담이 큰 경우 chunking 또는 intermediate feature table 저장 전략을 사용한다.

SIAT 실행:

- SIAT는 별도 CLI 또는 별도 pipeline step으로 실행된다.
- SIAT output은 GaitRec classifier에 병합되지 않는다.
- SIAT report는 EMG/torque timing 참고 분석임을 명시한다.

결과 해석:

- 모델은 진단명이 아니라 재활 평가 우선순위 후보를 제안한다.
- “진단”, “처방”, “원인 확정”, “특정 근육 약화 확정” 표현은 최종 report에 포함하지 않는다.
- AP/COP/SIAT 파일이 없는 run에서는 해당 feature가 unavailable로 명확히 표시된다.

## 7. Risks And Guardrails

### 데이터 용량과 메모리

GaitRec full processed CSV는 크고 pandas 메모리 사용량이 파일 크기보다 커질 수 있다. smoke subset으로 먼저 end-to-end 검증하고, full run에서는 chunk loading 또는 intermediate feature table 저장을 사용한다.

### Metadata confounding

현재 vertical-only 결과에서 metadata feature의 importance가 높았다. 최종 구현에서는 gait-only 모델과 metadata-inclusive 모델을 분리해 비교하는 옵션을 고려한다.

### SIAT 해석 오해

SIAT-LLMD는 건강인 데이터이며 GaitRec subject와 연결된 데이터가 아니다. 따라서 SIAT 결과는 정상/비정상 판정이나 원인 확정에 사용하지 않는다.

### 결론 표현

허용 표현:

- “이 subject는 특정 impairment group에서 자주 관찰되는 GRF/COP 패턴과 유사하다.”
- “재활 평가에서 체중부하 회피, push-off 기능, COP 안정성을 우선 확인할 후보로 제안한다.”

금지 표현:

- “이 subject를 진단한다.”
- “이 데이터만으로 처방할 수 있다.”
- “특정 근육 약화가 원인으로 확정된다.”

## 8. 다음 구현 문서에서 결정할 세부사항

다음 구현 문서는 이 결정 문서를 바탕으로 작성한다.

반드시 포함할 세부사항:

- manifest 파일 schema.
- downloader CLI 인자.
- smoke subset sampling 기준.
- full feature table schema.
- GaitRec AP/COP loader 검증 방식.
- SIAT parser에 필요한 파일 구조 조사 절차.
- tests/acceptance checks 목록.


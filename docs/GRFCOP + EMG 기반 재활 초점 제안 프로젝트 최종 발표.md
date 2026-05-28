# GRF/COP + EMG 기반 재활 초점 제안 프로젝트 최종 발표급 구현 계획

## Summary
- 목표는 MVP 데모가 아니라, **최종 발표에서 분석 결과·그림·한계·재활적 해석을 모두 제시할 수 있는 재현 가능한 분석 패키지**를 만드는 것이다.
- 본 분석은 **GaitRec processed GRF/COP 전체**를 사용하고, SIAT-LLMD는 **일부 샘플 기반 EMG/torque 참고 분석**으로만 사용한다.
- 최종 산출물은 `코드 + 결과 리포트 + 발표용 그림`이다. 진단/처방 모델이 아니라, impairment group과 관련된 보행 기능 이상을 설명하고 재활 평가에서 우선 확인할 후보를 제안한다.

## Key Interfaces And Outputs
- 입력 데이터:
  - GaitRec metadata
  - GaitRec processed left/right vertical GRF, AP GRF, ML GRF
  - GaitRec processed left/right COP AP, COP ML
  - SIAT-LLMD 일부 walking 샘플: sEMG, joint angle, joint torque, gait phase
- 핵심 feature table 출력:
  - 단위: `subject_id + trial_id + affected_side`
  - 포함 feature: loading asymmetry, vertical GRF peak asymmetry, AP braking impulse, AP propulsion impulse, push-off index, COP AP range, COP ML range, COP path length, stance symmetry, walking speed, age, sex, height, weight, shoe condition, impairment label
- 최종 결과물:
  - `results/tables/gaitrec_features.csv`
  - `results/tables/model_metrics.csv`
  - `results/tables/group_feature_summary.csv`
  - `results/figures/` 내 발표용 PNG/SVG
  - `results/reports/final_analysis_report.md` 또는 `.html`
  - `results/reports/example_subject_rehab_focus.md`

## Implementation Changes
- 프로젝트 구조:
  - `project/src/`에 Python 분석 모듈을 둔다.
  - `project/scripts/`에 실행용 CLI 스크립트를 둔다.
  - `project/results/`에 생성 결과만 저장한다.
  - `project/docs/`에는 설계 문서와 발표용 해석 문서를 둔다.
- 파이프라인:
  - `download_or_register_data`: 원본 데이터를 자동 다운로드하거나, 이미 받은 데이터 경로를 config에 등록한다.
  - `extract_gaitrec_features`: GaitRec processed GRF/COP에서 feature table을 만든다.
  - `analyze_group_patterns`: Healthy/Hip/Knee/Ankle/Calcaneus별 feature 차이, effect size, confidence interval, 평균 curve를 생성한다.
  - `train_explainable_model`: subject-level split으로 Logistic Regression과 Random Forest를 학습하고, confusion matrix, feature importance, permutation importance를 산출한다.
  - `generate_rehab_focus_report`: 모델 결과와 feature threshold를 바탕으로 개별 subject/trial용 재활 평가 후보 리포트를 생성한다.
  - `analyze_siat_reference`: SIAT 일부 walking 샘플에서 gait phase별 EMG와 joint torque 관계 그림을 만든다.
- 모델 원칙:
  - trial-level random split 금지.
  - 같은 subject의 trial이 train/test에 동시에 들어가지 않도록 subject-level split 사용.
  - 모델 평가는 accuracy만 보지 않고 macro F1, balanced accuracy, confusion matrix를 함께 본다.
  - class imbalance는 class weight 또는 balanced metric으로 처리한다.
- 발표용 핵심 그림:
  - 전체 연구 흐름도: GRF/COP input → feature extraction → explainable model → rehab focus output
  - class별 평균 vertical GRF curve
  - class별 AP propulsion/braking 비교
  - class별 COP path/range 비교
  - feature importance 또는 permutation importance
  - confusion matrix
  - example subject report figure
  - SIAT EMG/torque 참고 그림 1-2개
- 재활 해석 출력 규칙:
  - 허용 표현: “이 subject는 Ankle/Calcaneus group에서 자주 관찰되는 propulsion 감소 및 COP 이동 제한 패턴과 유사하다.”
  - 허용 표현: “재활 평가에서 발목 push-off 기능, 체중부하 회피, 통증 회피 전략을 우선 확인할 후보로 제안한다.”
  - 금지 표현: “비복근 약화가 원인이다”, “특정 질환을 진단한다”, “이 데이터만으로 처방할 수 있다.”
- SIAT 사용 원칙:
  - GaitRec과 직접 병합하거나 하나의 진단 모델에 넣지 않는다.
  - 건강인 참고 데이터로만 사용한다.
  - EMG는 “원인 확정”이 아니라 “push-off, torque, muscle activation timing의 생리학적 배경 설명”에 사용한다.

## Test Plan
- 데이터 무결성 테스트:
  - metadata와 processed signal 파일의 subject/trial key가 매칭되는지 확인한다.
  - affected/unaffected side 매핑이 누락되거나 뒤집힌 trial을 탐지한다.
  - GRF/COP signal 길이, NaN 비율, 비정상 범위를 검증한다.
- feature extraction 테스트:
  - synthetic GRF/COP 배열로 peak, impulse, symmetry, COP range, path length 계산값을 검증한다.
  - affected side와 unaffected side를 바꿨을 때 asymmetry sign이 예상대로 바뀌는지 검증한다.
- split 테스트:
  - train/test에 동일 subject_id가 동시에 존재하지 않는지 검증한다.
- 모델 테스트:
  - baseline dummy classifier보다 macro F1 또는 balanced accuracy가 높은지 확인한다.
  - confusion matrix와 feature importance 파일이 생성되는지 확인한다.
- 리포트 테스트:
  - example subject report가 최소 3개 근거 feature와 2개 이상 재활 확인 후보를 포함하는지 확인한다.
  - 금지 표현 목록이 report에 포함되지 않는지 검사한다.

## Assumptions And Defaults
- 구현 언어는 Python이다.
- 최종 발표용 구현은 앱/대시보드가 아니라 **재현 가능한 분석 파이프라인 + 리포트 + 그림**으로 만든다.
- GaitRec은 processed GRF/COP 전체를 사용한다.
- SIAT-LLMD는 전체 분석이 아니라 일부 walking 샘플 기반 참고 분석만 수행한다.
- 발표의 핵심 주장은 “AI가 진단명을 대체한다”가 아니라 “GRF/COP 기반 설명 가능한 분석으로 재활 평가 우선순위 후보를 제안한다”이다.
- 참고 자료는 GaitRec 논문과 SIAT-LLMD 논문을 기본 citation으로 사용한다:
  - [GaitRec, Scientific Data 2020](https://www.nature.com/articles/s41597-020-0481-z)
  - [SIAT-LLMD, Scientific Data 2023](https://www.nature.com/articles/s41597-023-02263-3)

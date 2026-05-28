# 전체 파이프라인 구현 설계서

작성일: 2026-05-26

기준 문서: `docs/IMPLEMENTATION_DECISIONS_FULL_PIPELINE.md`

## 1. 목적

이 문서는 현재 구현된 GaitRec vertical-GRF 검증판을 전체 GRF/COP + 보조 SIAT-LLMD reference pipeline으로 확장하기 위한 실행 설계서다. 구현자는 이 문서를 기준으로 manifest, downloader, smoke subset, feature extraction, modeling, figure/report, SIAT auxiliary analysis를 순서대로 구현한다.

최종 목표는 다음 세 가지다.

- 실제 GaitRec processed CSV 파일 기반의 재현 가능한 GRF/COP 분석 파이프라인.
- 250MB 미만 row-sampled smoke subset으로 동료가 빠르게 인수 검증할 수 있는 실행 경로.
- SIAT-LLMD를 GaitRec classifier와 섞지 않고 EMG/torque 해석 보조 자료로만 쓰는 분리된 reference pipeline.

결과 해석은 진단이나 처방이 아니라 재활 평가에서 우선 확인할 기능 후보 제안으로 제한한다.

## 2. 현재 repo 기준점

현재 구현되어 있는 핵심 파일은 다음과 같다.

| 영역 | 현재 파일 | 현재 책임 |
| --- | --- | --- |
| CLI | `scripts/run_pipeline.py` | demo/full pipeline 실행 |
| pipeline | `src/gait_rehab/pipeline.py` | data load, feature extraction, model, report orchestration |
| data | `src/gait_rehab/data.py` | GaitRec metadata/signal 파일 탐색과 로딩 |
| feature | `src/gait_rehab/features.py` | vertical/AP/COP feature 계산 골격 |
| model | `src/gait_rehab/modeling.py` | subject split, sklearn model, fallback model |
| plotting | `src/gait_rehab/plotting.py` | workflow, metrics, confusion, importance, group summary figure |
| reporting | `src/gait_rehab/reporting.py` | final report, example subject report, forbidden term guardrail |
| SIAT | `src/gait_rehab/siat.py` | 현재는 placeholder note/figure 생성 |
| tests | `tests/*.py` | feature formula, data loader, split/report guardrail |

현재 검증된 실제 데이터 범위는 `GRF_metadata.csv`, `GRF_F_V_PRO_left.csv`, `GRF_F_V_PRO_right.csv`다. AP GRF, ML GRF, COP AP, COP ML, SIAT parser는 아직 실제 파일 기반 검증이 끝나지 않았다.

## 3. 추가/수정할 파일 구조

### 3.1 새 파일

| 파일 | 책임 |
| --- | --- |
| `config/gaitrec_processed_manifest.json` | GaitRec processed source 파일의 공식 Figshare article/file metadata |
| `config/siat_manifest.json` | SIAT-LLMD source archive metadata |
| `scripts/download_data.py` | manifest 기반 다운로드와 size check |
| `scripts/build_smoke_subset.py` | key 정합성을 유지하는 250MB 미만 smoke subset 생성 |
| `scripts/inspect_siat.py` | SIAT 파일 구조 조사 결과를 table/report로 저장 |
| `tests/test_manifest_downloader.py` | manifest schema, file selection, size mismatch 테스트 |
| `tests/test_smoke_subset.py` | smoke subset key 정합성, size limit, label coverage 테스트 |
| `tests/test_sklearn_models.py` | sklearn 기본 모델 경로와 subject leakage guardrail 테스트 |
| `tests/test_siat_reference.py` | SIAT가 GaitRec classifier 입력에 병합되지 않는지 테스트 |

### 3.2 수정할 파일

| 파일 | 수정 내용 |
| --- | --- |
| `src/gait_rehab/data.py` | manifest role과 signal key mapping을 명시적으로 검증하고 unavailable file report를 반환 |
| `src/gait_rehab/features.py` | AP/COP feature가 실제 processed CSV에서 non-null로 생성되는지 보강 |
| `src/gait_rehab/modeling.py` | sklearn 모델을 기본 경로로 유지하고 fallback은 sklearn 미설치 시 보조 경로로만 사용 |
| `src/gait_rehab/plotting.py` | class별 평균 vertical GRF curve, AP impulse 비교, COP range/path 비교 figure 추가 |
| `src/gait_rehab/reporting.py` | available/unavailable feature, model comparison, SIAT limitation을 final report에 명시 |
| `src/gait_rehab/siat.py` | placeholder에서 inspect 기반 auxiliary reference analysis로 확장 |
| `scripts/run_pipeline.py` | unavailable feature report path와 smoke/full run 모드를 명확히 노출 |
| `README.md` | downloader, smoke subset, full run, SIAT reference 실행 절차 추가 |

## 4. Manifest 설계

### 4.1 Manifest 공통 schema

Manifest는 JSON으로 저장한다. downloader는 이 schema만 읽고 동작해야 한다.

```json
{
  "dataset": "gaitrec",
  "version": "processed-v1",
  "source_collection_id": 4788012,
  "source_collection_url": "https://api.figshare.com/v2/collections/4788012/articles?page_size=50",
  "files": [
    {
      "role": "metadata",
      "required_for": ["smoke", "full"],
      "article_id": 11394657,
      "file_id": 22062960,
      "filename": "GRF_metadata.csv",
      "size_bytes": 629530,
      "download_url": "https://ndownloader.figshare.com/files/22062960",
      "target_path": "GRF_metadata.csv",
      "sha256": null
    }
  ]
}
```

필드 규칙:

- `role`: pipeline 내부 역할. 예: `metadata`, `vgrf_left`, `ap_grf_right`, `cop_ml_left`.
- `required_for`: `smoke-source`, `smoke-v2`, `full`, `siat-inspect` 중 하나 이상.
- `article_id`, `file_id`, `download_url`: Figshare public API 기준 값.
- `size_bytes`: 다운로드 후 반드시 비교한다.
- `sha256`: 현재는 `null` 허용. 추후 hash를 확보하면 size check 이후 추가 검증으로 사용한다.
- `target_path`: `--output-root` 아래 저장할 상대 경로.

### 4.2 GaitRec processed manifest 항목

`config/gaitrec_processed_manifest.json`에는 아래 파일을 모두 기록한다.

| role | filename | article_id | file_id | size_bytes | required_for |
| --- | --- | ---: | ---: | ---: | --- |
| metadata | `GRF_metadata.csv` | 11394657 | 22062960 | 629530 | smoke-source, full |
| vgrf_left | `GRF_F_V_PRO_left.csv` | 11394819 | 22063191 | 136725253 | smoke-source, full |
| vgrf_right | `GRF_F_V_PRO_right.csv` | 11394804 | 22063119 | 136713924 | smoke-source, full |
| ap_grf_left | `GRF_F_AP_PRO_left.csv` | 11394816 | 22063185 | 147540140 | smoke-source, full |
| ap_grf_right | `GRF_F_AP_PRO_right.csv` | 11394792 | 22063101 | 147554970 | smoke-source, full |
| ml_grf_left | `GRF_F_ML_PRO_left.csv` | 11394801 | 22063113 | 148611612 | smoke-v2, full |
| ml_grf_right | `GRF_F_ML_PRO_right.csv` | 11394786 | 22063086 | 148595437 | smoke-v2, full |
| cop_ap_left | `GRF_COP_AP_PRO_left.csv` | 11394768 | 22062963 | 138556183 | smoke-source, full |
| cop_ap_right | `GRF_COP_AP_PRO_right.csv` | 11394765 | 22062957 | 138508833 | smoke-source, full |
| cop_ml_left | `GRF_COP_ML_PRO_left.csv` | 11394783 | 22063077 | 155283664 | smoke-source, full |
| cop_ml_right | `GRF_COP_ML_PRO_right.csv` | 11394780 | 22063071 | 154923817 | smoke-source, full |

Smoke v1은 metadata, vertical GRF, AP GRF, COP AP, COP ML을 필수로 한다. ML GRF는 용량과 우선순위 때문에 smoke v2에 둔다. Full run에서는 모든 processed GRF/COP 파일을 사용한다.

### 4.3 SIAT manifest 항목

`config/siat_manifest.json`에는 압축 archive 자체만 기록한다.

| role | filename | article_id | file_id | size_bytes | required_for |
| --- | --- | ---: | ---: | ---: | --- |
| siat_archive | `SIAT_LLMD20230404.rar` | 22776389 | 40468208 | 8374239382 | siat-inspect |

SIAT archive는 크기가 약 8.37GB이므로 smoke 검증에 포함하지 않는다. SIAT 구현은 먼저 사용자가 별도로 받은 sample 또는 압축 해제된 일부 walking 파일 구조를 inspect하는 단계로 시작한다.

## 5. CLI 계약

모든 명령은 project root에서 실행한다. 로컬 환경에 `python`이 PATH에 없을 수 있으므로 README에는 `<python>` placeholder를 먼저 정하도록 안내한다.

### 5.1 의존성 설치

```powershell
<python> -m pip install -r requirements.txt
```

Acceptance check:

- `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn` import가 성공해야 한다.
- 의존성이 없으면 tests는 `ModuleNotFoundError`로 실패할 수 있다.

### 5.2 Data downloader

```powershell
<python> scripts/download_data.py --dataset gaitrec --set smoke-source --manifest config/gaitrec_processed_manifest.json --output-root data/source/gaitrec
```

Args:

| arg | required | default | 의미 |
| --- | --- | --- | --- |
| `--dataset` | yes | none | `gaitrec` 또는 `siat` |
| `--set` | yes | none | `smoke-source`, `smoke-v2`, `full`, `siat-inspect` |
| `--manifest` | yes | none | manifest JSON path |
| `--output-root` | yes | none | 파일 저장 root |
| `--overwrite` | no | false | 기존 파일이 있어도 다시 다운로드 |
| `--dry-run` | no | false | 다운로드 없이 선택 파일과 예상 용량만 출력 |

Behavior:

- `required_for`에 `--set`이 포함된 file만 선택한다.
- 기존 파일이 있고 size가 맞으면 skip한다.
- 기존 파일 size가 다르면 실패한다. `--overwrite`가 있을 때만 다시 받는다.
- 다운로드 완료 후 실제 byte size가 `size_bytes`와 다르면 실패하고 해당 파일을 삭제하지 않는다.
- 네트워크 실패 시 어떤 file에서 실패했는지 출력한다.

### 5.3 Smoke subset builder

```powershell
<python> scripts/build_smoke_subset.py --input-root data/source/gaitrec --output-root data/gaitrec_smoke --max-bytes 250000000 --seed 42
```

Args:

| arg | required | default | 의미 |
| --- | --- | --- | --- |
| `--input-root` | yes | none | downloaded source CSV root |
| `--output-root` | yes | none | subset CSV 저장 root |
| `--max-bytes` | no | 250000000 | output total size limit |
| `--seed` | no | 42 | reproducible sampling seed |
| `--min-subjects-per-label` | no | 10 | label별 최소 subject 수 |
| `--min-trials-per-label` | no | 30 | label별 최소 trial 수 |
| `--include-ml-grf` | no | false | smoke v2에서 ML GRF 포함 |

Behavior:

- metadata에서 label별 subject를 seed 기반으로 뽑는다.
- 선택 단위는 row가 아니라 `SUBJECT_ID + SESSION_ID + TRIAL_ID` key다.
- 모든 selected key가 left/right 및 selected signal files에 존재해야 한다.
- output total size가 `--max-bytes`를 넘으면 label별 subject 수를 줄여 다시 생성한다.
- 최종 sampling manifest를 `output-root/smoke_sampling_manifest.json`에 기록한다.

### 5.4 Main pipeline

```powershell
<python> scripts/run_pipeline.py --gaitrec-root data/gaitrec_smoke --output-root results/gaitrec_smoke
```

Required outputs:

- `results/gaitrec_smoke/tables/gaitrec_features.csv`
- `results/gaitrec_smoke/tables/model_metrics.csv`
- `results/gaitrec_smoke/tables/group_feature_summary.csv`
- `results/gaitrec_smoke/tables/permutation_importance.csv`
- `results/gaitrec_smoke/tables/logistic_coefficients.csv`
- `results/gaitrec_smoke/figures/workflow.svg`
- `results/gaitrec_smoke/figures/model_metrics.svg`
- `results/gaitrec_smoke/figures/confusion_matrix.svg`
- `results/gaitrec_smoke/figures/permutation_importance.svg`
- `results/gaitrec_smoke/figures/group_mean_vgrf_curve.svg`
- `results/gaitrec_smoke/figures/group_ap_impulse_comparison.svg`
- `results/gaitrec_smoke/figures/group_cop_comparison.svg`
- `results/gaitrec_smoke/reports/final_analysis_report.md`
- `results/gaitrec_smoke/reports/example_subject_rehab_focus.md`
- `results/gaitrec_smoke/reports/siat_reference_note.md`

### 5.5 SIAT inspect

```powershell
<python> scripts/inspect_siat.py --siat-root data/source/siat --output-root results/siat_inspection
```

Required outputs:

- `results/siat_inspection/tables/siat_file_inventory.csv`
- `results/siat_inspection/tables/siat_column_candidates.csv`
- `results/siat_inspection/reports/siat_structure_report.md`

이 단계는 parser 구현 전에 파일 구조를 확정하기 위한 조사 단계다. 이 단계의 결과 없이 SIAT parser column mapping을 추측해서 구현하지 않는다.

## 6. Smoke subset sampling 기준

Smoke subset은 실제 CSV 구조를 유지하되 전체 용량을 250MB 미만으로 줄인 검증용 데이터다.

Sampling 기준:

- key columns: `SUBJECT_ID`, `SESSION_ID`, `TRIAL_ID`.
- metadata는 선택된 subject/session row만 남긴다.
- signal files는 선택된 key row만 남긴다.
- label은 `Healthy`, `Hip`, `Knee`, `Ankle`, `Calcaneus`를 모두 포함한다.
- left/right pair는 반드시 함께 남긴다.
- AP/COP selected files 간 key set은 동일해야 한다.

Size control:

- 먼저 label별 `min_subjects_per_label`을 만족하도록 subject를 선택한다.
- 예상 output byte를 계산한다.
- 250MB를 넘으면 label별 subject 수를 동일 비율로 줄인다.
- 줄여도 `min_trials_per_label`을 만족하지 못하면 실패하고 기준 완화를 안내한다.

Sampling manifest fields:

```json
{
  "seed": 42,
  "max_bytes": 250000000,
  "include_ml_grf": false,
  "selected_key_count": 1234,
  "selected_subject_count_by_label": {
    "Healthy": 10,
    "Hip": 10,
    "Knee": 10,
    "Ankle": 10,
    "Calcaneus": 10
  },
  "output_size_bytes": 0,
  "source_files": []
}
```

## 7. Feature table schema

분석 단위는 `subject_id + session_id + trial_id + affected_side`다.

### 7.1 Identity/covariate columns

| column | type | required | source |
| --- | --- | --- | --- |
| `subject_id` | string | yes | metadata/signal key |
| `session_id` | string | yes | metadata/signal key |
| `trial_id` | string | yes | signal key, session과 결합 가능 |
| `label` | string | yes | metadata class label |
| `affected_side` | string | yes | metadata affected side |
| `walking_speed` | float | yes if present | metadata |
| `age` | float | yes if present | metadata |
| `sex` | string | yes if present | metadata |
| `height` | float | yes if present | metadata |
| `weight` | float | yes if present | metadata |
| `shoe_condition` | string | yes if present | metadata |

### 7.2 GRF/COP feature columns

| column | type | missing policy | formula |
| --- | --- | --- | --- |
| `vgrf_peak_aff` | float | required when vertical exists | max affected-side vertical GRF |
| `vgrf_peak_unaff` | float | required when vertical exists | max unaffected-side vertical GRF |
| `vgrf_peak_asym` | float | NaN if either side missing | `(aff - unaff) / mean(abs(aff), abs(unaff))` |
| `loading_rate_asym` | float | NaN if either side missing | asymmetry of early loading proxy |
| `ap_braking_impulse_asym` | float | NaN if AP missing | asymmetry of negative AP impulse magnitude |
| `ap_propulsion_impulse_asym` | float | NaN if AP missing | asymmetry of positive AP impulse |
| `push_off_index` | float | NaN if AP missing | affected-side propulsion impulse |
| `cop_ap_range_aff` | float | NaN if COP AP missing | max-min affected-side COP AP |
| `cop_ml_range_aff` | float | NaN if COP ML missing | max-min affected-side COP ML |
| `cop_path_length_aff` | float | NaN if either COP AP/ML missing | sum of 2D COP step distances |
| `cop_ap_range_asym` | float | NaN if either side missing | COP AP range asymmetry |
| `cop_ml_range_asym` | float | NaN if either side missing | COP ML range asymmetry |

Missing policy:

- 계산할 수 없는 feature는 `NaN`으로 둔다.
- `0`으로 채우지 않는다.
- final report에는 available/unavailable feature list를 기록한다.

## 8. Data loader와 feature 구현 기준

Loader는 두 모드를 지원한다.

- 탐색 모드: 현재처럼 root 아래 파일명을 찾아 signal key로 매핑한다.
- manifest 모드: manifest의 `role`과 `target_path`를 기준으로 정확한 파일을 읽는다.

검증 규칙:

- vertical left/right는 GaitRec feature pipeline의 최소 필수 파일이다.
- AP/COP 파일이 없으면 pipeline은 계속 실행하되 해당 feature를 unavailable로 기록한다.
- 파일이 있는데 selected key가 맞지 않으면 실패한다.
- signal row는 wide format을 기본으로 한다. 첫 세 key column 이후 numeric columns를 gait-cycle series로 해석한다.
- `SUBJECT_ID`, `SESSION_ID`, `TRIAL_ID`는 canonical column인 `subject_id`, `session_id`, `trial_id`로 normalize한다.

Feature 구현 기준:

- 현재 `features.py`의 `peak`, `braking_impulse`, `propulsion_impulse`, `signal_range`, `cop_path_length`, `asymmetry` 함수를 유지한다.
- AP impulse는 processed AP signal의 음수 구간을 braking, 양수 구간을 propulsion으로 계산한다.
- COP path는 AP/ML 길이가 다르면 짧은 길이에 맞춰 계산한다.
- affected side가 없거나 알 수 없으면 기존처럼 기본 `left`로 두되, final report에 metadata warning을 남긴다.

## 9. Modeling 설계

모델 경로는 scikit-learn을 기본으로 한다.

Models:

- `dummy`: `DummyClassifier(strategy="most_frequent")`.
- `logistic_regression`: `LogisticRegression(multi_class="multinomial", class_weight="balanced", max_iter=2000)`.
- `random_forest`: `RandomForestClassifier(class_weight="balanced_subsample", n_estimators=500, max_depth=8, min_samples_leaf=5)`.

Preprocessing:

- numeric: median imputation + standard scaling.
- categorical: most-frequent imputation + one-hot encoding.
- train/test split: 반드시 `subject_id` 기준 split.

Fallback:

- scikit-learn이 없을 때만 numpy fallback을 사용한다.
- fallback 결과는 smoke 실행 가능성 확인용이며 final 발표 결과에서는 sklearn 결과를 우선한다.
- report에는 `used_sklearn` 여부를 명시한다.

Metrics:

- balanced accuracy.
- macro F1.
- class별 precision/recall/F1.
- confusion matrix.
- baseline 대비 improvement.
- permutation importance.
- logistic coefficient table.

## 10. Figure/report 설계

### 10.1 필수 table

- `label_counts.csv`: label별 subject/trial count.
- `group_feature_summary.csv`: label별 feature mean/std/95% CI.
- `model_metrics.csv`: model별 balanced accuracy/macro F1.
- `permutation_importance.csv`: best model 기준 permutation importance.
- `logistic_coefficients.csv`: logistic regression coefficient.

### 10.2 필수 figure

- `workflow.svg`: 전체 분석 흐름.
- `model_metrics.svg`: 모델별 지표.
- `confusion_matrix.svg`: best model confusion matrix.
- `permutation_importance.svg`: 상위 feature importance.
- `group_mean_vgrf_curve.svg`: class별 평균 vertical GRF curve.
- `group_ap_impulse_comparison.svg`: class별 AP braking/propulsion 비교.
- `group_cop_comparison.svg`: class별 COP AP/ML range 또는 COP path 비교.

Figure 생성 기준:

- matplotlib이 있으면 matplotlib로 생성한다.
- matplotlib이 없으면 현재 fallback SVG writer와 같은 단순 SVG writer를 사용한다.
- figure가 unavailable이면 빈 파일을 만들지 말고 설명 SVG를 생성한다.

### 10.3 필수 report

`final_analysis_report.md`는 다음 섹션을 포함한다.

- Problem Definition.
- Data.
- Methods.
- Available/Unavailable Features.
- Results.
- Group Feature Highlights.
- Model Interpretation.
- SIAT Reference Note.
- Limits.
- Future Direction.

`example_subject_rehab_focus.md`는 다음을 포함한다.

- subject/trial id.
- model pattern group.
- evidence features 3개 이상.
- priority check candidates 2개 이상.
- interpretation guardrail.

Forbidden terms:

- `진단`
- `처방`
- `원인 확정`
- `특정 근육 약화 확정`
- `diagnosis`
- `prescription`
- `cause confirmed`
- `specific muscle weakness confirmed`

Forbidden term guardrail은 tests에서 유지한다. 단, 문서가 금지 표현 목록 자체를 설명하는 경우는 테스트 대상 report output에서 제외한다.

## 11. SIAT-LLMD auxiliary 설계

SIAT는 GaitRec 모델 입력으로 병합하지 않는다.

구현 순서:

1. `scripts/inspect_siat.py`로 파일 inventory를 만든다.
2. walking 관련 sample 후보를 찾는다.
3. gait phase column 후보를 찾는다.
4. EMG channel 후보를 찾는다.
5. ankle/knee/hip torque column 후보를 찾는다.
6. column mapping을 `results/siat_inspection/reports/siat_structure_report.md`에 사람이 확인 가능하게 기록한다.
7. mapping이 확정된 뒤 `src/gait_rehab/siat.py`에 parser를 추가한다.

SIAT output:

- gait phase별 EMG 평균 curve.
- ankle/knee/hip torque와 EMG timing reference figure.
- `siat_reference_note.md`.

Report 표현:

- 허용: “SIAT는 건강인 walking sample에서 EMG/torque timing을 참고하기 위한 보조 자료다.”
- 금지: “SIAT가 GaitRec subject의 원인을 설명한다.”
- 금지: “SIAT reference가 정상/비정상 판정 기준이다.”

## 12. 테스트와 acceptance checks

### 12.1 기본 테스트 명령

```powershell
<python> -m unittest discover -s tests
```

현재 로컬 전역 Python 환경에서는 `pandas`, `numpy`가 없으면 테스트 import 단계에서 실패한다. 구현자는 먼저 `requirements.txt` 설치를 수행한다.

### 12.2 신규/수정 테스트 목록

| test file | check |
| --- | --- |
| `tests/test_manifest_downloader.py` | manifest required fields, selected set filtering, size mismatch failure |
| `tests/test_smoke_subset.py` | selected key consistency across metadata/signal files, output size under limit, label coverage |
| `tests/test_data.py` | manifest role 기반 loader, optional AP/COP missing behavior |
| `tests/test_features.py` | AP impulse, COP path/range, NaN missing policy |
| `tests/test_sklearn_models.py` | dummy/logistic/random forest run with sklearn, subject leakage prevention |
| `tests/test_split_and_reports.py` | report forbidden terms, available/unavailable features, markdown table generation |
| `tests/test_siat_reference.py` | SIAT output remains auxiliary and is not included in model feature columns |

### 12.3 Smoke acceptance command sequence

```powershell
<python> scripts/download_data.py --dataset gaitrec --set smoke-source --manifest config/gaitrec_processed_manifest.json --output-root data/source/gaitrec
<python> scripts/build_smoke_subset.py --input-root data/source/gaitrec --output-root data/gaitrec_smoke --max-bytes 250000000 --seed 42
<python> scripts/run_pipeline.py --gaitrec-root data/gaitrec_smoke --output-root results/gaitrec_smoke
<python> -m unittest discover -s tests
```

Pass 기준:

- smoke output total size가 250MB 미만이다.
- `gaitrec_features.csv`에 필수 identity/covariate와 GRF/COP feature columns가 있다.
- smoke v1에서는 vertical, AP, COP AP, COP ML 기반 필수 feature가 non-null로 생성된다.
- ML GRF가 없어도 pipeline은 실패하지 않는다.
- model metrics, confusion matrix, permutation importance, logistic coefficients가 생성된다.
- final report가 unavailable feature와 SIAT limitation을 명시한다.
- report output에 forbidden terms가 없다.

### 12.4 Full data acceptance command sequence

```powershell
<python> scripts/download_data.py --dataset gaitrec --set full --manifest config/gaitrec_processed_manifest.json --output-root data/source/gaitrec_full
<python> scripts/run_pipeline.py --gaitrec-root data/source/gaitrec_full --output-root results/gaitrec_full
```

Pass 기준:

- full processed GRF/COP 파일을 같은 feature/output schema로 처리한다.
- memory 부담이 큰 경우 intermediate feature table 저장 또는 chunking 전략을 적용한다.
- full run 결과도 smoke run과 같은 table/report/figure 이름을 사용한다.

### 12.5 SIAT acceptance command sequence

```powershell
<python> scripts/download_data.py --dataset siat --set siat-inspect --manifest config/siat_manifest.json --output-root data/source/siat
<python> scripts/inspect_siat.py --siat-root data/source/siat --output-root results/siat_inspection
```

Pass 기준:

- SIAT file inventory와 column candidate table이 생성된다.
- parser 구현 전이라도 `siat_structure_report.md`가 다음 단계 결정을 도와야 한다.
- SIAT output은 GaitRec classifier feature columns에 들어가지 않는다.

## 13. 구현 순서

권장 순서:

1. Manifest JSON과 downloader를 구현한다.
2. downloader mock test와 size check test를 통과시킨다.
3. smoke subset builder를 구현한다.
4. smoke key consistency와 size limit test를 통과시킨다.
5. AP/COP loader와 feature extraction 검증을 보강한다.
6. sklearn model path test를 추가하고 통과시킨다.
7. class별 GRF/AP/COP figure를 추가한다.
8. final/example report를 강화한다.
9. SIAT inspect CLI를 구현한다.
10. README 실행 절차를 업데이트한다.
11. smoke acceptance sequence를 처음부터 끝까지 실행한다.

## 14. Guardrails

- `data/`와 `results/`는 git에 포함하지 않는다.
- full source CSV와 smoke subset CSV를 commit하지 않는다.
- SIAT는 GaitRec classifier 입력에 병합하지 않는다.
- subject-level split만 사용한다.
- AP/COP 파일이 없는 run에서 feature를 `0`으로 채우지 않는다.
- final report는 높은 정확도 과시보다 설명 가능성과 재활 평가 후보 제안을 우선한다.
- 결과 문장은 impairment group과 유사한 gait pattern, weight-bearing, push-off, COP stability 같은 기능적 표현으로 제한한다.

## 15. 출처

- GaitRec paper: https://www.nature.com/articles/s41597-020-0481-z
- GaitRec Figshare collection API: https://api.figshare.com/v2/collections/4788012/articles?page_size=50
- SIAT-LLMD Figshare article API: https://api.figshare.com/v2/articles/22776389

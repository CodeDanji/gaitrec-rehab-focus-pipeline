# 동료용 전체 데이터셋 E2E 검증 가이드

이 문서는 동료가 전체 GaitRec processed 데이터셋을 내려받고, 프로젝트 파이프라인을 끝까지 실행해 결과물을 검증하기 위한 안내서입니다.

핵심 원칙:

- 기본 검증 경로는 `full` 데이터셋입니다.
- `data/`와 `results/`는 git에 올리지 않습니다.
- SIAT-LLMD는 GaitRec classifier 입력으로 섞지 않고, 별도 inspect/reference 경로로만 확인합니다.

## 1. 프로젝트 받기

```powershell
git clone https://github.com/CodeDanji/gaitrec-rehab-focus-pipeline.git
cd gaitrec-rehab-focus-pipeline
```

## 2. Python 환경 준비

Python 3.12 기준으로 검증했습니다. 아래 예시의 `<python>`은 본인 환경의 Python 명령으로 바꿔 쓰세요.

예:

```powershell
python
py -3
C:\path\to\python.exe
```

의존성 설치:

```powershell
<python> -m pip install -r requirements.txt
```

설치 확인:

```powershell
<python> -c "import pandas, numpy, sklearn, matplotlib, seaborn; print('ok')"
```

`ok`가 나오면 준비가 끝난 것입니다.

## 3. 전체 GaitRec 데이터셋 다운로드

전체 processed GRF/COP 파일을 받습니다. manifest 기준 총 11개 파일, 약 1.45GB입니다. pandas로 로딩하면 메모리를 더 사용하므로 여유 RAM과 디스크 공간을 확보하세요.

먼저 dry-run으로 받을 파일 목록과 총 용량을 확인합니다.

```powershell
<python> scripts\download_data.py --dataset gaitrec --set full --manifest config\gaitrec_processed_manifest.json --output-root data\source\gaitrec_full --dry-run
```

문제가 없으면 실제 다운로드를 진행합니다.

```powershell
<python> scripts\download_data.py --dataset gaitrec --set full --manifest config\gaitrec_processed_manifest.json --output-root data\source\gaitrec_full
```

예상 파일:

| role | file | bytes |
| --- | --- | ---: |
| metadata | `GRF_metadata.csv` | 629,530 |
| vgrf_left | `GRF_F_V_PRO_left.csv` | 136,725,253 |
| vgrf_right | `GRF_F_V_PRO_right.csv` | 136,713,924 |
| ap_grf_left | `GRF_F_AP_PRO_left.csv` | 147,540,140 |
| ap_grf_right | `GRF_F_AP_PRO_right.csv` | 147,554,970 |
| ml_grf_left | `GRF_F_ML_PRO_left.csv` | 148,611,612 |
| ml_grf_right | `GRF_F_ML_PRO_right.csv` | 148,595,437 |
| cop_ap_left | `GRF_COP_AP_PRO_left.csv` | 138,556,183 |
| cop_ap_right | `GRF_COP_AP_PRO_right.csv` | 138,508,833 |
| cop_ml_left | `GRF_COP_ML_PRO_left.csv` | 155,283,664 |
| cop_ml_right | `GRF_COP_ML_PRO_right.csv` | 154,923,817 |

다운로드가 중간에 끊기면 같은 명령을 다시 실행하세요. 이미 받은 파일의 size가 맞으면 skip하고 남은 파일만 처리합니다.

## 4. 전체 파이프라인 실행

전체 데이터셋으로 feature extraction, subject-level split, sklearn model training, table/figure/report 생성을 한 번에 실행합니다.

```powershell
<python> scripts\run_pipeline.py --gaitrec-root data\source\gaitrec_full --gaitrec-manifest config\gaitrec_processed_manifest.json --output-root results\gaitrec_full
```

성공하면 마지막에 다음과 비슷한 메시지가 나옵니다.

```text
Analysis outputs written to results\gaitrec_full
```

주의:

- full run은 빠른 검증보다 훨씬 오래 걸릴 수 있습니다.
- `RandomForestClassifier(n_estimators=500)`를 사용하므로 CPU와 메모리를 꽤 사용합니다.
- AP/COP 파일이 모두 있으면 AP impulse, COP range/path 기반 feature도 실제 값으로 생성됩니다.

## 5. 결과물 확인

결과는 `results\gaitrec_full` 아래에 생성됩니다.

테이블:

```powershell
Get-ChildItem results\gaitrec_full\tables
```

필수 table:

- `gaitrec_features.csv`
- `model_metrics.csv`
- `group_feature_summary.csv`
- `label_counts.csv`
- `permutation_importance.csv`
- `logistic_coefficients.csv`
- model별 classification report와 confusion matrix CSV

그림:

```powershell
Get-ChildItem results\gaitrec_full\figures
```

필수 figure:

- `workflow.svg`
- `model_metrics.svg`
- `confusion_matrix.svg`
- `permutation_importance.svg`
- `group_mean_vgrf_curve.svg`
- `group_ap_impulse_comparison.svg`
- `group_cop_comparison.svg`
- `siat_reference_placeholder.svg`

보고서:

```powershell
Get-ChildItem results\gaitrec_full\reports
```

필수 report:

- `final_analysis_report.md`
- `example_subject_rehab_focus.md`
- `siat_reference_note.md`

모델 성능 확인:

```powershell
Import-Csv results\gaitrec_full\tables\model_metrics.csv | Format-Table -AutoSize
```

feature table 크기 확인:

```powershell
<python> -c "import pandas as pd; f=pd.read_csv('results/gaitrec_full/tables/gaitrec_features.csv'); print(f.shape); print(f.notna().sum())"
```

## 6. 테스트와 guardrail 확인

full run이 끝난 뒤 unit test를 실행합니다.

```powershell
<python> -m unittest discover -s tests
```

보고서에 금지 표현이 들어가지 않았는지도 확인합니다.

```powershell
rg -n "diagnosis|prescription|cause confirmed|specific muscle weakness confirmed|진단|처방|원인 확정|특정 근육 약화 확정" results\gaitrec_full\reports
```

아무 출력도 없으면 통과입니다.

## 7. SIAT-LLMD inspect 실행

SIAT-LLMD는 GaitRec classifier 입력으로 병합하지 않습니다. 별도 보조 reference로만 다룹니다.

SIAT archive는 약 8.37GB입니다. 전체 검증에서 SIAT까지 확인하려면 먼저 다운로드합니다.

```powershell
<python> scripts\download_data.py --dataset siat --set siat-inspect --manifest config\siat_manifest.json --output-root data\source\siat
```

archive를 받은 뒤 압축을 풀어 walking 관련 파일을 확인할 수 있는 상태로 둡니다. 압축 해제 방식은 환경마다 다를 수 있습니다.

압축 해제된 SIAT 파일 구조를 inspect합니다.

```powershell
<python> scripts\inspect_siat.py --siat-root data\source\siat --output-root results\siat_inspection
```

생성 파일:

- `results\siat_inspection\tables\siat_file_inventory.csv`
- `results\siat_inspection\tables\siat_column_candidates.csv`
- `results\siat_inspection\reports\siat_structure_report.md`

이 단계는 parser mapping을 확정하는 단계가 아니라, 사람이 파일/컬럼 구조를 확인하기 위한 조사 단계입니다.

## 8. 자주 생기는 문제

### `python` 명령이 안 먹는 경우

Windows에서 `python`이 PATH에 없을 수 있습니다. Python 실행 파일 전체 경로를 `<python>` 자리에 넣어 실행하세요.

### 다운로드 중간 실패

같은 명령을 다시 실행하세요. size가 맞는 파일은 skip됩니다.

### 기존 파일 size mismatch

파일이 덜 받아졌거나 깨졌을 수 있습니다. 해당 파일을 삭제하고 다시 받거나 `--overwrite`를 사용하세요.

```powershell
<python> scripts\download_data.py --dataset gaitrec --set full --manifest config\gaitrec_processed_manifest.json --output-root data\source\gaitrec_full --overwrite
```

### full run이 너무 오래 걸리는 경우

전체 데이터셋은 파일 크기와 모델 학습 때문에 시간이 걸릴 수 있습니다. 다운로드가 끝났는지 먼저 확인하고, 실행 중이라면 `results\gaitrec_full\tables\gaitrec_features.csv`가 생성되었는지 확인해 어느 단계까지 진행됐는지 판단합니다.

### 결과를 git에 올리고 싶은 경우

올리지 마세요. `data/`와 `results/`는 `.gitignore` 대상입니다. 동료에게는 실행 명령, 검증 로그, 결과 요약만 공유하는 방식을 권장합니다.

## 9. 전체 데이터셋 인수 체크리스트

- [ ] `requirements.txt` 설치 성공
- [ ] `gaitrec --set full` 다운로드 성공
- [ ] `results\gaitrec_full\tables\gaitrec_features.csv` 생성
- [ ] `results\gaitrec_full\tables\model_metrics.csv` 생성
- [ ] `results\gaitrec_full\figures\confusion_matrix.svg` 생성
- [ ] `results\gaitrec_full\figures\group_ap_impulse_comparison.svg` 생성
- [ ] `results\gaitrec_full\figures\group_cop_comparison.svg` 생성
- [ ] `results\gaitrec_full\reports\final_analysis_report.md` 생성
- [ ] `python -m unittest discover -s tests` 통과
- [ ] report forbidden-term guardrail 통과
- [ ] SIAT inspect를 실행했다면 `siat_structure_report.md` 생성 확인
- [ ] `data/`와 `results/`를 git에 commit하지 않음

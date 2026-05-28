# 500MB Real-Data Validation Run

Date: 2026-05-26

This note records the push-readiness validation performed before publishing the
code. Data and generated results are intentionally excluded from git.

## Environment

- Python: bundled Codex Python runtime
- Dependencies installed from `requirements.txt`
- GitHub publish target: `CodeDanji/gaitrec-rehab-focus-pipeline`

## Downloaded Data

Command:

```powershell
<python> scripts\download_data.py --dataset gaitrec --set vertical-500mb --manifest config\gaitrec_processed_manifest.json --output-root data\gaitrec_subset_500mb
```

Downloaded files:

| file | bytes |
| --- | ---: |
| `GRF_metadata.csv` | 629,530 |
| `GRF_F_V_PRO_left.csv` | 136,725,253 |
| `GRF_F_V_PRO_right.csv` | 136,713,924 |

Total: 274,068,707 bytes.

## Pipeline Run

Command:

```powershell
<python> scripts\run_pipeline.py --gaitrec-root data\gaitrec_subset_500mb --gaitrec-manifest config\gaitrec_processed_manifest.json --output-root results\gaitrec_subset_500mb_sklearn
```

The run completed using scikit-learn models.

Feature table:

- Rows: 75,732
- Columns: 23
- Non-null vertical features: 75,732 rows for `vgrf_peak_aff`,
  `vgrf_peak_unaff`, `vgrf_peak_asym`, and `loading_rate_asym`

Label counts:

| label | subject_count | trial_count |
| --- | ---: | ---: |
| Ankle | 627 | 21,386 |
| Calcaneus | 382 | 13,970 |
| Healthy | 211 | 7,755 |
| Hip | 450 | 12,748 |
| Knee | 625 | 19,873 |

Model metrics:

| model | balanced_accuracy | macro_f1 | support | used_sklearn |
| --- | ---: | ---: | ---: | --- |
| dummy | 0.200 | 0.086 | 15,006 | True |
| logistic_regression | 0.367 | 0.329 | 15,006 | True |
| random_forest | 0.481 | 0.465 | 15,006 | True |

Best model by macro F1: `random_forest`.

## Generated Outputs

Expected tables, reports, and SVG figures were generated under
`results\gaitrec_subset_500mb_sklearn`.

Key figure files:

- `workflow.svg`
- `model_metrics.svg`
- `confusion_matrix.svg`
- `permutation_importance.svg`
- `group_mean_vgrf_curve.svg`
- `group_vgrf_peak_aff_summary.svg`
- `group_ap_impulse_comparison.svg`
- `group_cop_comparison.svg`
- `siat_reference_placeholder.svg`

Because this validation subset is vertical-GRF only, AP/COP derived features
are marked unavailable in the report and the AP/COP figures are explanatory
fallback SVGs.

## Test And Guardrail Checks

Command:

```powershell
<python> -m unittest discover -s tests
```

Result:

```text
Ran 19 tests in 11.502s

OK
```

Forbidden report wording check:

```powershell
rg -n "diagnosis|prescription|cause confirmed|specific muscle weakness confirmed|진단|처방|원인 확정|특정 근육 약화 확정" results\gaitrec_subset_500mb_sklearn\reports
```

Result: no matches.

## Known Scope Limits

- The under-500MB validation subset covers metadata and vertical GRF only.
- AP GRF and COP files are available in the manifest but were not downloaded for
  this under-500MB validation run.
- SIAT-LLMD is implemented as an inspection/auxiliary path and is not merged
  into GaitRec classifier features.
- Full processed GaitRec and full SIAT archive runs were not executed because
  they exceed the quick handoff validation size.

# Progress: GaitRec Vertical-GRF Subset Validation

Date: 2026-05-26

## Current Repository Scope

This repository is prepared as a public validation snapshot for a GaitRec
vertical-GRF subset pipeline. It is not yet a complete GRF/COP + SIAT-LLMD
analysis package.

Implemented and verified:

- GaitRec metadata loading.
- Processed vertical GRF left/right loading.
- GaitRec session/trial structure support using `SUBJECT_ID`, `SESSION_ID`, and
  `TRIAL_ID`.
- Feature extraction for vertical-GRF peak, vertical-GRF asymmetry, and loading
  rate asymmetry.
- Subject-level train/test split.
- Baseline and fallback classifiers in environments without `scikit-learn`.
- Metrics, confusion matrix, permutation importance, group summary, final report,
  and example subject rehab-focus report.

Not implemented as a working pipeline:

- SIAT-LLMD EMG/torque parser and analysis.
- AP GRF, ML GRF, COP AP, and COP ML verified real-data run.
- Full scikit-learn model verification in the bundled runtime, because
  `scikit-learn` was not installed there.

## Downloaded Data Used For Verification

The local verification used the official GaitRec Figshare collection:

- Collection DOI: https://doi.org/10.6084/m9.figshare.c.4788012.v1
- Paper: https://www.nature.com/articles/s41597-020-0481-z

Downloaded files:

| File | Size |
| --- | ---: |
| `GRF_metadata.csv` | 629,530 bytes |
| `GRF_F_V_PRO_left.csv` | 136,725,253 bytes |
| `GRF_F_V_PRO_right.csv` | 136,713,924 bytes |
| Total | 274,068,707 bytes |

These files are stored locally under `data/gaitrec_subset/` and are excluded
from git.

## Verified Run

Command:

```powershell
python scripts\run_pipeline.py --gaitrec-root data\gaitrec_subset --output-root results\gaitrec_vertical_subset_v2
```

Generated feature table:

- Rows: 75,732
- Columns: 23
- Unique subjects: 2,295
- Labels: `Healthy`, `Hip`, `Knee`, `Ankle`, `Calcaneus`

Label counts:

| Label | Subjects | Trials |
| --- | ---: | ---: |
| Ankle | 627 | 21,386 |
| Calcaneus | 382 | 13,970 |
| Healthy | 211 | 7,755 |
| Hip | 450 | 12,748 |
| Knee | 625 | 19,873 |

Model metrics from the minimal runtime fallback models:

| Model | Balanced accuracy | Macro F1 |
| --- | ---: | ---: |
| Dummy baseline | 0.200 | 0.086 |
| Softmax fallback | 0.365 | 0.327 |
| Nearest centroid fallback | 0.318 | 0.274 |

The final report explicitly states which features were available in this
vertical-only run and which AP/COP features were unavailable.

## Test Verification

Command:

```powershell
python -m unittest discover -s tests
```

Result:

```text
Ran 8 tests
OK
```

Tests cover:

- Synthetic feature formulas.
- Realistic GaitRec session/trial signal shape.
- Vertical-only subset loading.
- Subject-level split leakage prevention.
- Rehab report evidence and forbidden term guardrails.
- Markdown report generation without optional `tabulate`.

## SIAT-LLMD Status

SIAT-LLMD is currently a documented future extension only. The code writes a
placeholder note and figure, but it does not parse SIAT sample files or generate
real EMG/torque timing plots.

This is intentional for the current public snapshot: the verified claim is
limited to the GaitRec vertical-GRF subset pipeline.

## Public Release Notes

Before publishing:

- Keep `data/` and `results/` ignored.
- Include code, tests, docs, `README.md`, `requirements.txt`, `.gitignore`, and
  `LICENSE`.
- Describe the repository as a GaitRec vertical-GRF validation snapshot.
- Do not imply that SIAT-LLMD analysis is implemented.

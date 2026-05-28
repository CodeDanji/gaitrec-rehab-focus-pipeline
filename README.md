# GaitRec GRF/COP Rehab Focus Pipeline

This repository contains a reproducible gait-analysis pipeline for processed
GaitRec GRF/COP files, plus a separated SIAT-LLMD inspection path for auxiliary
EMG/torque reference context.

Current scope:

- Uses manifest-driven downloads for the official processed GaitRec files.
- Builds a key-consistent smoke subset under 250 MB for handoff validation.
- Extracts subject/trial-level vertical GRF, AP GRF, and COP features when the
  corresponding processed files are present.
- Runs subject-level train/test split, baseline, logistic regression, and random
  forest models when scikit-learn is installed. A lightweight numpy fallback is
  available for minimal environments.
- Writes model tables, feature summaries, required SVG figures, and reports.
- Does not include downloaded data or generated results in git.
- Keeps SIAT-LLMD separate from the GaitRec classifier. The first SIAT step is
  file/column inspection, not model input merging.

The full implementation contract is in `docs/IMPLEMENTATION_SPEC_FULL_PIPELINE.md`.

For Korean step-by-step handoff instructions, see
`docs/E2E_TEST_GUIDE_KO.md`.

## Python Setup

Choose a Python executable first. In the examples below, replace `<python>` with
your local command or full Python path.

```powershell
<python> -m pip install -r requirements.txt
```

## Download GaitRec Source Files

Dry-run the manifest selection:

```powershell
<python> scripts\download_data.py --dataset gaitrec --set smoke-source --manifest config\gaitrec_processed_manifest.json --output-root data\source\gaitrec --dry-run
```

For a quick real-data handoff under 500 MB, use the vertical-GRF subset:

```powershell
<python> scripts\download_data.py --dataset gaitrec --set vertical-500mb --manifest config\gaitrec_processed_manifest.json --output-root data\gaitrec_subset_500mb
<python> scripts\run_pipeline.py --gaitrec-root data\gaitrec_subset_500mb --gaitrec-manifest config\gaitrec_processed_manifest.json --output-root results\gaitrec_subset_500mb
```

Download the smoke-source files:

```powershell
<python> scripts\download_data.py --dataset gaitrec --set smoke-source --manifest config\gaitrec_processed_manifest.json --output-root data\source\gaitrec
```

For the full processed set, use `--set full` and a separate output directory if
you want to keep smoke and full sources apart.

Official data sources:

- GaitRec collection: https://api.figshare.com/v2/collections/4788012/articles?page_size=50
- GaitRec paper: https://www.nature.com/articles/s41597-020-0481-z

## Build a Smoke Subset

```powershell
<python> scripts\build_smoke_subset.py --input-root data\source\gaitrec --output-root data\gaitrec_smoke --max-bytes 250000000 --seed 42
```

The builder samples by `SUBJECT_ID + SESSION_ID + TRIAL_ID`, preserves matching
left/right signal rows across selected files, and writes
`data\gaitrec_smoke\smoke_sampling_manifest.json`.

## Run the Pipeline

Synthetic demo data:

```powershell
<python> scripts\run_pipeline.py --demo --output-root results\demo
```

Smoke subset:

```powershell
<python> scripts\run_pipeline.py --gaitrec-root data\gaitrec_smoke --gaitrec-manifest config\gaitrec_processed_manifest.json --output-root results\gaitrec_smoke
```

Full source directory:

```powershell
<python> scripts\download_data.py --dataset gaitrec --set full --manifest config\gaitrec_processed_manifest.json --output-root data\source\gaitrec_full
<python> scripts\run_pipeline.py --gaitrec-root data\source\gaitrec_full --gaitrec-manifest config\gaitrec_processed_manifest.json --output-root results\gaitrec_full
```

## Inspect SIAT-LLMD

```powershell
<python> scripts\download_data.py --dataset siat --set siat-inspect --manifest config\siat_manifest.json --output-root data\source\siat
<python> scripts\inspect_siat.py --siat-root data\source\siat --output-root results\siat_inspection
```

The SIAT inspection command writes inventory and candidate-column tables to help
decide a later parser mapping. SIAT outputs are not classifier features.

## Outputs

- `results/tables/gaitrec_features.csv`
- `results/tables/model_metrics.csv`
- `results/tables/group_feature_summary.csv`
- `results/tables/permutation_importance.csv`
- `results/tables/logistic_coefficients.csv`
- `results/figures/*.svg`
- `results/reports/final_analysis_report.md`
- `results/reports/example_subject_rehab_focus.md`
- `results/reports/siat_reference_note.md`

## Verify

```powershell
<python> -m unittest discover -s tests
```

The generated reports are constrained to gait-function screening language. They
avoid clinical determination or treatment assignment wording.

## License

Code in this repository is released under the MIT License. GaitRec data is not
redistributed here; download it from the official Figshare collection and cite
the original dataset/paper when using it.

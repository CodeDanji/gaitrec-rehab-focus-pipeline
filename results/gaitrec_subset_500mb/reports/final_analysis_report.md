# Final Analysis Report

## Problem Definition
This project uses processed GRF/COP gait features to quantify impairment-group-related gait function patterns and suggest rehabilitation assessment priorities.

## Data
The main analysis unit is subject_id + trial_id + affected_side. Train/test separation is performed by subject_id to avoid trial leakage.

### Label Counts
| label     | subject_count | trial_count |
| --------- | ------------- | ----------- |
| Ankle     | 627           | 21386       |
| Calcaneus | 382           | 13970       |
| Healthy   | 211           | 7755        |
| Hip       | 450           | 12748       |
| Knee      | 625           | 19873       |

## Methods
Feature extraction summarizes the gait signals available in the provided data subset.
Models include a most-frequent baseline plus interpretable and nonlinear classifiers when scikit-learn is available. Offline fallback models are used only to keep the demo runnable.

## Available/Unavailable Features
Available gait features in this run: vgrf_peak_aff, vgrf_peak_unaff, vgrf_peak_asym, loading_rate_asym.
Unavailable because the corresponding AP/COP files were not included: ap_braking_impulse_asym, ap_propulsion_impulse_asym, push_off_index, cop_ap_range_aff, cop_ml_range_aff, cop_path_length_aff, cop_ap_range_asym, cop_ml_range_asym.

## Results
| model               | balanced_accuracy | macro_f1 | support | used_sklearn |
| ------------------- | ----------------- | -------- | ------- | ------------ |
| dummy               | 0.200             | 0.086    | 15006   | True         |
| logistic_regression | 0.367             | 0.329    | 15006   | True         |
| random_forest       | 0.481             | 0.465    | 15006   | True         |

Best model by macro F1: `random_forest` with macro_f1=0.465 and balanced_accuracy=0.481.

### Group Feature Highlights
| label     | feature           | mean   | std   | n     |
| --------- | ----------------- | ------ | ----- | ----- |
| Healthy   | vgrf_peak_unaff   | 1.157  | 0.096 | 7755  |
| Healthy   | vgrf_peak_aff     | 1.155  | 0.098 | 7755  |
| Hip       | vgrf_peak_unaff   | 1.103  | 0.068 | 12748 |
| Calcaneus | vgrf_peak_unaff   | 1.102  | 0.073 | 13970 |
| Hip       | vgrf_peak_aff     | 1.099  | 0.068 | 12748 |
| Knee      | vgrf_peak_aff     | 1.098  | 0.068 | 19873 |
| Ankle     | vgrf_peak_aff     | 1.097  | 0.070 | 21386 |
| Knee      | vgrf_peak_unaff   | 1.097  | 0.069 | 19873 |
| Calcaneus | vgrf_peak_aff     | 1.094  | 0.069 | 13970 |
| Ankle     | vgrf_peak_unaff   | 1.094  | 0.067 | 21386 |
| Calcaneus | loading_rate_asym | -0.007 | 0.078 | 13970 |
| Hip       | loading_rate_asym | -0.004 | 0.085 | 12748 |
| Ankle     | loading_rate_asym | 0.003  | 0.077 | 21386 |
| Healthy   | loading_rate_asym | -0.001 | 0.045 | 7755  |
| Knee      | loading_rate_asym | 0.001  | 0.085 | 19873 |

## Model Interpretation
`used_sklearn` in this run: True. Feature importance and coefficient tables should be read as screening support for gait-function patterns, not clinical conclusions.

## SIAT Reference Note
SIAT-LLMD is kept separate from the GaitRec classifier and can only provide auxiliary EMG/torque timing context when inspected sample files are available.

## Limits
GaitRec does not include EMG, so muscle-level mechanisms are not inferred from this dataset. SIAT-LLMD is used only as healthy-reference context for EMG/torque timing.

## Future Direction
Add real processed GaitRec files, compare speed-normalized features, and use SIAT walking samples for a limited EMG/torque reference figure.
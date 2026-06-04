import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path.cwd() / "src"))
from gait_rehab.pipeline import run_stage1, get_cohort_primary_clean, get_cohort_all

def undersample_diverse(df: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    df = df.copy()
    is_healthy = df["label"].astype(str).str.lower().isin(["healthy", "control"])
    healthy_df = df[is_healthy]
    impaired_df = df[~is_healthy]
    
    n_healthy_subjects = healthy_df["subject_id"].nunique()
    if n_healthy_subjects == 0 or len(impaired_df) == 0:
        return df
        
    impaired_subtypes = impaired_df["label"].unique()
    target_subjects_per_subtype = max(1, n_healthy_subjects // len(impaired_subtypes))
    
    sampled_impaired_list = []
    rng = np.random.default_rng(random_state)
    
    # 🌟 Trial Cap: 각 환자당 최대 몇 번의 걸음(Trial)만 반영할 것인지 제한
    max_trials_per_subject = 10 
    
    for subtype in impaired_subtypes:
        subtype_df = impaired_df[impaired_df["label"] == subtype]
        subtype_subjects = subtype_df["subject_id"].unique()
        n_to_sample = min(len(subtype_subjects), target_subjects_per_subtype)
        
        # 1. 환자(Subject)를 50명 뽑음
        sampled_subjects = rng.choice(subtype_subjects, size=n_to_sample, replace=False)
        selected_subject_df = subtype_df[subtype_df["subject_id"].isin(sampled_subjects)]
        
        # 2. 뽑힌 환자들의 데이터를 가져오되, 한 환자당 최대 10개의 Trial만 샘플링 (과대표집 완벽 차단!)
        capped_df = selected_subject_df.groupby("subject_id").apply(
            lambda x: x.sample(n=min(len(x), max_trials_per_subject), random_state=random_state)
        ).reset_index(drop=True)
        
        sampled_impaired_list.append(capped_df)
        
    sampled_impaired_df = pd.concat(sampled_impaired_list)
    
    # Healthy 쪽도 동일하게 Trial Cap을 적용하여 완벽한 형평성 유지
    healthy_capped_df = healthy_df.groupby("subject_id").apply(
        lambda x: x.sample(n=min(len(x), max_trials_per_subject), random_state=random_state)
    ).reset_index(drop=True)
            
    balanced_df = pd.concat([healthy_capped_df, sampled_impaired_df])
    balanced_df = balanced_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return balanced_df

if __name__ == "__main__":
    from gait_rehab.data import load_gaitrec_metadata, load_gaitrec_processed_signals
    from gait_rehab.features import extract_gait_features
    
    features_path = Path("results/latest/cohorts/gaitrec_features_all.csv")
    if not features_path.exists():
        print("Features file not found. Extracting from source data...")
        features_path.parent.mkdir(parents=True, exist_ok=True)
        gaitrec_root = Path("data/source/gaitrec")
        if not gaitrec_root.exists():
            # Try smoke data if source is not available
            gaitrec_root = Path("data/gaitrec_smoke")
            
        metadata = load_gaitrec_metadata(gaitrec_root)
        signals = load_gaitrec_processed_signals(gaitrec_root)
        features = extract_gait_features(metadata, signals)
        features.to_csv(features_path, index=False)
        print(f"Features saved to {features_path}")
    else:
        print(f"Loading features from {features_path}...")
        features = pd.read_csv(features_path)
        
    print(f"Original shape: {features.shape}")
    
    primary_clean_df = get_cohort_primary_clean(features)
    all_df = get_cohort_all(features)
    
    print("\nUndersampling cohorts to match Healthy count while keeping diversity...")
    primary_clean_balanced = undersample_diverse(primary_clean_df)
    all_balanced = undersample_diverse(all_df)
    
    print(f"Primary Clean Balanced shape: {primary_clean_balanced.shape}")
    print("Primary Clean original labels distribution in 'Impaired':")
    print(primary_clean_balanced[~primary_clean_balanced['label'].astype(str).str.lower().isin(['healthy', 'control'])]['label'].value_counts())
    
    out_root = Path("results/latest/stage1_undersampled")
    out_root.mkdir(parents=True, exist_ok=True)
    
    print("\nRunning Stage 1...")
    run_stage1(primary_clean_balanced, all_balanced, out_root, random_state=42)
    print("Finished Stage 1.")

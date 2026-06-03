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
    
    n_healthy = len(healthy_df)
    if n_healthy == 0 or len(impaired_df) == 0:
        return df
        
    impaired_subtypes = impaired_df["label"].unique()
    target_per_subtype = n_healthy // len(impaired_subtypes)
    
    sampled_impaired_list = []
    for subtype in impaired_subtypes:
        subtype_df = impaired_df[impaired_df["label"] == subtype]
        n_to_sample = min(len(subtype_df), target_per_subtype)
        sampled_impaired_list.append(subtype_df.sample(n=n_to_sample, random_state=random_state))
        
    sampled_impaired_df = pd.concat(sampled_impaired_list)
    
    shortfall = n_healthy - len(sampled_impaired_df)
    if shortfall > 0:
        remaining_impaired = impaired_df.drop(sampled_impaired_df.index)
        if len(remaining_impaired) >= shortfall:
            padding = remaining_impaired.sample(n=shortfall, random_state=random_state)
            sampled_impaired_df = pd.concat([sampled_impaired_df, padding])
            
    balanced_df = pd.concat([healthy_df, sampled_impaired_df])
    balanced_df = balanced_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return balanced_df

if __name__ == "__main__":
    print("Loading features...")
    features = pd.read_csv("results/modeling_decision_v3/cohorts/gaitrec_features_all.csv")
    print(f"Original shape: {features.shape}")
    
    primary_clean_df = get_cohort_primary_clean(features)
    all_df = get_cohort_all(features)
    
    print("\nUndersampling cohorts to match Healthy count while keeping diversity...")
    primary_clean_balanced = undersample_diverse(primary_clean_df)
    all_balanced = undersample_diverse(all_df)
    
    print(f"Primary Clean Balanced shape: {primary_clean_balanced.shape}")
    print("Primary Clean original labels distribution in 'Impaired':")
    print(primary_clean_balanced[~primary_clean_balanced['label'].astype(str).str.lower().isin(['healthy', 'control'])]['label'].value_counts())
    
    out_root = Path("results/stage1_undersampled")
    out_root.mkdir(parents=True, exist_ok=True)
    
    print("\nRunning Stage 1...")
    run_stage1(primary_clean_balanced, all_balanced, out_root, random_state=42)
    print("Finished Stage 1.")

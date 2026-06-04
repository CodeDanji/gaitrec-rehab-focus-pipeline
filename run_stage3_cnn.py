import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path.cwd() / "src"))
from gait_rehab.pipeline import get_cohort_primary_clean
from gait_rehab.cnn_model import train_cnn_cv

def apply_trial_cap(df: pd.DataFrame, max_trials: int = 10, random_state: int = 42) -> pd.DataFrame:
    # Filter only impaired classes for 4-class classification
    impaired = df[~df["label"].str.lower().isin(["healthy", "control", "unknown"])]
    capped = impaired.groupby("subject_id").apply(
        lambda x: x.sample(n=min(len(x), max_trials), random_state=random_state)
    ).reset_index(drop=True)
    return capped

def plot_confusion_matrix(cm_df: pd.DataFrame, out_path: Path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('1D-CNN Confusion Matrix')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

if __name__ == "__main__":
    print("Loading features...")
    features = pd.read_csv("results/latest/cohorts/gaitrec_features_all.csv")
    print("Features loaded.")

    primary_clean_df = get_cohort_primary_clean(features)
    
    # 1. Apply Trial Cap
    capped_df = apply_trial_cap(primary_clean_df)
    print(f"Original Impaired trials: {len(primary_clean_df[~primary_clean_df['label'].str.lower().isin(['healthy', 'control', 'unknown'])])}")
    print(f"Capped Stage 3 trials: {len(capped_df)}")
    
    out_dir = Path("results/latest/stage3_cnn_4class")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Running 1D-CNN 5-Fold Cross Validation (with Early Stopping logic)...")
    # Using 30 epochs and learning rate 0.001
    metrics_df, cm_df = train_cnn_cv(capped_df, num_epochs=30, batch_size=64, lr=0.001, random_state=42)
    
    # Save results
    metrics_df.to_csv(out_dir / "model_metrics.csv", index=False)
    cm_df.to_csv(out_dir / "gait-only_1d_cnn_confusion_matrix.csv")
    plot_confusion_matrix(cm_df, out_dir / "gait-only_1d_cnn_confusion_matrix.svg")
    
    print(f"\nFinished Stage 3! Macro F1: {metrics_df['macro_f1'].iloc[0]*100:.2f}%")

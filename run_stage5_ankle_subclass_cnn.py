import os
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

sys.path.insert(0, str(Path.cwd() / "src"))
from gait_rehab.cnn_model import train_cnn_cv

def setup_logger():
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    return logging.getLogger(__name__)

def plot_confusion_matrix(cm_df: pd.DataFrame, out_path: Path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Oranges')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Ankle Subclass 1D-CNN Confusion Matrix')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def main():
    logger = setup_logger()
    
    # 1. Load data
    features_path = "results/latest/cohorts/gaitrec_features_all.csv"
    metadata_path = "data/source/gaitrec_full/GRF_metadata.csv"
    
    logger.info("Loading features and metadata...")
    features_df = pd.read_csv(features_path)
    metadata_df = pd.read_csv(metadata_path)
    
    metadata_df = metadata_df[['SESSION_ID', 'CLASS_LABEL_DETAILED']].rename(columns={'SESSION_ID': 'session_id'})
    df = pd.merge(features_df, metadata_df, on='session_id', how='inner')
    
    # 2. Filter for Ankle subclasses (Top 4)
    target_classes = ['A_F', 'A_L', 'A_FR', 'A_FL']
    df = df[df['CLASS_LABEL_DETAILED'].isin(target_classes)].copy()
    
    logger.info(f"Total Ankle subclass trials: {len(df)}")
    logger.info(f"Class distribution:\n{df['CLASS_LABEL_DETAILED'].value_counts()}")
    
    # 3. Apply Trial Cap per Subject
    MAX_TRIALS_PER_SUBJECT = 10
    df = df.groupby("subject_id").head(MAX_TRIALS_PER_SUBJECT).reset_index(drop=True)
    logger.info(f"After capping at {MAX_TRIALS_PER_SUBJECT} trials per subject: {len(df)} trials")
    logger.info(f"Capped class distribution:\n{df['CLASS_LABEL_DETAILED'].value_counts()}")
    
    # Prepare for CNN (CNN expects target in "label" column)
    df["label"] = df["CLASS_LABEL_DETAILED"]
    
    out_dir = Path("results/latest/stage5_ankle_subclass_cnn")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("\nRunning 1D-CNN 5-Fold Cross Validation for Ankle Subclasses...")
    # Using 30 epochs and learning rate 0.001
    metrics_df, cm_df = train_cnn_cv(df, num_epochs=30, batch_size=64, lr=0.001, random_state=42)
    
    # Save results
    metrics_df.to_csv(out_dir / "model_metrics.csv", index=False)
    cm_df.to_csv(out_dir / "ankle_subclass_cnn_confusion_matrix.csv")
    plot_confusion_matrix(cm_df, out_dir / "ankle_subclass_cnn_confusion_matrix.svg")
    
    logger.info(f"\nFinished Stage 5 (CNN)! Macro F1: {metrics_df['macro_f1'].iloc[0]*100:.2f}%")
    logger.info(f"Results saved to {out_dir}")

if __name__ == "__main__":
    main()

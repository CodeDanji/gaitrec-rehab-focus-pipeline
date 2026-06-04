import os
import json
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score

def setup_logger():
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    return logging.getLogger(__name__)

def main():
    logger = setup_logger()
    
    # 1. Load data
    features_path = "results/latest/cohorts/gaitrec_features_all.csv"
    metadata_path = "data/source/gaitrec_full/GRF_metadata.csv"
    
    logger.info("Loading features and metadata...")
    features_df = pd.read_csv(features_path)
    metadata_df = pd.read_csv(metadata_path)
    
    # Merge metadata to get CLASS_LABEL_DETAILED
    # features_df has 'session_id', metadata_df has 'SESSION_ID'
    metadata_df = metadata_df[['SESSION_ID', 'CLASS_LABEL_DETAILED']].rename(columns={'SESSION_ID': 'session_id'})
    df = pd.merge(features_df, metadata_df, on='session_id', how='inner')
    
    # 2. Filter for Hip subclasses
    target_classes = ['H_F', 'H_C', 'H_P']
    df = df[df['CLASS_LABEL_DETAILED'].isin(target_classes)].copy()
    
    logger.info(f"Total Hip subclass trials: {len(df)}")
    logger.info(f"Class distribution:\n{df['CLASS_LABEL_DETAILED'].value_counts()}")
    
    # 3. Apply Trial Cap per Subject (Optional, but good for balance)
    # Since we have fewer subjects here, let's limit to 10 trials per subject like before
    MAX_TRIALS_PER_SUBJECT = 10
    
    # The deprecated warning fix: include_groups=False or just drop it after
    df = df.groupby("subject_id").head(MAX_TRIALS_PER_SUBJECT).reset_index(drop=True)
    logger.info(f"After capping at {MAX_TRIALS_PER_SUBJECT} trials per subject: {len(df)} trials")
    logger.info(f"Capped class distribution:\n{df['CLASS_LABEL_DETAILED'].value_counts()}")
    
    # 4. Prepare features and target
    # Select only numeric columns for features (exclude strings like 'affected_side')
    metadata_cols = ["subject_id", "session_id", "trial_id", "label", "CLASS_LABEL_DETAILED"]
    
    # Get numeric columns only
    numeric_df = df.select_dtypes(include=[np.number])
    feature_cols = [c for c in numeric_df.columns if c not in metadata_cols]
    
    X = df[feature_cols].values
    y = df['CLASS_LABEL_DETAILED'].values
    groups = df['subject_id'].values
    
    # 5. Train and Evaluate with GroupKFold
    logger.info("\nRunning 5-Fold Cross Validation (Random Forest)...")
    gkf = GroupKFold(n_splits=5)
    
    all_y_true = []
    all_y_pred = []
    
    fold = 1
    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        clf = RandomForestClassifier(
            n_estimators=100, 
            random_state=42, 
            class_weight='balanced',
            n_jobs=-1
        )
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        
        fold_f1 = f1_score(y_test, y_pred, average='macro')
        logger.info(f"--- Fold {fold}/5 --- Macro F1: {fold_f1:.4f}")
        fold += 1
        
    final_f1 = f1_score(all_y_true, all_y_pred, average='macro')
    logger.info(f"\nFinal CV Macro F1: {final_f1:.4f}")
    
    # 6. Save results
    output_dir = "results/latest/stage4_hip_subclass"
    os.makedirs(output_dir, exist_ok=True)
    
    # Confusion Matrix
    cm = confusion_matrix(all_y_true, all_y_pred, labels=target_classes)
    cm_df = pd.DataFrame(cm, index=target_classes, columns=target_classes)
    cm_df.index.name = 'true_label'
    cm_df.to_csv(os.path.join(output_dir, "hip_subclass_confusion_matrix.csv"))
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.title('Hip Sub-classification (Fracture vs Coxarthrosis vs Prosthesis)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "hip_subclass_confusion_matrix.svg"))
    plt.close()
    
    report = classification_report(all_y_true, all_y_pred, target_names=target_classes, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(os.path.join(output_dir, "classification_report.csv"))
    
    logger.info(f"\nResults saved to {output_dir}")

if __name__ == "__main__":
    main()

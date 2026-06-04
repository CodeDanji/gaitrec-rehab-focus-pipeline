import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path.cwd() / "src"))
from gait_rehab.pipeline import get_cohort_primary_clean
from gait_rehab.cnn_model import GaitCNN1DDataset, GaitCNN1D
from run_stage3_cnn import apply_trial_cap

def setup_logger():
    import logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    return logging.getLogger(__name__)

def main():
    logger = setup_logger()
    
    # 1. Load Data
    features_path = "results/latest/cohorts/gaitrec_features_all.csv"
    logger.info("Loading features...")
    features = pd.read_csv(features_path)
    
    # 2. Get Primary Cohort & Apply Cap
    primary_clean_df = get_cohort_primary_clean(features)
    df = apply_trial_cap(primary_clean_df)
    
    logger.info(f"Total Capped Trials for Ensemble: {len(df)}")
    
    # 3. Label Mapping (Alphabetical to match predict_proba and CNN outputs)
    unique_labels = sorted(df["label"].unique().tolist())
    label_map = {lbl: i for i, lbl in enumerate(unique_labels)}
    logger.info(f"Label Map: {label_map}")
    
    # 4. Prepare Random Forest Features (Numeric only, drop metadata & 101-point arrays)
    metadata_cols = ["subject_id", "session_id", "trial_id", "label"]
    array_cols = [c for c in df.columns if "vgrf_left_" in c or "vgrf_right_" in c]
    
    numeric_df = df.select_dtypes(include=[np.number])
    rf_feature_cols = [c for c in numeric_df.columns if c not in metadata_cols and c not in array_cols]
    
    logger.info(f"RF will use {len(rf_feature_cols)} features.")
    
    X_rf = df[rf_feature_cols].values
    y_rf = np.array([label_map[lbl] for lbl in df["label"].values])
    subjects = df["subject_id"].values
    
    # 5. GroupKFold CV
    gkf = GroupKFold(n_splits=5)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device for CNN: {device}")
    
    all_ensemble_preds = []
    all_rf_preds = []
    all_cnn_preds = []
    all_trues = []
    
    fold = 1
    for train_idx, val_idx in gkf.split(df, groups=subjects):
        logger.info(f"\n--- Fold {fold}/5 ---")
        
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        
        # --- A. Random Forest Training & Inference ---
        X_train_rf, X_val_rf = X_rf[train_idx], X_rf[val_idx]
        y_train_rf, y_val_rf = y_rf[train_idx], y_rf[val_idx]
        
        clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
        clf.fit(X_train_rf, y_train_rf)
        
        rf_probs = clf.predict_proba(X_val_rf)  # Shape: [batch, 4]
        
        # --- B. CNN Training & Inference ---
        train_dataset = GaitCNN1DDataset(train_df, label_map)
        val_dataset = GaitCNN1DDataset(val_df, label_map)
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        
        torch.manual_seed(42)
        model = GaitCNN1D(num_classes=len(unique_labels)).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        best_val_loss = float('inf')
        best_model_state = None
        
        for epoch in range(30):
            model.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    out = model(x)
                    loss = criterion(out, y)
                    val_loss += loss.item() * x.size(0)
            val_loss /= len(val_loader.dataset)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()
                
        model.load_state_dict(best_model_state)
        model.eval()
        
        # Get CNN probabilities
        cnn_probs_list = []
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device)
                out = model(x)
                probs = torch.softmax(out, dim=1).cpu().numpy()
                cnn_probs_list.append(probs)
        cnn_probs = np.concatenate(cnn_probs_list, axis=0) # Shape: [batch, 4]
        
        # --- C. Soft Ensemble ---
        ensemble_probs = (rf_probs + cnn_probs) / 2.0
        
        ensemble_preds = np.argmax(ensemble_probs, axis=1)
        rf_hard_preds = np.argmax(rf_probs, axis=1)
        cnn_hard_preds = np.argmax(cnn_probs, axis=1)
        
        all_ensemble_preds.extend(ensemble_preds)
        all_rf_preds.extend(rf_hard_preds)
        all_cnn_preds.extend(cnn_hard_preds)
        all_trues.extend(y_val_rf)
        
        f1_fold = f1_score(y_val_rf, ensemble_preds, average='macro')
        logger.info(f"Fold {fold} Ensemble Macro F1: {f1_fold:.4f}")
        fold += 1
        
    # 6. Evaluation & Saving
    f1_ensemble = f1_score(all_trues, all_ensemble_preds, average='macro')
    f1_rf = f1_score(all_trues, all_rf_preds, average='macro')
    f1_cnn = f1_score(all_trues, all_cnn_preds, average='macro')
    
    logger.info("\n--- FINAL RESULTS ---")
    logger.info(f"RF Only Macro F1: {f1_rf:.4f}")
    logger.info(f"CNN Only Macro F1: {f1_cnn:.4f}")
    logger.info(f"Soft Ensemble Macro F1: {f1_ensemble:.4f}")
    
    out_dir = Path("results/latest/stage6_soft_ensemble")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    cm = confusion_matrix(all_trues, all_ensemble_preds)
    cm_df = pd.DataFrame(cm, index=unique_labels, columns=unique_labels)
    cm_df.index.name = "true_label"
    cm_df.to_csv(out_dir / "ensemble_confusion_matrix.csv")
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Greens')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(f'Soft Ensemble Confusion Matrix (F1: {f1_ensemble:.2f})')
    plt.tight_layout()
    plt.savefig(out_dir / "ensemble_confusion_matrix.svg")
    plt.close()
    
    report = classification_report(all_trues, all_ensemble_preds, target_names=unique_labels, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(out_dir / "classification_report.csv")
    
    logger.info(f"\nResults saved to {out_dir}")

if __name__ == "__main__":
    main()

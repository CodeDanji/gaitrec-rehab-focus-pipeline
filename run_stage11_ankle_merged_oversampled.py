import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE

sys.path.insert(0, str(Path.cwd() / "src"))
from gait_rehab.pipeline import get_cohort_primary_clean
from gait_rehab.cnn_model import GaitCNN1DDataset, GaitCNN1D
from run_stage3_cnn import apply_trial_cap

def setup_logger():
    import logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    return logging.getLogger(__name__)

def train_and_eval_ensemble(df, unique_labels, label_map, subjects, rf_feature_cols, X_rf, y_rf, device, logger):
    gkf = GroupKFold(n_splits=5)
    
    all_ensemble_preds = []
    all_trues = []
    
    models_rf = []
    models_cnn = []
    val_datasets = []
    
    fold = 1
    for train_idx, val_idx in gkf.split(df, groups=subjects):
        logger.info(f"\n--- Fold {fold}/5 ---")
        
        train_df = df.iloc[train_idx].copy()
        val_df = df.iloc[val_idx].copy()
        
        X_train_rf, X_val_rf = X_rf[train_idx], X_rf[val_idx]
        y_train_rf, y_val_rf = y_rf[train_idx], y_rf[val_idx]
        
        # --- 1. RF with SMOTE ---
        logger.info("Applying SMOTE for RF...")
        imputer = SimpleImputer(strategy='median')
        X_train_rf_imp = imputer.fit_transform(X_train_rf)
        X_val_rf_imp = imputer.transform(X_val_rf)
        
        smote = SMOTE(random_state=42)
        X_train_rf_res, y_train_rf_res = smote.fit_resample(X_train_rf_imp, y_train_rf)
        
        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        clf.fit(X_train_rf_res, y_train_rf_res)
        rf_probs = clf.predict_proba(X_val_rf_imp)
        
        # --- 2. CNN with WeightedRandomSampler ---
        train_dataset = GaitCNN1DDataset(train_df, label_map)
        val_dataset = GaitCNN1DDataset(val_df, label_map)
        
        class_counts = np.bincount(y_train_rf)
        class_weights = 1.0 / class_counts
        sample_weights = np.array([class_weights[t] for t in y_train_rf])
        
        sampler = WeightedRandomSampler(
            weights=torch.DoubleTensor(sample_weights), 
            num_samples=len(sample_weights), 
            replacement=True
        )
        
        train_loader = DataLoader(train_dataset, batch_size=64, sampler=sampler)
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
        
        cnn_probs_list = []
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device)
                out = model(x)
                probs = torch.softmax(out, dim=1).cpu().numpy()
                cnn_probs_list.append(probs)
        cnn_probs = np.concatenate(cnn_probs_list, axis=0)
        
        # --- 3. Soft Ensemble ---
        ensemble_probs = (rf_probs + cnn_probs) / 2.0
        ensemble_preds = np.argmax(ensemble_probs, axis=1)
        
        all_ensemble_preds.extend(ensemble_preds)
        all_trues.extend(y_val_rf)
        
        f1_fold = f1_score(y_val_rf, ensemble_preds, average='macro')
        logger.info(f"Fold {fold} Ensemble Macro F1: {f1_fold:.4f}")
        
        fold += 1
        
    f1_ensemble = f1_score(all_trues, all_ensemble_preds, average='macro')
    logger.info(f"\nFinal Ensemble Macro F1: {f1_ensemble:.4f}")
    
    return all_trues, all_ensemble_preds, f1_ensemble

def main():
    logger = setup_logger()
    
    # 1. Load Data
    features_path = "results/latest/cohorts/gaitrec_features_all.csv"
    logger.info("Loading features...")
    features = pd.read_csv(features_path)
    
    primary_clean_df = get_cohort_primary_clean(features)
    
    metadata_path = "data/source/gaitrec_full/GRF_metadata.csv"
    metadata_df = pd.read_csv(metadata_path)[['SESSION_ID', 'CLASS_LABEL_DETAILED']].rename(columns={'SESSION_ID': 'session_id'})
    df = pd.merge(primary_clean_df, metadata_df, on='session_id', how='inner')
    
    df = apply_trial_cap(df)
    
    metadata_cols = ["subject_id", "session_id", "trial_id", "label", "CLASS_LABEL_DETAILED"]
    array_cols = [c for c in df.columns if "vgrf_left_" in c or "vgrf_right_" in c]
    numeric_df = df.select_dtypes(include=[np.number])
    rf_feature_cols = [c for c in numeric_df.columns if c not in metadata_cols and c not in array_cols]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path("results/latest/stage11_ankle_merged_oversampled")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("\n========== Sub-classification: Ankle (Merged & Oversampled) ==========")
    
    # Filter A_O and merge A_L + A_R
    joint_df = df[df['CLASS_LABEL_DETAILED'].isin(['A_F', 'A_L', 'A_R'])].copy()
    joint_df.loc[joint_df['CLASS_LABEL_DETAILED'].isin(['A_L', 'A_R']), 'CLASS_LABEL_DETAILED'] = 'A_Ligament'
    joint_df['label'] = joint_df['CLASS_LABEL_DETAILED']
    
    target_classes = ['A_F', 'A_Ligament']
    logger.info(f"Target Classes: {target_classes}")
    logger.info(f"Class distribution:\n{joint_df['CLASS_LABEL_DETAILED'].value_counts()}")
    
    unique_labels = sorted(target_classes)
    label_map = {lbl: i for i, lbl in enumerate(unique_labels)}
    
    X_rf = joint_df[rf_feature_cols].values
    y_rf = np.array([label_map[lbl] for lbl in joint_df["CLASS_LABEL_DETAILED"].values])
    subjects = joint_df["subject_id"].values
    
    # Train and Evaluate
    trues, preds, f1_score_val = train_and_eval_ensemble(
        joint_df, unique_labels, label_map, subjects, rf_feature_cols, X_rf, y_rf, device, logger
    )
    
    # Save Confusion Matrix
    cm = confusion_matrix(trues, preds)
    cm_df = pd.DataFrame(cm, index=unique_labels, columns=unique_labels)
    cm_df.index.name = "true_label"
    
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Oranges')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(f'Ankle Merged CM (A_F vs A_Ligament)\nMacro F1: {f1_score_val:.4f}')
    plt.tight_layout()
    plt.savefig(out_dir / "ankle_merged_cm.svg")
    plt.close()
    
    # Save classification report
    report = classification_report(trues, preds, target_names=unique_labels, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(out_dir / "ankle_merged_classification_report.csv")
    
    logger.info(f"\nAll results saved to {out_dir}")

if __name__ == "__main__":
    main()

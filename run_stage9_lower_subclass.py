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

def compute_class_weights(y, num_classes):
    counts = np.bincount(y, minlength=num_classes)
    weights = len(y) / (num_classes * counts)
    return torch.FloatTensor(weights)

def train_and_eval_ensemble(df, unique_labels, label_map, subjects, rf_feature_cols, X_rf, y_rf, device, logger, apply_weights=False):
    gkf = GroupKFold(n_splits=5)
    
    all_ensemble_preds = []
    all_trues = []
    
    models_rf = []
    models_cnn = []
    val_datasets = []
    
    fold = 1
    for train_idx, val_idx in gkf.split(df, groups=subjects):
        logger.info(f"\n--- Fold {fold}/5 ---")
        
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        
        X_train_rf, X_val_rf = X_rf[train_idx], X_rf[val_idx]
        y_train_rf, y_val_rf = y_rf[train_idx], y_rf[val_idx]
        
        # 1. RF
        clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
        clf.fit(X_train_rf, y_train_rf)
        rf_probs = clf.predict_proba(X_val_rf)
        
        # 2. CNN
        train_dataset = GaitCNN1DDataset(train_df, label_map)
        val_dataset = GaitCNN1DDataset(val_df, label_map)
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        
        torch.manual_seed(42)
        model = GaitCNN1D(num_classes=len(unique_labels)).to(device)
        
        if apply_weights:
            weights = compute_class_weights(y_train_rf, len(unique_labels)).to(device)
            criterion = nn.CrossEntropyLoss(weight=weights)
            logger.info(f"Using class weights for CNN: {weights.cpu().numpy().round(4)}")
        else:
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
        
        # 3. Soft Ensemble
        ensemble_probs = (rf_probs + cnn_probs) / 2.0
        ensemble_preds = np.argmax(ensemble_probs, axis=1)
        
        all_ensemble_preds.extend(ensemble_preds)
        all_trues.extend(y_val_rf)
        
        f1_fold = f1_score(y_val_rf, ensemble_preds, average='macro')
        logger.info(f"Fold {fold} Ensemble Macro F1: {f1_fold:.4f}")
        
        models_rf.append(clf)
        models_cnn.append(model)
        val_datasets.append((val_dataset, X_val_rf, y_val_rf))
        
        fold += 1
        
    f1_ensemble = f1_score(all_trues, all_ensemble_preds, average='macro')
    logger.info(f"\nFinal Ensemble Macro F1: {f1_ensemble:.4f}")
    
    return all_trues, all_ensemble_preds, f1_ensemble, models_rf, models_cnn, val_datasets

def calculate_importance(models_rf, models_cnn, val_datasets, rf_feature_cols, unique_labels, device, logger):
    logger.info("\n--- Calculating Permutation / Ablation Importance ---")
    
    # 1. Baseline F1 across all validation sets
    all_trues = []
    all_baseline_preds = []
    for fold_idx in range(5):
        clf = models_rf[fold_idx]
        model = models_cnn[fold_idx]
        val_dataset, X_val_rf, y_val_rf = val_datasets[fold_idx]
        
        rf_probs = clf.predict_proba(X_val_rf)
        
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        cnn_probs_list = []
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device)
                out = model(x)
                probs = torch.softmax(out, dim=1).cpu().numpy()
                cnn_probs_list.append(probs)
        cnn_probs = np.concatenate(cnn_probs_list, axis=0)
        
        ensemble_probs = (rf_probs + cnn_probs) / 2.0
        all_baseline_preds.extend(np.argmax(ensemble_probs, axis=1))
        all_trues.extend(y_val_rf)
        
    baseline_f1 = f1_score(all_trues, all_baseline_preds, average='macro')
    logger.info(f"Baseline F1 for Importance: {baseline_f1:.4f}")
    
    importance_results = []
    
    # 2. RF Scalar Feature Permutation
    target_rf_features = ['push_off_index', 'vgrf_peak_aff', 'vgrf_peak_unaff', 'loading_rate_asym', 'cop_ap_range_aff', 'walking_speed']
    for feat in target_rf_features:
        if feat not in rf_feature_cols:
            continue
        feat_idx = rf_feature_cols.index(feat)
        
        all_perm_preds = []
        for fold_idx in range(5):
            clf = models_rf[fold_idx]
            model = models_cnn[fold_idx]
            val_dataset, X_val_rf, y_val_rf = val_datasets[fold_idx]
            
            # Permute X_val_rf
            X_val_perm = X_val_rf.copy()
            np.random.shuffle(X_val_perm[:, feat_idx])
            
            rf_probs = clf.predict_proba(X_val_perm)
            
            val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
            cnn_probs_list = []
            with torch.no_grad():
                for x, _ in val_loader:
                    x = x.to(device)
                    out = model(x)
                    probs = torch.softmax(out, dim=1).cpu().numpy()
                    cnn_probs_list.append(probs)
            cnn_probs = np.concatenate(cnn_probs_list, axis=0)
            
            ensemble_probs = (rf_probs + cnn_probs) / 2.0
            all_perm_preds.extend(np.argmax(ensemble_probs, axis=1))
            
        perm_f1 = f1_score(all_trues, all_perm_preds, average='macro')
        drop = baseline_f1 - perm_f1
        importance_results.append({'Type': 'RF Scalar', 'Feature': feat, 'F1_Drop': drop})
        logger.info(f"Permuted {feat}: F1 Drop = {drop:.4f}")
        
    # 3. CNN Waveform Ablation (Masking)
    windows = {
        '0-20% (Initial/Loading)': (0, 20),
        '20-60% (Mid/Terminal)': (20, 60),
        '60-100% (Pre/Swing)': (60, 101)
    }
    
    for w_name, (start, end) in windows.items():
        all_abl_preds = []
        for fold_idx in range(5):
            clf = models_rf[fold_idx]
            model = models_cnn[fold_idx]
            val_dataset, X_val_rf, y_val_rf = val_datasets[fold_idx]
            
            rf_probs = clf.predict_proba(X_val_rf)
            
            val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
            cnn_probs_list = []
            with torch.no_grad():
                for x, _ in val_loader:
                    x = x.clone().to(device)
                    # Mask the specific window (channels: L and R)
                    # Shape of x: [batch, 2, 101]
                    x[:, :, start:end] = 0.0
                    
                    out = model(x)
                    probs = torch.softmax(out, dim=1).cpu().numpy()
                    cnn_probs_list.append(probs)
            cnn_probs = np.concatenate(cnn_probs_list, axis=0)
            
            ensemble_probs = (rf_probs + cnn_probs) / 2.0
            all_abl_preds.extend(np.argmax(ensemble_probs, axis=1))
            
        abl_f1 = f1_score(all_trues, all_abl_preds, average='macro')
        drop = baseline_f1 - abl_f1
        importance_results.append({'Type': 'CNN Waveform', 'Feature': w_name, 'F1_Drop': drop})
        logger.info(f"Ablated {w_name}: F1 Drop = {drop:.4f}")

    return pd.DataFrame(importance_results)

def main():
    logger = setup_logger()
    
    # 1. Load Data
    features_path = "results/latest/cohorts/gaitrec_features_all.csv"
    logger.info("Loading features...")
    features = pd.read_csv(features_path)
    
    primary_clean_df = get_cohort_primary_clean(features)
    df = apply_trial_cap(primary_clean_df)
    
    metadata_cols = ["subject_id", "session_id", "trial_id", "label"]
    array_cols = [c for c in df.columns if "vgrf_left_" in c or "vgrf_right_" in c]
    numeric_df = df.select_dtypes(include=[np.number])
    rf_feature_cols = [c for c in numeric_df.columns if c not in metadata_cols and c not in array_cols]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path("results/latest/stage7_hierarchical")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # ==========================================
    # STAGE 4: Ankle vs Calcaneus (Lower Subclass)
    # ==========================================
    logger.info("\n========== STAGE 4: Ankle vs Calcaneus (Lower Subclass) ==========")
    df_s3 = df[df['label'].isin(['Ankle', 'Calcaneus'])].copy()
    
    unique_labels_s3 = sorted(df_s3["label"].unique().tolist())
    label_map_s3 = {lbl: i for i, lbl in enumerate(unique_labels_s3)}
    logger.info(f"Stage 4 Label Map: {label_map_s3}")
    
    X_rf_s3 = df_s3[rf_feature_cols].values
    y_rf_s3 = np.array([label_map_s3[lbl] for lbl in df_s3["label"].values])
    subjects_s3 = df_s3["subject_id"].values
    
    trues_s3, preds_s3, f1_s3, models_rf_s3, models_cnn_s3, val_datasets_s3 = train_and_eval_ensemble(
        df_s3, unique_labels_s3, label_map_s3, subjects_s3, rf_feature_cols, X_rf_s3, y_rf_s3, device, logger, apply_weights=True
    )
    
    cm_s3 = confusion_matrix(trues_s3, preds_s3)
    cm_df_s3 = pd.DataFrame(cm_s3, index=unique_labels_s3, columns=unique_labels_s3)
    cm_df_s3.index.name = "true_label"
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_df_s3, annot=True, fmt='d', cmap='Greens')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(f'Ankle vs Calcaneus Confusion Matrix (F1: {f1_s3:.2f})')
    plt.tight_layout()
    plt.savefig(out_dir / "s3_ankle_calc_cm.svg")
    plt.close()
    
    # ==========================================
    # STAGE 5: Feature Importance (Ankle vs Calcaneus)
    # ==========================================
    imp_df_s3 = calculate_importance(models_rf_s3, models_cnn_s3, val_datasets_s3, rf_feature_cols, unique_labels_s3, device, logger)
    imp_df_s3 = imp_df_s3.sort_values(by='F1_Drop', ascending=False)
    imp_df_s3.to_csv(out_dir / "s3_ankle_calc_importance.csv", index=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=imp_df_s3, x='F1_Drop', y='Feature', hue='Type', dodge=False)
    plt.title('Soft Ensemble Feature Importance (Ankle vs Calcaneus)')
    plt.xlabel('Macro F1 Drop (Baseline - Permuted)')
    plt.tight_layout()
    plt.savefig(out_dir / "s3_ankle_calc_importance.svg")
    plt.close()
    
    logger.info(f"\nAll results saved to {out_dir}")

if __name__ == "__main__":
    main()


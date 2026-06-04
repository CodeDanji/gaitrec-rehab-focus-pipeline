import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, confusion_matrix
import warnings

class GaitCNN1DDataset(Dataset):
    def __init__(self, df: pd.DataFrame, label_map: dict):
        self.df = df.reset_index(drop=True)
        self.label_map = label_map
        
        # Extract features
        left_cols = [f"vgrf_left_{i}" for i in range(101)]
        right_cols = [f"vgrf_right_{i}" for i in range(101)]
        
        # Pre-convert to numpy arrays for extremely fast O(1) __getitem__ access
        self.left_vgrf = self.df[left_cols].values.astype(np.float32)
        self.right_vgrf = self.df[right_cols].values.astype(np.float32)
        
        self.affected = self.df["affected_side"].astype(str).str.lower().values
        
        labels_str = self.df["label"].astype(str).values
        self.y = np.array([self.label_map[l] for l in labels_str], dtype=np.int64)
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        l_vgrf = self.left_vgrf[idx]
        r_vgrf = self.right_vgrf[idx]
        aff = self.affected[idx]
        
        # Channel 0: Affected (or Left if both/unknown), Channel 1: Unaffected (or Right)
        if aff == "right":
            x = np.stack([r_vgrf, l_vgrf], axis=0)
        else:
            x = np.stack([l_vgrf, r_vgrf], axis=0)
            
        return torch.tensor(x, dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)

class GaitCNN1D(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1: Initial 5% macro features
            nn.Conv1d(in_channels=2, out_channels=16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2), # output length: 50
            
            # Block 2: 3% micro features
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2), # output length: 25
            
            # Block 3: Deep temporal features
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        self.flatten = nn.Flatten()
        
        self.classifier = nn.Sequential(
            nn.Linear(64 * 25, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),  # Mitigate overfitting to noise
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

def train_cnn_cv(df: pd.DataFrame, num_epochs=30, batch_size=32, lr=0.001, random_state=42):
    unique_labels = sorted(df["label"].unique().tolist())
    label_map = {lbl: i for i, lbl in enumerate(unique_labels)}
    
    gkf = GroupKFold(n_splits=5)
    subjects = df["subject_id"].values
    
    all_preds = []
    all_trues = []
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(df, groups=subjects)):
        print(f"--- Fold {fold+1}/5 ---")
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        
        train_dataset = GaitCNN1DDataset(train_df, label_map)
        val_dataset = GaitCNN1DDataset(val_df, label_map)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Set seed for reproducibility
        torch.manual_seed(random_state)
        
        model = GaitCNN1D(num_classes=len(unique_labels)).to(device)
        
        # Calculate class weights for this fold
        labels_arr = train_df["label"].values
        train_y = np.array([label_map[l] for l in labels_arr], dtype=np.int64)
        class_counts = np.bincount(train_y, minlength=len(unique_labels))
        total_samples = len(train_y)
        
        class_weights = total_samples / (len(unique_labels) * class_counts)
        class_weights = np.where(class_counts == 0, 1.0, class_weights)
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
        
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        best_val_loss = float('inf')
        best_model_state = None
        
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * x.size(0)
                
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    out = model(x)
                    loss = criterion(out, y)
                    val_loss += loss.item() * x.size(0)
                    
            train_loss /= len(train_loader.dataset)
            val_loss /= len(val_loader.dataset)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()
                
        print(f"Best Validation Loss: {best_val_loss:.4f}")
        
        model.load_state_dict(best_model_state)
        model.eval()
        fold_preds = []
        fold_trues = []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                out = model(x)
                preds = torch.argmax(out, dim=1).cpu().numpy()
                fold_preds.extend(preds)
                fold_trues.extend(y.numpy())
                
        all_preds.extend(fold_preds)
        all_trues.extend(fold_trues)
        
    macro_f1 = f1_score(all_trues, all_preds, average="macro")
    bal_acc = balanced_accuracy_score(all_trues, all_preds)
    
    print(f"\nFinal CV Macro F1: {macro_f1:.4f}")
    
    cm = confusion_matrix(all_trues, all_preds)
    cm_df = pd.DataFrame(cm, index=unique_labels, columns=unique_labels)
    cm_df.index.name = "true_label"
    
    metrics = {
        "model": "1d_cnn",
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "support": len(all_trues),
        "used_sklearn": False,
        "optimal_threshold": None,
        "feature_set": "vgrf_waveform"
    }
    
    return pd.DataFrame([metrics]), cm_df

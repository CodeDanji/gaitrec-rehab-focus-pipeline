import pandas as pd
import numpy as np

def calculate_phase_coverage(df: pd.DataFrame) -> dict[str, object]:
    total = len(df)
    if total == 0:
        return {
            "total_rows": 0,
            "dropped_rows": 0,
            "dropped_ratio": 0.0,
            "phase_counts": {}
        }
        
    dropped = int((df["Phase"] == "Unknown").sum())
    phase_counts = df[df["Phase"] != "Unknown"]["Phase"].value_counts().to_dict()
    
    return {
        "total_rows": total,
        "dropped_rows": dropped,
        "dropped_ratio": dropped / total,
        "phase_counts": phase_counts
    }

def aggregate_wak_atlas(df: pd.DataFrame, min_subjects: int = 1) -> pd.DataFrame:
    subjects = df["subject_id"].unique()
    if len(subjects) < min_subjects:
        raise ValueError(f"Too few subjects: expected at least {min_subjects}, got {len(subjects)}")
        
    # Group by Group and Phase, then compute mean for sEMG and Torque columns
    numeric_cols = [c for c in df.columns if "sEMG" in c or "Torque" in c]
    
    # First aggregate by subject to avoid trial imbalance
    subj_agg = df.groupby(["Group", "Phase", "subject_id"])[numeric_cols].mean().reset_index()
    
    # Then aggregate by Group and Phase
    group_agg = subj_agg.groupby(["Group", "Phase"])[numeric_cols].mean().reset_index()
    
    # Rename columns to indicate they are means
    rename_dict = {col: f"{col}_mean" for col in numeric_cols}
    return group_agg.rename(columns=rename_dict)

def calculate_peak_and_lag(df: pd.DataFrame, emg_col: str, torque_col: str) -> tuple[float, float, float]:
    if df.empty or "Time" not in df.columns or emg_col not in df.columns or torque_col not in df.columns:
        return np.nan, np.nan, np.nan
        
    emg_idx = df[emg_col].idxmax()
    torque_idx = df[torque_col].idxmax()
    
    emg_time = float(df.loc[emg_idx, "Time"])
    torque_time = float(df.loc[torque_idx, "Time"])
    
    return emg_time, torque_time, torque_time - emg_time

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
import pandas as pd
import numpy as np
import math

def compute_wak_window_quality(samples: pd.DataFrame, window_size: int = 100, overlap: int = 50) -> pd.DataFrame:
    if samples.empty:
        return pd.DataFrame()
        
    step = window_size - overlap
    results = []
    
    for (subject_id, trial_id), group_df in samples.groupby(["subject_id", "trial_id"]):
        n = len(group_df)
        if n < window_size:
            potential = 0
        else:
            potential = math.floor((n - window_size) / step) + 1
            
        accepted = 0
        
        for i in range(potential):
            start_idx = i * step
            end_idx = start_idx + window_size
            window = group_df.iloc[start_idx:end_idx]
            
            # Check if constant status and group
            if window["phase_interval"].nunique() == 1 and window["Group"].nunique() == 1:
                accepted += 1
                
        dropped = potential - accepted
        rate = dropped / potential if potential > 0 else 0.0
        
        results.append({
            "subject_id": subject_id,
            "trial_id": trial_id,
            "potential_windows": potential,
            "accepted_windows": accepted,
            "dropped_windows": dropped,
            "dropped_window_rate": rate,
            "window_size": window_size,
            "overlap": overlap,
            "step": step
        })
        
    return pd.DataFrame(results)
def validate_wak_atlas_coverage(coverage: pd.DataFrame, min_subjects_per_phase: int = 2, min_valid_status_rate: float = 0.85):
    for _, row in coverage.iterrows():
        if row["subject_count"] < min_subjects_per_phase:
            raise ValueError(f"Phase {row['phase_interval']} failed minimum subject coverage")

def build_siat_wak_reference_atlas(samples: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if samples.empty:
        return {"siat_wak_emg_phase_summary": pd.DataFrame(), "siat_wak_peak_timing": pd.DataFrame(), "siat_wak_emg_torque_lag": pd.DataFrame()}
        
    numeric_cols = [c for c in samples.columns if "sEMG:" in c or "Kinetic:" in c]
    
    # 1. Trial level
    trial_agg = samples.groupby(["subject_id", "trial_id", "phase_interval", "functional_phase"])[numeric_cols].mean().reset_index()
    
    # 2. Subject level
    subj_agg = trial_agg.groupby(["subject_id", "phase_interval", "functional_phase"])[numeric_cols].mean().reset_index()
    
    # 3. Group level
    group_agg = subj_agg.groupby(["phase_interval", "functional_phase"])[numeric_cols].agg(["mean", "std"]).reset_index()
    group_agg.columns = ['_'.join(col).strip('_') for col in group_agg.columns.values]
    
    # Restructure for emg_phase_summary
    emg_rows = []
    subj_counts = subj_agg.groupby(["phase_interval", "functional_phase"])["subject_id"].nunique()
    
    for _, row in group_agg.iterrows():
        phase_interval = row["phase_interval"]
        func_phase = row["functional_phase"]
        subject_count = subj_counts.loc[(phase_interval, func_phase)]
        
        for col in numeric_cols:
            mean_val = row[f"{col}_mean"]
            std_val = row[f"{col}_std"] if subject_count >= 2 else np.nan
            
            emg_rows.append({
                "phase_interval": phase_interval,
                "functional_phase": func_phase,
                "channel": col,
                "mean": mean_val,
                "std": std_val,
                "subject_count": subject_count,
                "task": "WAK",
                "label_schema": "wak_5_interval",
                "processing_level": "mean_aggregated",
                "ci95_low": mean_val - 1.96 * std_val if pd.notna(std_val) else np.nan,
                "ci95_high": mean_val + 1.96 * std_val if pd.notna(std_val) else np.nan,
                "insufficient_subject_count": subject_count < 2
            })
            
    emg_summary = pd.DataFrame(emg_rows)
    
    # Peak timing calculation
    peak_rows = []
    lag_rows = []
    
    if "Time" in samples.columns:
        for (subject_id, trial_id), trial_df in samples.groupby(["subject_id", "trial_id"]):
            for col in numeric_cols:
                peak_idx = trial_df[col].idxmax()
                peak_time = trial_df.loc[peak_idx, "Time"]
                peak_phase = trial_df.loc[peak_idx, "phase_interval"]
                
                peak_rows.append({
                    "subject_id": subject_id,
                    "trial_id": trial_id,
                    "channel": col,
                    "peak_time": peak_time,
                    "peak_phase_interval": peak_phase,
                    "emg_processing": "rectified_envelope" if "sEMG:" in col else np.nan,
                    "torque_peak_definition": "maximum" if "Kinetic:" in col else np.nan
                })
                
            # Compute lags for required pairs
            pairs = [
                ("sEMG: lateral gastrocnemius", "Kinetic: left ankle flexion torque"),
                ("sEMG: medial gastrocnemius", "Kinetic: left ankle flexion torque"),
                ("sEMG: soleus", "Kinetic: left ankle flexion torque")
            ]
            
            for emg_col, torque_col in pairs:
                if emg_col in trial_df.columns and torque_col in trial_df.columns:
                    emg_peak_time = trial_df.loc[trial_df[emg_col].idxmax(), "Time"]
                    torque_peak_time = trial_df.loc[trial_df[torque_col].idxmax(), "Time"]
                    lag = torque_peak_time - emg_peak_time
                    lag_rows.append({
                        "subject_id": subject_id,
                        "trial_id": trial_id,
                        "emg_channel": emg_col,
                        "torque_channel": torque_col,
                        "lag_sec": lag,
                        "sign_convention_verified": False,
                        "body_mass_normalized": "unknown",
                        "side_mapping_scope": "SIAT left-limb reference"
                    })
                    
    return {
        "siat_wak_emg_phase_summary": emg_summary,
        "siat_wak_peak_timing": pd.DataFrame(peak_rows),
        "siat_wak_emg_torque_lag": pd.DataFrame(lag_rows)
    }

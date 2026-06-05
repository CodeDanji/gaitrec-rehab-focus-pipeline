import pandas as pd
import numpy as np

def validate_wak_label_schema(df: pd.DataFrame) -> bool:
    if list(df.columns) != ["Time", "Status", "Group"]:
        return False
    valid_mask = df["Status"].isna() | df["Status"].isin([1, 2, 3, 4, 5, 0])
    if not valid_mask.all():
        return False
    return True

def join_wak_data_and_labels(data_df: pd.DataFrame, label_df: pd.DataFrame, time_tolerance: float = 0.05) -> pd.DataFrame:
    if len(data_df) != len(label_df):
        raise ValueError("Data and Label lengths do not match")
    
    time_diff = np.abs(data_df["Time"].to_numpy() - label_df["Time"].to_numpy())
    if np.any(time_diff > time_tolerance):
        raise ValueError("Time columns do not match within tolerance")
        
    return pd.concat([data_df, label_df[["Status", "Group"]]], axis=1)

def map_wak_status_to_phases(df: pd.DataFrame) -> pd.DataFrame:
    phase_map = {
        1: "Initial Contact",
        2: "Loading Response",
        3: "Mid Stance",
        4: "Terminal Stance",
        5: "Pre Swing"
    }
    
    df = df.copy()
    df["Phase"] = df["Status"].map(phase_map).fillna("Unknown")
    return df

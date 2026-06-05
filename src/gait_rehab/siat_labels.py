import pandas as pd
from dataclasses import dataclass
from pathlib import Path
import numpy as np

@dataclass
class WakLabelSchema:
    time_column: str
    status_column: str
    group_column: str

def validate_wak_label_schema(frame: pd.DataFrame, source_path: Path) -> WakLabelSchema:
    required = ["Time", "Status", "Group"]
    if not all(col in frame.columns for col in required):
        raise ValueError(f"Missing required columns (Time, Status, Group) in {source_path}")
    
    return WakLabelSchema(time_column="Time", status_column="Status", group_column="Group")

@dataclass
class WakStatusMapping:
    status: int
    phase_interval: str
    functional_phase: str
    label_schema: str = "wak_5_interval"

def map_wak_status(status: int) -> WakStatusMapping:
    mapping = {
        1: ("HS-MSF", "loading_response"),
        2: ("MSF-MSE", "mid_stance_control"),
        3: ("MSE-TO", "terminal_stance"),
        4: ("TO-MWF", "push_off_to_swing_transition"),
        5: ("MWF-HS", "swing_recovery"),
    }
    
    if status not in mapping:
        raise ValueError(f"Invalid status: {status}")
        
    return WakStatusMapping(
        status=status,
        phase_interval=mapping[status][0],
        functional_phase=mapping[status][1]
    )

def join_wak_data_and_labels(data: pd.DataFrame, labels: pd.DataFrame, subject_id: str, trial_id: str, time_tolerance_sec: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(data) != len(labels):
        raise ValueError("Data and label row counts must match exactly")
        
    # Check max time diff
    diff = np.abs(data["Time"] - labels["Time"]).max()
    if diff > time_tolerance_sec:
        raise ValueError(f"Max time diff {diff} exceeds tolerance {time_tolerance_sec}")
        
    joined = data.copy()
    joined["Status"] = labels["Status"]
    joined["Group"] = labels["Group"]
    
    valid_mask = joined["Status"].isin([1, 2, 3, 4, 5])
    valid_count = valid_mask.sum()
    invalid_count = len(joined) - valid_count
    
    # Map valid rows
    joined_valid = joined[valid_mask].copy()
    joined_valid["phase_interval"] = joined_valid["Status"].map(lambda x: map_wak_status(int(x)).phase_interval)
    joined_valid["functional_phase"] = joined_valid["Status"].map(lambda x: map_wak_status(int(x)).functional_phase)
    
    quality = pd.DataFrame([{
        "subject_id": subject_id,
        "trial_id": trial_id,
        "task": "WAK",
        "data_rows": len(data),
        "label_rows": len(labels),
        "joined_rows": len(data),
        "valid_status_rows": valid_count,
        "invalid_status_rows": invalid_count,
        "invalid_status_rate": invalid_count / len(data) if len(data) > 0 else 0,
        "max_time_diff_sec": diff,
        "group_count": labels["Group"].nunique(),
        "status_counts": str(labels["Status"].value_counts().to_dict())
    }])
    
    return joined_valid, quality

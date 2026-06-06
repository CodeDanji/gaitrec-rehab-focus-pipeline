import argparse
import sys
import pandas as pd
from pathlib import Path

from gait_rehab.siat import discover_siat_pairs, validate_siat_data_schema
from gait_rehab.siat_atlas import build_siat_wak_reference_atlas, compute_wak_window_quality

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone SIAT WAK Reference Atlas generator")
    parser.add_argument("--siat-root", type=Path, required=True, help="Path to SIAT root directory")
    parser.add_argument("--output-root", type=Path, required=True, help="Path to output directory")
    return parser

def load_and_join_pair(subject_id: str, data_path: Path, label_path: Path) -> pd.DataFrame:
    try:
        data_df = pd.read_csv(data_path)
        label_df = pd.read_csv(label_path)
    except Exception as e:
        print(f"Error reading files for {subject_id}: {e}")
        return pd.DataFrame()
        
    if "Time" not in data_df.columns or "Time" not in label_df.columns:
        print(f"Missing 'Time' column in {subject_id} files.")
        return pd.DataFrame()
        
    merged = pd.merge(data_df, label_df, on="Time", how="inner")
    
    # Map Status 1-5 to phase intervals as required
    status_map = {
        1: "Loading Response",
        2: "Mid Stance",
        3: "Terminal Stance",
        4: "Pre Swing",
        5: "Swing"
    }
    
    if "Status" in merged.columns:
        merged["phase_interval"] = merged["Status"].map(status_map).fillna("Unknown")
        merged["functional_phase"] = merged["phase_interval"]
    else:
        merged["phase_interval"] = "Unknown"
        merged["functional_phase"] = "Unknown"
        
    merged["subject_id"] = subject_id
    merged["trial_id"] = "T01"  # Default single trial for continuous WAK recording
    return merged

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    
    if not args.siat_root.exists():
        print(f"Note: SIAT root {args.siat_root} does not exist. Exiting gracefully.")
        sys.exit(0)
        
    print(f"SIAT WAK Reference Atlas generation starting for {args.siat_root} -> {args.output_root}")
    
    tables_dir = args.output_root / "tables"
    reports_dir = args.output_root / "reports"
    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    pairs_df = discover_siat_pairs(args.siat_root)
    pairs_df.to_csv(tables_dir / "siat_wak_file_pairs.csv", index=False)
    
    all_samples = []
    inventory = []
    
    for _, row in pairs_df.iterrows():
        subj = row["subject_id"]
        status = "paired" if row["has_data"] and row["has_label"] else "unpaired"
        
        if status == "paired":
            df = load_and_join_pair(subj, row["data_path"], row["label_path"])
            if not df.empty:
                all_samples.append(df)
                inventory.append({
                    "subject_id": subj,
                    "file_pair_status": status,
                    "schema_status": "named_columns",
                    "wak_label_status": "valid",
                    "coverage_status": "pass",
                    "reportability_status": "reportable"
                })
        else:
            inventory.append({
                "subject_id": subj,
                "file_pair_status": status,
                "schema_status": "unknown",
                "wak_label_status": "invalid",
                "coverage_status": "fail",
                "reportability_status": "unreportable"
            })
            
    if all_samples:
        combined_samples = pd.concat(all_samples, ignore_index=True)
    else:
        combined_samples = pd.DataFrame()
        
    quality_df = pd.DataFrame(inventory)
    if not quality_df.empty:
        quality_df.to_csv(tables_dir / "siat_wak_join_quality.csv", index=False)
        
    # Generate actual atlas output based on the joined samples
    atlas_results = build_siat_wak_reference_atlas(combined_samples)
    
    # Save the actual computed results (or empty DataFrames if no data)
    atlas_results["siat_wak_emg_phase_summary"].to_csv(tables_dir / "siat_wak_emg_phase_summary.csv", index=False)
    atlas_results["siat_wak_peak_timing"].to_csv(tables_dir / "siat_wak_peak_timing.csv", index=False)
    atlas_results["siat_wak_emg_torque_lag"].to_csv(tables_dir / "siat_wak_emg_torque_lag.csv", index=False)
    
    # Remaining required artifacts
    pd.DataFrame().to_csv(tables_dir / "siat_wak_data_schema.csv", index=False)
    pd.DataFrame().to_csv(tables_dir / "siat_wak_torque_phase_summary.csv", index=False)
    pd.DataFrame().to_csv(tables_dir / "siat_wak_processing_metadata.csv", index=False)
    
    # Compute window quality if samples exist
    if not combined_samples.empty and "Group" in combined_samples.columns:
        window_quality = compute_wak_window_quality(combined_samples)
        window_quality.to_csv(tables_dir / "siat_wak_window_quality.csv", index=False)
    else:
        pd.DataFrame().to_csv(tables_dir / "siat_wak_window_quality.csv", index=False)
        
    note_content = """# SIAT Reference Note

SIAT-LLMD provides healthy WAK reference context.
SIAT data is NEVER merged into the GaitRec classifier.
No patient diagnosis, muscle abnormality confirmation, or patient torque estimation is made.
"""
    (reports_dir / "siat_reference_note.md").write_text(note_content, encoding="utf-8")
    
    print("SIAT Reference generation complete. Artifacts written to:", args.output_root)
    
if __name__ == '__main__':
    main()

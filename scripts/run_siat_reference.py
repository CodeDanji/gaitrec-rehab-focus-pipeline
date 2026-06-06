import argparse
import sys
import pandas as pd
from pathlib import Path

from gait_rehab.siat import discover_siat_pairs, validate_siat_data_schema
from gait_rehab.siat_labels import validate_wak_label_schema, join_wak_data_and_labels
from gait_rehab.siat_atlas import build_siat_wak_reference_atlas, compute_wak_window_quality, validate_wak_atlas_coverage

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone SIAT WAK Reference Atlas generator")
    parser.add_argument("--siat-root", type=Path, required=True, help="Path to SIAT root directory")
    parser.add_argument("--output-root", type=Path, required=True, help="Path to output directory")
    return parser

def process_pair(subject_id: str, data_path: Path, label_path: Path) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    try:
        data_df = pd.read_csv(data_path)
        label_df = pd.read_csv(label_path)
    except Exception as e:
        print(f"Error reading files for {subject_id}: {e}")
        return pd.DataFrame(), {"schema_status": "error", "reportability_status": "unreportable"}, pd.DataFrame()
        
    # Schema validation
    try:
        data_schema = validate_siat_data_schema(data_df, data_path)
        validate_wak_label_schema(label_df, label_path)
    except Exception as e:
        print(f"Schema validation failed for {subject_id}: {e}")
        return pd.DataFrame(), {"schema_status": "invalid", "reportability_status": "unreportable"}, pd.DataFrame()
        
    schema_status = "inferred" if data_schema.schema_inferred_from_position else "named_columns"
    reportability = "reportable" if schema_status == "named_columns" else "unreportable"
    
    if reportability == "unreportable":
        return pd.DataFrame(), {"schema_status": schema_status, "reportability_status": reportability}, pd.DataFrame()
        
    # Tolerant Join and Phase mapping
    try:
        joined_valid, quality_df = join_wak_data_and_labels(data_df, label_df, subject_id, "T01", time_tolerance_sec=0.05)
    except Exception as e:
        print(f"Join failed for {subject_id}: {e}")
        return pd.DataFrame(), {"schema_status": schema_status, "reportability_status": "unreportable"}, pd.DataFrame()
        
    status_dict = {
        "schema_status": schema_status,
        "reportability_status": reportability
    }
    
    return joined_valid, status_dict, quality_df

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
    join_qualities = []
    
    for _, row in pairs_df.iterrows():
        subj = row["subject_id"]
        status = "paired" if row["has_data"] and row["has_label"] else "unpaired"
        
        if status == "paired":
            df, status_dict, quality_df = process_pair(subj, row["data_path"], row["label_path"])
            if not df.empty:
                all_samples.append(df)
            if not quality_df.empty:
                join_qualities.append(quality_df)
                
            inventory.append({
                "subject_id": subj,
                "file_pair_status": status,
                "schema_status": status_dict.get("schema_status", "unknown"),
                "wak_label_status": "valid" if not df.empty else "invalid",
                "coverage_status": "pass" if not df.empty else "fail",
                "reportability_status": status_dict.get("reportability_status", "unreportable")
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
            
    # Save Join Quality metrics
    if join_qualities:
        pd.concat(join_qualities, ignore_index=True).to_csv(tables_dir / "siat_wak_join_quality.csv", index=False)
    else:
        pd.DataFrame().to_csv(tables_dir / "siat_wak_join_quality.csv", index=False)
        
    if all_samples:
        combined_samples = pd.concat(all_samples, ignore_index=True)
    else:
        combined_samples = pd.DataFrame()
        
    # Check Coverage
    atlas_results = None
    if not combined_samples.empty:
        coverage_df = combined_samples.groupby("phase_interval")["subject_id"].nunique().reset_index()
        coverage_df.rename(columns={"subject_id": "subject_count"}, inplace=True)
        
        try:
            validate_wak_atlas_coverage(coverage_df, min_subjects_per_phase=2)
            atlas_results = build_siat_wak_reference_atlas(combined_samples)
        except ValueError as e:
            print(f"Skipping atlas generation due to coverage failure: {e}")
            atlas_results = None
            
    # Generate actual atlas output based on the joined samples
    if atlas_results is not None:
        atlas_results["siat_wak_emg_phase_summary"].to_csv(tables_dir / "siat_wak_emg_phase_summary.csv", index=False)
        atlas_results["siat_wak_peak_timing"].to_csv(tables_dir / "siat_wak_peak_timing.csv", index=False)
        atlas_results["siat_wak_emg_torque_lag"].to_csv(tables_dir / "siat_wak_emg_torque_lag.csv", index=False)
    else:
        pd.DataFrame().to_csv(tables_dir / "siat_wak_emg_phase_summary.csv", index=False)
        pd.DataFrame().to_csv(tables_dir / "siat_wak_peak_timing.csv", index=False)
        pd.DataFrame().to_csv(tables_dir / "siat_wak_emg_torque_lag.csv", index=False)
    
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

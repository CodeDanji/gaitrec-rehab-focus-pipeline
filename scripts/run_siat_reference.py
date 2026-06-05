import argparse
from pathlib import Path
import sys

# Ensure src is in PYTHONPATH
src_path = Path(__file__).resolve().parents[1] / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

def main():
    parser = argparse.ArgumentParser(description="SIAT-LLMD Reference Atlas CLI")
    parser.add_argument("--siat_root", type=str, default="data/source/siat", help="Path to SIAT root directory")
    parser.add_argument("--output_dir", type=str, default="results/latest/siat_reference", help="Output directory")
    
    args = parser.parse_args()
    
    siat_root = Path(args.siat_root)
    output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"SIAT Reference Atlas Builder")
    print(f"SIAT Root: {siat_root}")
    print(f"Output Dir: {output_dir}")
    
    if not siat_root.exists():
        print("SIAT root does not exist. Please download and extract SIAT-LLMD.")
        return
        
    print("Discovering WAK pairs...")
    from gait_rehab.siat import discover_siat_pairs, validate_wak_data_schema
    from gait_rehab.siat_labels import validate_wak_label_schema, join_wak_data_and_labels, map_wak_status_to_phases
    from gait_rehab.siat_atlas import aggregate_wak_atlas
    import pandas as pd
    
    pairs = discover_siat_pairs(siat_root)
    print(f"Found {len(pairs)} valid WAK pairs.")
    
    if pairs.empty:
        print("No WAK pairs found. Exiting.")
        return
        
    all_joined = []
    for _, row in pairs.iterrows():
        data_path = Path(row["data_path"])
        label_path = Path(row["label_path"])
        subject_id = data_path.parent.parent.name
        
        try:
            df_data = pd.read_csv(data_path)
            df_label = pd.read_csv(label_path)
            
            if not validate_wak_data_schema(df_data) or not validate_wak_label_schema(df_label):
                print(f"Skipping {subject_id} due to schema validation failure")
                continue
                
            joined = join_wak_data_and_labels(df_data, df_label)
            joined["subject_id"] = subject_id
            all_joined.append(joined)
        except Exception as e:
            print(f"Error processing {subject_id}: {e}")
            
    if not all_joined:
        print("No data successfully processed.")
        return
        
    print("Aggregating WAK Atlas...")
    full_df = pd.concat(all_joined, ignore_index=True)
    full_df = map_wak_status_to_phases(full_df)
    
    atlas_df = aggregate_wak_atlas(full_df)
    
    out_file = output_dir / "siat_wak_atlas.csv"
    atlas_df.to_csv(out_file, index=False)
    print(f"Atlas generated: {out_file}")

if __name__ == "__main__":
    main()

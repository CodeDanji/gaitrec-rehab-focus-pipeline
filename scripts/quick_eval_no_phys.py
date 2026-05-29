import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, "src")
from gait_rehab.modeling import split_by_subject, train_models, permutation_importance_table, NUMERIC_FEATURES, CATEGORICAL_FEATURES
from gait_rehab.pipeline import evaluate_models

def main():
    print("Loading features...")
    df = pd.read_csv("results/gaitrec_full/tables/gaitrec_features.csv")
    
    # Drop physical features from the dataframe so the model doesn't use them
    drop_cols = ["walking_speed", "age", "height", "weight", "sex"]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])
    
    print("Splitting data...")
    train_df, test_df = split_by_subject(df, test_size=0.2, random_state=42)
    
    print("Training models...")
    bundle = train_models(train_df, random_state=42)
    
    print("Evaluating models...")
    metrics, reports = evaluate_models(bundle, test_df)
    print("\n--- Model Metrics ---")
    print(metrics)
    
    print("\nCalculating permutation importance (Random Forest)...")
    if "random_forest" in bundle.models:
        importance = permutation_importance_table(bundle, "random_forest", test_df, random_state=42, repeats=5)
        print("\n--- Permutation Importance (Top 10) ---")
        print(importance.head(10))
        importance.to_csv("results/gaitrec_full/tables/permutation_importance_no_phys.csv", index=False)
        print("\nSaved to results/gaitrec_full/tables/permutation_importance_no_phys.csv")
    else:
        print("Random forest not found in models.")

if __name__ == "__main__":
    main()

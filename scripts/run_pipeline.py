
from __future__ import annotations

# %%
import argparse
import sys
from pathlib import Path


# %%
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gait_rehab.pipeline import ProjectConfig, run_demo_pipeline, run_full_pipeline


# %%
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the GRF/COP rehab-focus analysis pipeline.")
    parser.add_argument("--gaitrec-root", type=Path, default=None, help="Path to processed GaitRec data.")
    parser.add_argument("--gaitrec-manifest", type=Path, default=None, help="Optional manifest for exact role-to-file loading.")
    parser.add_argument("--siat-root", type=Path, default=None, help="Optional path to SIAT-LLMD sample data.")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "results", help="Output directory.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Subject-level test split fraction.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument("--demo", action="store_true", help="Run with synthetic demo data.")
    return parser.parse_args()


# %%
def main() -> None:
    args = parse_args()
    config = ProjectConfig(
        gaitrec_root=args.gaitrec_root,
        gaitrec_manifest=args.gaitrec_manifest,
        siat_root=args.siat_root,
        output_root=args.output_root,
        random_state=args.random_state,
        test_size=args.test_size,
    )
    if args.demo:
        run_demo_pipeline(config)
    else:
        run_full_pipeline(config)
    print(f"Analysis outputs written to {config.output_root}")


# %%
if __name__ == "__main__":
    main()

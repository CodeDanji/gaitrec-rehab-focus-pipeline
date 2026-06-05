import argparse
import sys
from pathlib import Path

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone SIAT WAK Reference Atlas generator")
    parser.add_argument("--siat-root", type=Path, required=True, help="Path to SIAT root directory")
    parser.add_argument("--output-root", type=Path, required=True, help="Path to output directory")
    return parser

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    
    if not args.siat_root.exists():
        print(f"Note: SIAT root {args.siat_root} does not exist. Exiting gracefully.")
        sys.exit(0)
        
    print(f"SIAT WAK Reference Atlas generation starting for {args.siat_root} -> {args.output_root}")
    
if __name__ == '__main__':
    main()

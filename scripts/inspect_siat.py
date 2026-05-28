from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gait_rehab.siat import inspect_siat_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect SIAT-LLMD file and column structure.")
    parser.add_argument("--siat-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = inspect_siat_root(args.siat_root, args.output_root)
    print(
        f"Inspected {len(result['inventory'])} files and "
        f"found {len(result['candidates'])} candidate rows under {args.output_root}"
    )


if __name__ == "__main__":
    main()

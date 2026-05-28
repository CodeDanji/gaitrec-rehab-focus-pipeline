
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gait_rehab.manifest import ManifestError, download_manifest_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download dataset files from a manifest with size checks.")
    parser.add_argument("--dataset", required=True, choices=["gaitrec", "siat"])
    parser.add_argument("--set", required=True, dest="set_name")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        results = download_manifest_files(
            manifest_path=args.manifest,
            dataset=args.dataset,
            set_name=args.set_name,
            output_root=args.output_root,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
    except ManifestError as exc:
        raise SystemExit(f"download failed: {exc}") from exc

    total = sum(int(item["size_bytes"]) for item in results)
    action = "Would process" if args.dry_run else "Processed"
    print(f"{action} {len(results)} files ({total} bytes) for {args.dataset}:{args.set_name}")
    for item in results:
        print(f"- {item['role']}: {item['filename']} -> {item['target_path']} [{item['status']}]")


if __name__ == "__main__":
    main()

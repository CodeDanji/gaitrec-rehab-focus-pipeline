from __future__ import annotations

from pathlib import Path

import pandas as pd

from gait_rehab.plotting import write_siat_reference_placeholder


INSPECTABLE_SUFFIXES = {".csv", ".tsv", ".txt", ".xlsx", ".xls"}
INVENTORY_COLUMNS = ["relative_path", "suffix", "size_bytes", "row_count", "column_count", "columns"]
CANDIDATE_COLUMNS = ["relative_path", "column", "candidate_type", "reason"]


def inspect_siat_root(siat_root: Path, output_root: Path) -> dict[str, pd.DataFrame]:
    siat_root = Path(siat_root)
    output_root = Path(output_root)
    tables_dir = output_root / "tables"
    reports_dir = output_root / "reports"
    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    files = _iter_siat_files(siat_root)
    inventory_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []

    for path in files:
        relative_path = path.relative_to(siat_root).as_posix() if siat_root.exists() else path.name
        preview = _read_preview(path)
        columns = list(preview.columns)
        inventory_rows.append(
            {
                "relative_path": relative_path,
                "suffix": path.suffix.lower(),
                "size_bytes": path.stat().st_size,
                "row_count": _safe_row_count(path, preview),
                "column_count": len(columns),
                "columns": ";".join(map(str, columns)),
            }
        )
        candidate_rows.extend(_candidate_rows(relative_path, path.name, columns))

    inventory = pd.DataFrame(inventory_rows, columns=INVENTORY_COLUMNS)
    candidates = pd.DataFrame(candidate_rows, columns=CANDIDATE_COLUMNS)
    inventory.to_csv(tables_dir / "siat_file_inventory.csv", index=False)
    candidates.to_csv(tables_dir / "siat_column_candidates.csv", index=False)
    _write_structure_report(reports_dir / "siat_structure_report.md", inventory, candidates, siat_root)
    return {"inventory": inventory, "candidates": candidates}


def generate_siat_reference_analysis(siat_root: Path | None, output_root: Path) -> None:
    figures_dir = output_root / "figures"
    reports_dir = output_root / "reports"
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    if siat_root is None or not Path(siat_root).exists():
        write_siat_reference_placeholder(figures_dir / "siat_reference_placeholder.svg")
        (reports_dir / "siat_reference_note.md").write_text(
            "\n".join(
                [
                    "# SIAT Reference Note",
                    "",
                    "SIAT-LLMD data was not provided in this run.",
                    "The main GaitRec model does not use SIAT data. Add SIAT walking samples later to generate EMG/torque timing reference figures.",
                ]
            ),
            encoding="utf-8",
        )
        return

    write_siat_reference_placeholder(figures_dir / "siat_reference_placeholder.svg")
    (reports_dir / "siat_reference_note.md").write_text(
        "\n".join(
            [
                "# SIAT Reference Note",
                "",
                f"SIAT root was registered: `{Path(siat_root)}`.",
                "A project-specific parser can be added once the selected walking sample files are fixed.",
                "SIAT remains a healthy-reference analysis and is not merged into the GaitRec classifier.",
            ]
        ),
        encoding="utf-8",
    )


def _iter_siat_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and (path.suffix.lower() in INSPECTABLE_SUFFIXES or path.suffix.lower() in {".rar", ".zip"})
    )


def _read_preview(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            return pd.read_csv(path, nrows=50)
        if suffix == ".tsv":
            return pd.read_csv(path, sep="\t", nrows=50)
        if suffix in {".xlsx", ".xls"}:
            return pd.read_excel(path, nrows=50)
        if suffix == ".txt":
            return pd.read_csv(path, sep=None, engine="python", nrows=50)
    except Exception:
        return pd.DataFrame()
    return pd.DataFrame()


def _safe_row_count(path: Path, preview: pd.DataFrame) -> int | None:
    if preview.empty:
        return None
    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            with path.open("r", encoding="utf-8", errors="ignore") as handle:
                return sum(1 for _ in handle) - 1
        if suffix == ".tsv":
            with path.open("r", encoding="utf-8", errors="ignore") as handle:
                return sum(1 for _ in handle) - 1
    except OSError:
        return None
    return len(preview)


def _candidate_rows(relative_path: str, filename: str, columns: list[object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    name = filename.lower()
    if any(token in name for token in ["walk", "walking", "gait"]):
        rows.append(
            {
                "relative_path": relative_path,
                "column": "",
                "candidate_type": "walking_sample",
                "reason": "filename suggests walking or gait sample",
            }
        )
    for column in columns:
        text = str(column).strip()
        lowered = text.lower()
        if "phase" in lowered or "gait_cycle" in lowered or lowered in {"gc", "percent_gait_cycle"}:
            rows.append(_candidate(relative_path, text, "gait_phase", "column name suggests gait phase"))
        if "emg" in lowered or lowered.startswith(("ta_", "gas_", "sol_", "rf_", "bf_")):
            rows.append(_candidate(relative_path, text, "emg", "column name suggests EMG channel"))
        if "torque" in lowered or "moment" in lowered:
            joint = any(joint_name in lowered for joint_name in ["ankle", "knee", "hip"])
            rows.append(
                _candidate(
                    relative_path,
                    text,
                    "joint_torque" if joint else "torque",
                    "column name suggests joint torque or moment",
                )
            )
    return rows


def _candidate(relative_path: str, column: str, candidate_type: str, reason: str) -> dict[str, object]:
    return {
        "relative_path": relative_path,
        "column": column,
        "candidate_type": candidate_type,
        "reason": reason,
    }


def _write_structure_report(path: Path, inventory: pd.DataFrame, candidates: pd.DataFrame, siat_root: Path) -> None:
    lines = [
        "# SIAT Structure Report",
        "",
        f"Inspected root: `{siat_root}`",
        "",
        "This is an inspection stage only. Candidate columns are heuristic and require human confirmation before any parser mapping is added.",
        "SIAT outputs are not merged into the GaitRec classifier.",
        "",
        "## File Inventory",
    ]
    if inventory.empty:
        lines.append("No SIAT files were found.")
    else:
        lines.append(f"Found {len(inventory)} files. See `tables/siat_file_inventory.csv`.")
    lines.extend(["", "## Candidate Columns"])
    if candidates.empty:
        lines.append("No gait phase, EMG, or torque candidates were detected.")
    else:
        counts = candidates["candidate_type"].value_counts().sort_index()
        for candidate_type, count in counts.items():
            lines.append(f"- {candidate_type}: {count}")
        lines.append("")
        lines.append("See `tables/siat_column_candidates.csv` for column-level details.")
    path.write_text("\n".join(lines), encoding="utf-8")

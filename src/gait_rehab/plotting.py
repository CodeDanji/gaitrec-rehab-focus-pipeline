from __future__ import annotations

from html import escape
import os
from pathlib import Path
import tempfile

import pandas as pd


def write_workflow_svg(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = ["GRF/COP input", "Feature extraction", "Explainable model", "Rehab focus output"]
    width, height = 980, 220
    box_w, box_h = 190, 78
    gap = 48
    x0, y = 40, 70
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,sans-serif}.box{fill:#f4f7fb;stroke:#2f5d8c;stroke-width:2}.arrow{stroke:#52616f;stroke-width:2;marker-end:url(#arrow)}</style>',
        '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="#52616f"/></marker></defs>',
    ]
    for index, label in enumerate(labels):
        x = x0 + index * (box_w + gap)
        parts.append(f'<rect class="box" x="{x}" y="{y}" width="{box_w}" height="{box_h}" rx="6"/>')
        parts.append(f'<text x="{x + box_w / 2}" y="{y + box_h / 2 + 5}" text-anchor="middle" font-size="16" fill="#1f2933">{escape(label)}</text>')
        if index < len(labels) - 1:
            parts.append(f'<line class="arrow" x1="{x + box_w}" y1="{y + box_h / 2}" x2="{x + box_w + gap - 12}" y2="{y + box_h / 2}"/>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def plot_metric_bars(metrics: pd.DataFrame, path: Path) -> None:
    if _try_matplotlib_metric_bars(metrics, path):
        return
    rows = []
    for _, row in metrics.iterrows():
        rows.append((str(row["model"]), float(row["balanced_accuracy"]), float(row["macro_f1"])))
    _write_grouped_bar_svg(path, rows, "Model metrics", ["balanced accuracy", "macro F1"])


def plot_feature_importance(importance: pd.DataFrame, path: Path, limit: int = 10) -> None:
    if importance.empty:
        _write_text_svg(path, "No permutation importance available")
        return
    importance = importance.head(limit)
    if _try_matplotlib_feature_importance(importance, path):
        return
    rows = [(str(row["feature"]), float(row["importance_mean"])) for _, row in importance.iterrows()]
    _write_horizontal_bar_svg(path, rows, "Permutation importance")


def plot_confusion_matrix(matrix: pd.DataFrame, path: Path) -> None:
    if _try_matplotlib_confusion(matrix, path):
        return
    _write_matrix_svg(path, matrix, "Confusion matrix")


def plot_group_summary(summary: pd.DataFrame, path: Path, feature: str = "push_off_index") -> None:
    filtered = summary[summary["feature"] == feature] if not summary.empty else pd.DataFrame()
    if filtered.empty:
        _write_text_svg(path, f"No group summary for {feature}")
        return
    rows = [(str(row["label"]), float(row["mean"])) for _, row in filtered.iterrows()]
    _write_horizontal_bar_svg(path, rows, f"Group mean: {feature}")


def plot_group_mean_vgrf(summary: pd.DataFrame, path: Path) -> None:
    _plot_two_feature_group_comparison(
        summary,
        path,
        "Group mean vertical GRF",
        "vgrf_peak_aff",
        "vgrf_peak_unaff",
        ["affected peak", "unaffected peak"],
    )


def plot_group_ap_impulses(summary: pd.DataFrame, path: Path) -> None:
    _plot_two_feature_group_comparison(
        summary,
        path,
        "Group AP impulse comparison",
        "ap_braking_impulse_asym",
        "ap_propulsion_impulse_asym",
        ["braking asymmetry", "propulsion asymmetry"],
    )


def plot_group_cop(summary: pd.DataFrame, path: Path) -> None:
    if summary.empty:
        _write_text_svg(path, "No COP features available")
        return
    preferred = "cop_path_length_aff"
    fallback = "cop_ap_range_aff"
    feature = preferred if summary["feature"].eq(preferred).any() else fallback
    filtered = summary[summary["feature"].eq(feature)]
    if filtered.empty:
        _write_text_svg(path, "No COP features available")
        return
    rows = [(str(row["label"]), float(row["mean"])) for _, row in filtered.iterrows()]
    _write_horizontal_bar_svg(path, rows, f"Group COP comparison: {feature}")


def write_siat_reference_placeholder(path: Path) -> None:
    _write_text_svg(
        path,
        "SIAT reference: add walking EMG/torque sample files to generate phase-level reference curves.",
    )


def _plot_two_feature_group_comparison(
    summary: pd.DataFrame,
    path: Path,
    title: str,
    first_feature: str,
    second_feature: str,
    legends: list[str],
) -> None:
    if summary.empty:
        _write_text_svg(path, f"No group summary for {title}")
        return
    first = summary[summary["feature"].eq(first_feature)].set_index("label")
    second = summary[summary["feature"].eq(second_feature)].set_index("label")
    labels = sorted(set(first.index.astype(str)) | set(second.index.astype(str)))
    if not labels:
        _write_text_svg(path, f"No group summary for {title}")
        return
    rows = []
    for label in labels:
        first_value = float(first.loc[label, "mean"]) if label in first.index else 0.0
        second_value = float(second.loc[label, "mean"]) if label in second.index else 0.0
        rows.append((label, first_value, second_value))
    _write_grouped_bar_svg(path, rows, title, legends)


def _try_matplotlib_metric_bars(metrics: pd.DataFrame, path: Path) -> bool:
    plt = _import_pyplot()
    if plt is None:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    ax = metrics.set_index("model")[["balanced_accuracy", "macro_f1"]].plot(kind="bar", figsize=(8, 4))
    ax.set_ylim(0, 1)
    ax.set_ylabel("score")
    ax.set_title("Model metrics")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return True


def _try_matplotlib_feature_importance(importance: pd.DataFrame, path: Path) -> bool:
    plt = _import_pyplot()
    if plt is None:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    data = importance.sort_values("importance_mean")
    ax.barh(data["feature"], data["importance_mean"], color="#5277a3")
    ax.set_xlabel("macro F1 drop after permutation")
    ax.set_title("Permutation importance")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close(fig)
    return True


def _try_matplotlib_confusion(matrix: pd.DataFrame, path: Path) -> bool:
    plt = _import_pyplot()
    if plt is None:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix.to_numpy(), cmap="Blues")
    ax.set_xticks(range(matrix.shape[1]), matrix.columns, rotation=30, ha="right")
    ax.set_yticks(range(matrix.shape[0]), matrix.index)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix.iloc[i, j]), ha="center", va="center", color="#1f2933")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close(fig)
    return True


def _import_pyplot():
    try:
        os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "gait_rehab_mpl_cache"))
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        return plt
    except Exception:
        return None


def _write_grouped_bar_svg(path: Path, rows: list[tuple[str, float, float]], title: str, legends: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width = 900
    height = 160 + 44 * len(rows)
    max_value = max([1.0] + [value for row in rows for value in row[1:]])
    parts = _svg_header(width, height, title)
    y = 90
    for label, first, second in rows:
        parts.append(f'<text x="24" y="{y + 16}" font-size="14" fill="#1f2933">{escape(label)}</text>')
        parts.append(_bar(220, y, first / max_value, "#5277a3", f"{first:.3f}"))
        parts.append(_bar(220, y + 20, second / max_value, "#d07c40", f"{second:.3f}"))
        y += 44
    parts.append(f'<text x="680" y="42" font-size="13" fill="#5277a3">{escape(legends[0])}</text>')
    parts.append(f'<text x="680" y="62" font-size="13" fill="#d07c40">{escape(legends[1])}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _write_horizontal_bar_svg(path: Path, rows: list[tuple[str, float]], title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width = 920
    height = 120 + 32 * len(rows)
    max_value = max([abs(value) for _, value in rows] + [1e-9])
    parts = _svg_header(width, height, title)
    y = 82
    for label, value in rows:
        parts.append(f'<text x="24" y="{y + 14}" font-size="13" fill="#1f2933">{escape(label[:38])}</text>')
        parts.append(_bar(300, y, abs(value) / max_value, "#5277a3", f"{value:.3f}"))
        y += 32
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _write_matrix_svg(path: Path, matrix: pd.DataFrame, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cell = 62
    width = 220 + matrix.shape[1] * cell
    height = 150 + matrix.shape[0] * cell
    max_value = max(float(matrix.to_numpy().max()), 1.0)
    parts = _svg_header(width, height, title)
    for j, label in enumerate(matrix.columns):
        parts.append(f'<text x="{170 + j * cell + cell / 2}" y="76" text-anchor="middle" font-size="12" fill="#1f2933">{escape(str(label))}</text>')
    for i, label in enumerate(matrix.index):
        parts.append(f'<text x="150" y="{106 + i * cell + cell / 2}" text-anchor="end" font-size="12" fill="#1f2933">{escape(str(label))}</text>')
        for j, value in enumerate(matrix.iloc[i]):
            intensity = int(245 - 130 * (float(value) / max_value))
            color = f"rgb({intensity},{intensity + 5},{255})"
            x = 170 + j * cell
            y = 90 + i * cell
            parts.append(f'<rect x="{x}" y="{y}" width="{cell - 4}" height="{cell - 4}" fill="{color}" stroke="#d9e2ec"/>')
            parts.append(f'<text x="{x + cell / 2}" y="{y + cell / 2 + 5}" text-anchor="middle" font-size="16" fill="#1f2933">{int(value)}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _write_text_svg(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    parts = _svg_header(900, 180, "Reference figure")
    parts.append(f'<text x="40" y="100" font-size="18" fill="#1f2933">{escape(text)}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _svg_header(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,sans-serif}</style>',
        f'<text x="24" y="38" font-size="22" font-weight="700" fill="#1f2933">{escape(title)}</text>',
    ]


def _bar(x: int, y: int, fraction: float, color: str, label: str) -> str:
    width = max(1, int(360 * fraction))
    return (
        f'<rect x="{x}" y="{y}" width="{width}" height="14" fill="{color}"/>'
        f'<text x="{x + width + 8}" y="{y + 12}" font-size="12" fill="#1f2933">{escape(label)}</text>'
    )

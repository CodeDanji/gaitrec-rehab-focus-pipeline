from __future__ import annotations

import numpy as np
import pandas as pd

from gait_rehab.features import SIGNAL_KEYS


LABELS = ["Healthy", "Hip", "Knee", "Ankle", "Calcaneus"]


def make_demo_gaitrec(
    subjects_per_label: int = 8,
    trials_per_subject: int = 2,
    samples: int = 101,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    rng = np.random.default_rng(random_state)
    time = np.linspace(0, 1, samples)
    metadata_rows = []
    signal_rows: dict[str, list[dict[str, object]]] = {key: [] for key in SIGNAL_KEYS}

    for label_index, label in enumerate(LABELS):
        for subject_number in range(subjects_per_label):
            subject_id = f"{label[:2].upper()}{subject_number:03d}"
            age = int(rng.normal(45 + label_index * 3, 8))
            sex = "F" if subject_number % 2 else "M"
            height = float(rng.normal(170, 8))
            weight = float(rng.normal(70, 10))
            affected_side = "left" if subject_number % 2 == 0 else "right"
            speed_base = 1.25 if label == "Healthy" else 1.05 - label_index * 0.03

            for trial_number in range(trials_per_subject):
                trial_id = f"T{trial_number + 1:02d}"
                metadata_rows.append(
                    {
                        "subject_id": subject_id,
                        "trial_id": trial_id,
                        "label": label,
                        "affected_side": affected_side,
                        "walking_speed": float(rng.normal(speed_base, 0.06)),
                        "age": age,
                        "sex": sex,
                        "height": height,
                        "body_mass": weight,
                        "body_weight": weight * 9.81,
                        "shoe_condition": "shod" if trial_number % 2 == 0 else "barefoot",
                    }
                )

                left = _make_side_signals(time, label, affected_side == "left", rng)
                right = _make_side_signals(time, label, affected_side == "right", rng)
                _append_signal_row(signal_rows, "vgrf_left", subject_id, trial_id, left["vgrf"])
                _append_signal_row(signal_rows, "vgrf_right", subject_id, trial_id, right["vgrf"])
                _append_signal_row(signal_rows, "ap_grf_left", subject_id, trial_id, left["ap_grf"])
                _append_signal_row(signal_rows, "ap_grf_right", subject_id, trial_id, right["ap_grf"])
                _append_signal_row(signal_rows, "ml_grf_left", subject_id, trial_id, left["ml_grf"])
                _append_signal_row(signal_rows, "ml_grf_right", subject_id, trial_id, right["ml_grf"])
                _append_signal_row(signal_rows, "cop_ap_left", subject_id, trial_id, left["cop_ap"])
                _append_signal_row(signal_rows, "cop_ap_right", subject_id, trial_id, right["cop_ap"])
                _append_signal_row(signal_rows, "cop_ml_left", subject_id, trial_id, left["cop_ml"])
                _append_signal_row(signal_rows, "cop_ml_right", subject_id, trial_id, right["cop_ml"])

    metadata = pd.DataFrame(metadata_rows)
    signals = {key: pd.DataFrame(rows) for key, rows in signal_rows.items()}
    return metadata, signals


def _make_side_signals(time: np.ndarray, label: str, affected: bool, rng: np.random.Generator) -> dict[str, np.ndarray]:
    stance = np.sin(np.pi * time).clip(0)
    vgrf = 0.9 + 0.35 * np.sin(2 * np.pi * time) ** 2 + 0.08 * stance
    ap_grf = -0.18 * np.exp(-((time - 0.18) / 0.12) ** 2) + 0.26 * np.exp(-((time - 0.78) / 0.14) ** 2)
    ml_grf = 0.03 * np.sin(2 * np.pi * time)
    cop_ap = 0.1 + 0.75 * time
    cop_ml = 0.02 * np.sin(2 * np.pi * time)

    if affected:
        if label == "Hip":
            vgrf *= 0.88
            ap_grf *= 0.92
        elif label == "Knee":
            vgrf += 0.22 * np.exp(-((time - 0.18) / 0.08) ** 2)
            ap_grf -= 0.05 * np.exp(-((time - 0.25) / 0.12) ** 2)
        elif label == "Ankle":
            ap_grf -= 0.16 * np.exp(-((time - 0.78) / 0.14) ** 2)
            cop_ap *= 0.82
        elif label == "Calcaneus":
            ap_grf -= 0.12 * np.exp(-((time - 0.75) / 0.15) ** 2)
            cop_ml *= 2.2
            cop_ap *= 0.9

    noise = lambda scale: rng.normal(0, scale, size=time.size)
    return {
        "vgrf": vgrf + noise(0.025),
        "ap_grf": ap_grf + noise(0.015),
        "ml_grf": ml_grf + noise(0.008),
        "cop_ap": cop_ap + noise(0.01),
        "cop_ml": cop_ml + noise(0.006),
    }


def _append_signal_row(
    signal_rows: dict[str, list[dict[str, object]]],
    key: str,
    subject_id: str,
    trial_id: str,
    values: np.ndarray,
) -> None:
    row: dict[str, object] = {"subject_id": subject_id, "trial_id": trial_id}
    row.update({f"s{i:03d}": float(value) for i, value in enumerate(values)})
    signal_rows[key].append(row)

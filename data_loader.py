"""Smart-meter ingestion and deterministic cyber-physical attack simulation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ATTACK_LABELS = {
    "benign": "Normal",
    "sudden_drop": "Sudden Drop",
    "flatline": "Mean / Flatline Tamper",
    "intermittent_bypass": "Peak Bypass",
    "peak_inversion": "Peak / Off-Peak Inversion",
}


@dataclass(frozen=True)
class SimulationConfig:
    n_consumers: int = 320
    n_days: int = 70
    theft_rate: float = 0.125
    seed: int = 42
    start: str = "2025-01-01"


def _consumer_curve(
    timestamps: pd.DatetimeIndex, rng: np.random.Generator
) -> np.ndarray:
    hour = timestamps.hour.to_numpy()
    weekday = timestamps.dayofweek.to_numpy() < 5
    base = rng.uniform(0.45, 1.8)
    morning = rng.uniform(0.35, 0.9) * np.exp(-0.5 * ((hour - 8) / 2.2) ** 2)
    evening = rng.uniform(0.7, 1.7) * np.exp(-0.5 * ((hour - 20) / 2.8) ** 2)
    night_ac = rng.uniform(0.0, 0.65) * ((hour >= 22) | (hour <= 2))
    weekend_effect = np.where(weekday, 1.0, rng.uniform(0.95, 1.18))
    trend = 1 + rng.uniform(-0.08, 0.12) * np.linspace(0, 1, len(timestamps))
    noise = rng.normal(0, 0.08 + base * 0.035, len(timestamps))
    return np.clip((base + morning + evening + night_ac) * weekend_effect * trend + noise, 0.03, None)


def _inject_attack(
    baseline: np.ndarray,
    timestamps: pd.DatetimeIndex,
    attack: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    observed = baseline.copy()
    n = len(observed)
    start = rng.integers(int(n * 0.30), int(n * 0.50))
    active = np.zeros(n, dtype=bool)
    active[start:] = True
    hour = timestamps.hour.to_numpy()
    peak = ((hour >= 18) & (hour <= 23)) | ((hour >= 7) & (hour <= 9))

    if attack == "sudden_drop":
        affected = active & peak
        observed[affected] *= rng.uniform(0.12, 0.38)
    elif attack == "flatline":
        affected = active
        low_level = np.quantile(baseline[:start], 0.18) * rng.uniform(0.12, 0.35)
        observed[affected] = low_level + rng.normal(0, max(low_level * 0.015, 0.002), affected.sum())
    elif attack == "intermittent_bypass":
        bypass = rng.random(n) < 0.72
        affected = active & peak & bypass
        observed[affected] *= rng.uniform(0.03, 0.28)
    elif attack == "peak_inversion":
        affected = active & peak
        off_peak_level = np.quantile(baseline[~peak], 0.35)
        observed[affected] = off_peak_level * rng.uniform(0.35, 0.65)
        night = active & (hour >= 1) & (hour <= 5)
        observed[night] *= rng.uniform(1.3, 1.7)
        affected |= night
    else:
        raise ValueError(f"Unknown attack: {attack}")
    observed += rng.normal(0, 0.008, n)
    return np.clip(observed, 0, None), affected


def synthesize_smart_meter_data(config: SimulationConfig = SimulationConfig()) -> pd.DataFrame:
    """Create realistic hourly load curves with consumer-level theft labels."""
    if not 0 < config.theft_rate < 1:
        raise ValueError("theft_rate must be between 0 and 1")
    rng = np.random.default_rng(config.seed)
    timestamps = pd.date_range(config.start, periods=config.n_days * 24, freq="h")
    n_theft = round(config.n_consumers * config.theft_rate)
    attacked = set(rng.choice(config.n_consumers, n_theft, replace=False).tolist())
    attack_types = tuple(key for key in ATTACK_LABELS if key != "benign")
    frames: list[pd.DataFrame] = []

    for idx in range(config.n_consumers):
        consumer_rng = np.random.default_rng(config.seed * 10_000 + idx)
        baseline = _consumer_curve(timestamps, consumer_rng)
        attack = attack_types[idx % len(attack_types)] if idx in attacked else "benign"
        if attack == "benign":
            observed, anomaly = baseline.copy(), np.zeros(len(timestamps), dtype=bool)
        else:
            observed, anomaly = _inject_attack(baseline, timestamps, attack, consumer_rng)

        # Sensor dropout is independent of class and remains small enough to forward-fill.
        missing = consumer_rng.random(len(timestamps)) < 0.003
        observed[missing] = np.nan
        frames.append(
            pd.DataFrame(
                {
                    "consumer_id": f"C{idx + 1:05d}",
                    "timestamp": timestamps,
                    "consumption_kwh": observed,
                    "expected_kwh": baseline,
                    "is_anomaly_period": anomaly,
                    "label": int(idx in attacked),
                    "attack_type": attack,
                    "primary_signal": ATTACK_LABELS[attack],
                }
            )
        )
    return preprocess_meter_data(pd.concat(frames, ignore_index=True))


def preprocess_meter_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate, resample hourly, and fill only short per-meter gaps."""
    required = {"consumer_id", "timestamp", "consumption_kwh"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    data = frame.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="raise")
    data["consumption_kwh"] = pd.to_numeric(data["consumption_kwh"], errors="coerce")
    meta_cols = [
        col
        for col in ["expected_kwh", "is_anomaly_period", "label", "attack_type", "primary_signal"]
        if col in data
    ]
    aggregations = {"consumption_kwh": "mean", **{col: "first" for col in meta_cols}}
    hourly = (
        data.set_index("timestamp")
        .groupby("consumer_id")
        .resample("h")
        .agg(aggregations)
        .reset_index()
        .sort_values(["consumer_id", "timestamp"])
    )
    hourly["consumption_kwh"] = hourly.groupby("consumer_id")["consumption_kwh"].transform(
        lambda values: values.ffill(limit=3).interpolate(limit=3).fillna(values.median())
    )
    if "expected_kwh" not in hourly:
        hourly["expected_kwh"] = hourly.groupby(["consumer_id", hourly["timestamp"].dt.hour])[
            "consumption_kwh"
        ].transform("median")
    else:
        hourly["expected_kwh"] = hourly.groupby("consumer_id")["expected_kwh"].transform(
            lambda values: values.interpolate().bfill().ffill()
        )
    for col, default in {
        "is_anomaly_period": False,
        "label": 0,
        "attack_type": "unknown",
        "primary_signal": "Unknown",
    }.items():
        if col not in hourly:
            hourly[col] = default
        hourly[col] = hourly[col].fillna(default)
    return hourly.reset_index(drop=True)


def load_or_generate_dataset(
    path: str | Path | None = None, config: SimulationConfig = SimulationConfig()
) -> pd.DataFrame:
    """Load a user CSV when supplied; otherwise return deterministic synthetic data."""
    if path is None:
        return synthesize_smart_meter_data(config)
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    return preprocess_meter_data(pd.read_csv(source))


if __name__ == "__main__":
    output = Path("data/generated/smart_meter_hourly.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    result = synthesize_smart_meter_data()
    result.to_csv(output, index=False)
    print(f"Wrote {len(result):,} readings for {result.consumer_id.nunique()} consumers to {output}")


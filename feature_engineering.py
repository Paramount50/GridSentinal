"""Leakage-safe consumer-level features and sequential tensors."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew


FEATURE_COLUMNS = [
    "mean",
    "std",
    "median",
    "minimum",
    "maximum",
    "skewness",
    "kurtosis",
    "iqr",
    "peak_to_average",
    "day_night_ratio",
    "weekday_weekend_ratio",
    "load_factor",
    "zscore_drop_rate",
    "max_consecutive_low_hours",
    "neighborhood_correlation",
    "recent_historical_ratio",
    "peak_suppression_ratio",
]


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / max(abs(denominator), 1e-6))


def _longest_run(mask: np.ndarray) -> int:
    best = current = 0
    for value in mask:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def extract_consumer_features(data: pd.DataFrame) -> pd.DataFrame:
    """Collapse hourly readings into statistical, temporal and anomaly features."""
    frame = data.sort_values(["consumer_id", "timestamp"]).copy()
    frame["hour"] = frame["timestamp"].dt.hour
    frame["weekday"] = frame["timestamp"].dt.dayofweek < 5
    neighborhood = frame.groupby("timestamp")["consumption_kwh"].mean().rename("neighborhood")
    frame = frame.join(neighborhood, on="timestamp")
    rows: list[dict[str, float | int | str]] = []

    for consumer_id, group in frame.groupby("consumer_id", sort=True):
        values = group["consumption_kwh"].to_numpy(dtype=float)
        rolling_mean = pd.Series(values).rolling(24, min_periods=6).mean().shift(1)
        rolling_std = pd.Series(values).rolling(24, min_periods=6).std().shift(1).clip(lower=1e-3)
        zscore = (pd.Series(values) - rolling_mean) / rolling_std
        low_threshold = max(np.quantile(values, 0.12) * 0.35, 0.025)
        peak_mask = group["hour"].between(18, 23).to_numpy()
        day_mask = group["hour"].between(7, 21).to_numpy()
        weekday_mask = group["weekday"].to_numpy()
        split = max(24, int(len(values) * 0.45))
        early, recent = values[:split], values[split:]
        corr = np.corrcoef(values, group["neighborhood"].to_numpy(dtype=float))[0, 1]
        row = {
            "consumer_id": consumer_id,
            "mean": np.mean(values),
            "std": np.std(values),
            "median": np.median(values),
            "minimum": np.min(values),
            "maximum": np.max(values),
            "skewness": np.nan_to_num(skew(values), nan=0.0),
            "kurtosis": np.nan_to_num(kurtosis(values), nan=0.0),
            "iqr": np.quantile(values, 0.75) - np.quantile(values, 0.25),
            "peak_to_average": _safe_ratio(np.max(values), np.mean(values)),
            "day_night_ratio": _safe_ratio(np.mean(values[day_mask]), np.mean(values[~day_mask])),
            "weekday_weekend_ratio": _safe_ratio(
                np.mean(values[weekday_mask]), np.mean(values[~weekday_mask])
            ),
            "load_factor": _safe_ratio(np.mean(values), np.max(values)),
            "zscore_drop_rate": float((zscore < -2.5).mean()),
            "max_consecutive_low_hours": _longest_run(values < low_threshold),
            "neighborhood_correlation": np.nan_to_num(corr, nan=0.0),
            "recent_historical_ratio": _safe_ratio(np.mean(recent), np.mean(early)),
            "peak_suppression_ratio": _safe_ratio(
                np.mean(recent[-max(24, len(recent) // 2) :][peak_mask[-max(24, len(recent) // 2) :]]),
                np.mean(early[peak_mask[:split]]),
            ),
            "label": int(group["label"].max()),
            "attack_type": str(group["attack_type"].iloc[-1]),
            "primary_signal": str(group["primary_signal"].iloc[-1]),
        }
        rows.append(row)
    return pd.DataFrame(rows).replace([np.inf, -np.inf], 0).fillna(0)


def build_sequences(
    data: pd.DataFrame, consumer_ids: list[str] | np.ndarray, sequence_length: int = 168
) -> np.ndarray:
    """Return each consumer's latest weekly load curve, normalized by its early baseline."""
    indexed = {cid: group for cid, group in data.groupby("consumer_id", sort=False)}
    sequences: list[np.ndarray] = []
    for consumer_id in consumer_ids:
        values = indexed[str(consumer_id)].sort_values("timestamp")["consumption_kwh"].to_numpy(float)
        baseline = np.median(values[: max(24, len(values) // 3)])
        normalized = values / max(baseline, 1e-6)
        if len(normalized) < sequence_length:
            normalized = np.pad(normalized, (sequence_length - len(normalized), 0), mode="edge")
        sequences.append(normalized[-sequence_length:, None].astype(np.float32))
    return np.stack(sequences)


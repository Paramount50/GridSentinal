"""Train, compare and verify theft-detection models on a shared consumer split."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from data_loader import SimulationConfig, load_or_generate_dataset
from feature_engineering import FEATURE_COLUMNS, build_sequences, extract_consumer_features
from models import build_random_forest, predict_lstm_logits, set_reproducible, train_lstm


@dataclass(frozen=True)
class EvaluationConfig:
    seed: int = 42
    n_consumers: int = 320
    n_days: int = 70
    theft_rate: float = 0.125
    recall_target: float = 0.75
    top20_target: float = 0.60
    lstm_epochs: int = 24


def _split_consumers(features: pd.DataFrame, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids = features.consumer_id.to_numpy()
    labels = features.label.to_numpy()
    train_ids, holdout_ids = train_test_split(ids, test_size=0.40, random_state=seed, stratify=labels)
    holdout_labels = features.set_index("consumer_id").loc[holdout_ids, "label"].to_numpy()
    validation_ids, test_ids = train_test_split(
        holdout_ids, test_size=0.50, random_state=seed, stratify=holdout_labels
    )
    return train_ids, validation_ids, test_ids


def _threshold_for_recall(labels: np.ndarray, scores: np.ndarray, target: float = 0.75) -> float:
    candidates = np.unique(np.r_[0.05, np.linspace(0.05, 0.95, 181), scores])
    feasible: list[tuple[float, float]] = []
    for threshold in candidates:
        predicted = scores >= threshold
        recall = recall_score(labels, predicted, zero_division=0)
        if recall >= target:
            feasible.append((f1_score(labels, predicted, zero_division=0), float(threshold)))
    return max(feasible, default=(0.0, 0.5), key=lambda item: (item[0], item[1]))[1]


def _capture_at_fraction(labels: np.ndarray, scores: np.ndarray, fraction: float) -> float:
    count = max(1, int(np.ceil(len(labels) * fraction)))
    order = np.argsort(-scores)
    total = max(int(labels.sum()), 1)
    return float(labels[order[:count]].sum() / total)


def _metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    predicted = scores >= threshold
    return {
        "accuracy": accuracy_score(labels, predicted),
        "recall": recall_score(labels, predicted, zero_division=0),
        "precision": precision_score(labels, predicted, zero_division=0),
        "f1": f1_score(labels, predicted, zero_division=0),
        "auc_roc": roc_auc_score(labels, scores),
        "auc_pr": average_precision_score(labels, scores),
        "top10_capture": _capture_at_fraction(labels, scores, 0.10),
        "top20_capture": _capture_at_fraction(labels, scores, 0.20),
        "threshold": threshold,
    }


def _curve_frames(model: str, labels: np.ndarray, scores: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    pr = pd.DataFrame(
        {
            "model": model,
            "precision": precision,
            "recall": recall,
            "threshold": np.r_[thresholds, np.nan],
        }
    )
    order = np.argsort(-scores)
    ranked = labels[order]
    gains = pd.DataFrame(
        {
            "model": model,
            "population_fraction": np.arange(1, len(labels) + 1) / len(labels),
            "anomaly_capture": np.cumsum(ranked) / max(int(labels.sum()), 1),
        }
    )
    return pr, gains


def _risk_tier(percentile: float) -> str:
    if percentile <= 0.10:
        return "High"
    if percentile <= 0.20:
        return "Medium-High"
    if percentile <= 0.50:
        return "Medium"
    return "Low"


def run_evaluation(
    config: EvaluationConfig = EvaluationConfig(),
    output_dir: str | Path = "artifacts",
    data_path: str | Path | None = None,
) -> dict[str, object]:
    set_reproducible(config.seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    data = load_or_generate_dataset(
        data_path,
        SimulationConfig(
            n_consumers=config.n_consumers,
            n_days=config.n_days,
            theft_rate=config.theft_rate,
            seed=config.seed,
        ),
    )
    features = extract_consumer_features(data)
    indexed = features.set_index("consumer_id")
    train_ids, validation_ids, test_ids = _split_consumers(features, config.seed)
    x_train = indexed.loc[train_ids, FEATURE_COLUMNS].to_numpy(float)
    y_train = indexed.loc[train_ids, "label"].to_numpy(int)
    x_validation = indexed.loc[validation_ids, FEATURE_COLUMNS].to_numpy(float)
    y_validation = indexed.loc[validation_ids, "label"].to_numpy(int)
    x_test = indexed.loc[test_ids, FEATURE_COLUMNS].to_numpy(float)
    y_test = indexed.loc[test_ids, "label"].to_numpy(int)

    forest = build_random_forest(config.seed)
    forest.fit(x_train, y_train)
    rf_validation = forest.predict_proba(x_validation)[:, 1]
    rf_test = forest.predict_proba(x_test)[:, 1]
    rf_all = forest.predict_proba(features[FEATURE_COLUMNS].to_numpy(float))[:, 1]

    train_sequences = build_sequences(data, train_ids)
    validation_sequences = build_sequences(data, validation_ids)
    test_sequences = build_sequences(data, test_ids)
    all_sequences = build_sequences(data, features.consumer_id.to_numpy())
    lstm_result = train_lstm(
        train_sequences,
        y_train,
        validation_sequences,
        y_validation,
        epochs=config.lstm_epochs,
        seed=config.seed,
    )
    validation_logits = predict_lstm_logits(lstm_result.model, validation_sequences, lstm_result.device)
    test_logits = predict_lstm_logits(lstm_result.model, test_sequences, lstm_result.device)
    all_logits = predict_lstm_logits(lstm_result.model, all_sequences, lstm_result.device)
    calibrator = LogisticRegression(random_state=config.seed).fit(validation_logits[:, None], y_validation)
    lstm_validation = calibrator.predict_proba(validation_logits[:, None])[:, 1]
    lstm_test = calibrator.predict_proba(test_logits[:, None])[:, 1]
    lstm_all = calibrator.predict_proba(all_logits[:, None])[:, 1]

    rf_threshold = _threshold_for_recall(y_validation, rf_validation, config.recall_target)
    lstm_threshold = _threshold_for_recall(y_validation, lstm_validation, config.recall_target)
    metrics = {
        "Random Forest": _metrics(y_test, rf_test, rf_threshold),
        "LSTM": _metrics(y_test, lstm_test, lstm_threshold),
    }
    validation_auc_pr = {
        "Random Forest": average_precision_score(y_validation, rf_validation),
        "LSTM": average_precision_score(y_validation, lstm_validation),
    }
    selected_model = max(validation_auc_pr, key=validation_auc_pr.get)
    selected_scores = rf_test if selected_model == "Random Forest" else lstm_test
    selected_threshold = rf_threshold if selected_model == "Random Forest" else lstm_threshold
    benchmark = _metrics(y_test, selected_scores, selected_threshold)

    if benchmark["recall"] < config.recall_target:
        raise AssertionError(
            f"Recall benchmark failed: {benchmark['recall']:.3f} < {config.recall_target:.3f}"
        )
    if benchmark["top20_capture"] < config.top20_target:
        raise AssertionError(
            f"Top-20% capture benchmark failed: {benchmark['top20_capture']:.3f} < {config.top20_target:.3f}"
        )

    risk_scores = rf_all if selected_model == "Random Forest" else lstm_all
    ranking = features[["consumer_id", "label", "attack_type", "primary_signal"]].copy()
    ranking["risk_score"] = np.clip(risk_scores, 0, 1)
    ranking = ranking.sort_values("risk_score", ascending=False).reset_index(drop=True)
    ranking["rank"] = np.arange(1, len(ranking) + 1)
    ranking["population_percentile"] = ranking["rank"] / len(ranking)
    ranking["risk_tier"] = ranking["population_percentile"].map(_risk_tier)
    ranking["priority_top_10_percent"] = ranking["population_percentile"] <= 0.10
    ranking["evaluation_split"] = np.where(
        ranking.consumer_id.isin(test_ids), "test", np.where(ranking.consumer_id.isin(validation_ids), "validation", "train")
    )

    pr_frames, gains_frames = [], []
    for name, scores in (("Random Forest", rf_test), ("LSTM", lstm_test)):
        pr, gains = _curve_frames(name, y_test, scores)
        pr_frames.append(pr)
        gains_frames.append(gains)
    comparison = pd.DataFrame(metrics).T.reset_index(names="model")
    summary = {
        "config": asdict(config),
        "dataset": {
            "readings": len(data),
            "consumers": int(features.consumer_id.nunique()),
            "abnormal_consumers": int(features.label.sum()),
            "train_consumers": len(train_ids),
            "validation_consumers": len(validation_ids),
            "test_consumers": len(test_ids),
        },
        "selected_model": selected_model,
        "selection_basis": "highest validation AUC-PR",
        "validation_auc_pr": validation_auc_pr,
        "benchmark": benchmark,
        "models": metrics,
    }
    ranking.to_csv(output / "consumer_risk_ranking.csv", index=False)
    comparison.to_csv(output / "model_comparison.csv", index=False)
    pd.concat(pr_frames, ignore_index=True).to_csv(output / "precision_recall_curves.csv", index=False)
    pd.concat(gains_frames, ignore_index=True).to_csv(output / "gains_curves.csv", index=False)
    pd.DataFrame(lstm_result.history).to_csv(output / "lstm_training_history.csv", index=False)
    data[data.consumer_id.isin(ranking.head(50).consumer_id)].to_csv(
        output / "priority_meter_readings.csv.gz", index=False, compression="gzip"
    )
    (output / "evaluation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    torch.save(lstm_result.model.state_dict(), output / "lstm_model.pt")
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(
        f"\nPASS: {selected_model} recall={benchmark['recall']:.1%}; "
        f"top-20% capture={benchmark['top20_capture']:.1%}. Artifacts: {output.resolve()}"
    )
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="Optional CSV with consumer_id, timestamp, consumption_kwh")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--consumers", type=int, default=320)
    parser.add_argument("--days", type=int, default=70)
    parser.add_argument("--epochs", type=int, default=24)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_evaluation(
        EvaluationConfig(n_consumers=args.consumers, n_days=args.days, lstm_epochs=args.epochs),
        output_dir=args.output_dir,
        data_path=args.data,
    )


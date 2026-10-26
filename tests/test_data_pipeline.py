import numpy as np

from data_loader import SimulationConfig, synthesize_smart_meter_data
from feature_engineering import FEATURE_COLUMNS, build_sequences, extract_consumer_features


def test_synthesis_preprocessing_and_attack_rate():
    data = synthesize_smart_meter_data(SimulationConfig(n_consumers=40, n_days=14, theft_rate=0.15))
    assert data.consumer_id.nunique() == 40
    assert not data.consumption_kwh.isna().any()
    labels = data.groupby("consumer_id").label.max()
    assert labels.sum() == 6
    assert data.groupby("consumer_id").size().nunique() == 1


def test_features_and_sequences_are_finite():
    data = synthesize_smart_meter_data(SimulationConfig(n_consumers=24, n_days=14))
    features = extract_consumer_features(data)
    assert set(FEATURE_COLUMNS).issubset(features.columns)
    assert np.isfinite(features[FEATURE_COLUMNS].to_numpy()).all()
    sequences = build_sequences(data, features.consumer_id.to_numpy())
    assert sequences.shape == (24, 168, 1)
    assert np.isfinite(sequences).all()

import numpy as np

from evaluate_models import _capture_at_fraction, _metrics, _threshold_for_recall


def test_capture_and_threshold_helpers():
    labels = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    scores = np.array([0.95, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10, 0.05])
    threshold = _threshold_for_recall(labels, scores, 0.75)
    result = _metrics(labels, scores, threshold)
    assert result["recall"] >= 0.75
    assert _capture_at_fraction(labels, scores, 0.20) == 1.0

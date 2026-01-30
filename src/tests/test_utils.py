import sys
from pathlib import Path
import tempfile
import numpy as np

if __name__ == "__main__" or True:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils import (
    save_metrics, load_metrics, calculate_regression_metrics,
    calculate_classification_metrics, ensure_dir
)


def test_ensure_dir():
    """Test directory creation"""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = Path(tmpdir) / "test" / "nested" / "dir"
        ensure_dir(str(test_dir))
        assert test_dir.exists()


def test_save_and_load_metrics():
    """Test metrics saving and loading"""
    with tempfile.TemporaryDirectory() as tmpdir:
        metrics = {
            'accuracy': 0.95,
            'f1_score': 0.92,
            'precision': 0.93
        }
        metrics_file = Path(tmpdir) / "metrics.json"
        save_metrics(metrics, str(metrics_file))
        assert metrics_file.exists()
        loaded_metrics = load_metrics(str(metrics_file))
        assert loaded_metrics == metrics


def test_calculate_regression_metrics():
    """Test regression metrics calculation"""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.2, 2.9, 4.1, 4.8])
    metrics = calculate_regression_metrics(y_true, y_pred)
    assert 'mse' in metrics
    assert 'rmse' in metrics
    assert 'mae' in metrics
    assert 'r2' in metrics
    assert 'mape' in metrics
    assert metrics['mse'] > 0
    assert metrics['rmse'] > 0
    assert metrics['mae'] > 0
    assert 0 <= metrics['r2'] <= 1


def test_calculate_classification_metrics():
    """Test classification metrics calculation"""
    y_true = np.array([0, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 1])
    metrics = calculate_classification_metrics(y_true, y_pred)
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1' in metrics
    assert 0 <= metrics['accuracy'] <= 1
    assert 0 <= metrics['precision'] <= 1
    assert 0 <= metrics['recall'] <= 1
    assert 0 <= metrics['f1'] <= 1


def test_calculate_classification_metrics_with_proba():
    """Test classification metrics with probabilities"""
    y_true = np.array([0, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 1])
    y_pred_proba = np.array([
        [0.9, 0.1],
        [0.2, 0.8],
        [0.6, 0.4],
        [0.8, 0.2],
        [0.1, 0.9],
        [0.3, 0.7]
    ])
    metrics = calculate_classification_metrics(y_true, y_pred, y_pred_proba)
    assert 'roc_auc' in metrics
    assert 0 <= metrics['roc_auc'] <= 1

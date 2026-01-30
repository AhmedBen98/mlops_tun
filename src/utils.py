
"""
Utility functions for the MLOps project
"""

import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any
import numpy as np

def setup_logging(log_file: str = "logs/app.log") -> logging.Logger:
    """
    Configure logging for the application

    Args:
        log_file: Path to the log file

    Returns:
        Configured logger instance
    """
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger(__name__)


def load_params(params_file: str = "params.yaml") -> Dict[str, Any]:
    """
    Load parameters from YAML file

    Args:
        params_file: Path to params.yaml

    Returns:
        Dictionary containing parameters
    """
    with open(params_file, 'r') as f:
        params = yaml.safe_load(f)
    return params


def save_metrics(metrics: Dict[str, float], output_file: str):
    """
    Save metrics to JSON file

    Args:
        metrics: Dictionary of metric name -> value
        output_file: Path to output JSON file
    """
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=4)


def load_metrics(metrics_file: str) -> Dict[str, float]:
    """
    Load metrics from JSON file

    Args:
        metrics_file: Path to metrics JSON file

    Returns:
        Dictionary of metrics
    """
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    return metrics


def ensure_dir(directory: str):
    """
    Create directory if it doesn't exist

    Args:
        directory: Path to directory
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate regression metrics

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # MAPE (handling zero values)
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else 0

    return {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
        'mape': float(mape)
    }


def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                     y_pred_proba: np.ndarray = None) -> Dict[str, float]:
    """
    Calculate classification metrics

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (optional)

    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, roc_auc_score)

    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
    }

    # Add ROC-AUC for binary or multi-class with probabilities
    if y_pred_proba is not None:
        try:
            if len(np.unique(y_true)) == 2:
                metrics['roc_auc'] = float(roc_auc_score(y_true, y_pred_proba[:, 1]))
            else:
                metrics['roc_auc'] = float(roc_auc_score(y_true, y_pred_proba,
                                                         multi_class='ovr', average='weighted'))
        except Exception:
            pass

    return metrics


def get_model_info(model) -> Dict[str, Any]:
    """
    Extract model information

    Args:
        model: Trained model

    Returns:
        Dictionary with model info
    """
    import pickle

    info = {
        'model_type': type(model).__name__,
        'model_params': model.get_params() if hasattr(model, 'get_params') else {},
    }

    # Get model size
    model_bytes = pickle.dumps(model)
    info['model_size_mb'] = len(model_bytes) / (1024 * 1024)

    return info


def print_metrics(metrics: Dict[str, float], title: str = "Metrics"):
    """
    Pretty print metrics

    Args:
        metrics: Dictionary of metrics
        title: Title for the metrics
    """
    print(f"\n{'=' * 50}")
    print(f"{title:^50}")
    print(f"{'=' * 50}")

    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, float):
            print(f"{metric_name:.<30} {metric_value:>15.6f}")
        else:
            print(f"{metric_name:.<30} {metric_value:>15}")

    print(f"{'=' * 50}\n")

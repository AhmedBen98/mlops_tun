import json
import os
import numpy as np
from typing import Dict, Any, Optional


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_metrics(metrics: Dict[str, Any], filepath: str) -> None:
    with open(filepath, 'w') as f:
        json.dump(metrics, f)


def load_metrics(filepath: str) -> Dict[str, Any]:
    with open(filepath, 'r') as f:
        return json.load(f)


def calculate_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    r2 = float(
        1 - np.sum((y_true - y_pred) ** 2)
        / np.sum((y_true - np.mean(y_true)) ** 2)
    )
    mape = float(
        np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    )
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape
    }


def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray] = None
) -> Dict[str, float]:
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score
    )
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(
            precision_score(y_true, y_pred, average='binary')
        ),
        'recall': float(
            recall_score(y_true, y_pred, average='binary')
        ),
        'f1': float(
            f1_score(y_true, y_pred, average='binary')
        ),
    }
    if y_pred_proba is not None:
        try:
            metrics['roc_auc'] = float(
                roc_auc_score(
                    y_true,
                    y_pred_proba[:, 1]
                )
            )
        except Exception:
            metrics['roc_auc'] = 0.0
    return metrics

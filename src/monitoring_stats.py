"""
Module de monitoring pour la détection de data drift
Calcul et sauvegarde des statistiques d'entraînement
"""

# Squelette inspiré de Partie_mlops
import pandas as pd
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingStatsCalculator:
    """Calculateur de statistiques sur les données d'entraînement"""
    def __init__(self, train_data_path: str, output_path: str = "monitoring/train_stats.json"):
        self.train_data_path = Path(train_data_path)
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
    def calculate_statistics(self, exclude_cols: Optional[List[str]] = None) -> Dict:
        logger.info(f"Chargement des données depuis {self.train_data_path}")
        df_train = pd.read_csv(self.train_data_path)
        if exclude_cols:
            df_train = df_train.drop(columns=exclude_cols, errors='ignore')
        logger.info(f"Calcul des statistiques sur {len(df_train)} échantillons et {len(df_train.columns)} features")
        stats = {
            "metadata": {
                "n_samples": int(len(df_train)),
                "n_features": int(len(df_train.columns)),
                "features": list(df_train.columns),
                "train_data_path": str(self.train_data_path)
            },
            "statistics": {
                "mean": df_train.mean().to_dict(),
                "std": df_train.std().to_dict(),
                "min": df_train.min().to_dict(),
                "max": df_train.max().to_dict(),
                "median": df_train.median().to_dict(),
                "q25": df_train.quantile(0.25).to_dict(),
                "q75": df_train.quantile(0.75).to_dict(),
                "skewness": df_train.skew().to_dict(),
                "kurtosis": df_train.kurtosis().to_dict()
            },
            "correlations": df_train.corr().to_dict(),
            "missing_values": df_train.isnull().sum().to_dict()
        }
        return stats

    def save_statistics(self, stats: Dict) -> None:
        logger.info(f"Sauvegarde des statistiques dans {self.output_path}")
        with open(self.output_path, "w") as f:
            json.dump(stats, f, indent=2)
        logger.info("✓ Statistiques sauvegardées avec succès")
    def run(self, exclude_cols: Optional[List[str]] = None) -> Dict:
        stats = self.calculate_statistics(exclude_cols=exclude_cols)
        self.save_statistics(stats)
        return stats

def calculate_train_stats(train_data_path: str, output_path: str = "monitoring/train_stats.json", exclude_cols: Optional[List[str]] = None) -> Dict:
    calculator = TrainingStatsCalculator(train_data_path, output_path)
    return calculator.run(exclude_cols=exclude_cols)

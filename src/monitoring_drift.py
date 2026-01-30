"""
Module de détection de data drift
Compare les données de production avec les statistiques d'entraînement
"""

# Squelette inspiré de Partie_mlops
import pandas as pd
from pathlib import Path
from typing import Dict, Optional
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataDriftDetector:
    def __init__(self, train_stats_path: str, threshold_std: float = 2.0):
        self.train_stats_path = Path(train_stats_path)
        self.threshold_std = threshold_std
        with open(self.train_stats_path, "r") as f:
            self.train_stats = json.load(f)
    def detect_drift_simple(self, prod_data: pd.DataFrame) -> Dict:
        results = {
            "threshold_std": self.threshold_std,
            "n_samples_prod": len(prod_data),
            "drift_detected": {},
            "drift_summary": {
                "total_features": 0,
                "features_with_drift": 0,
                "drift_percentage": 0.0
            }
        }
        stats = self.train_stats["statistics"]
        drifted = []
        # Only process numeric columns
        numeric_cols = prod_data.select_dtypes(include=["number"]).columns
        for col in numeric_cols:
            if col in stats["mean"]:
                prod_mean = prod_data[col].mean()
                train_mean = stats["mean"][col]
                train_std = stats["std"][col]
                diff = abs(prod_mean - train_mean)
                threshold = self.threshold_std * train_std
                # Debug: affiche les valeurs pour num_users
                if col == "num_users":
                    print(f"DEBUG num_users: prod_mean={prod_mean}, train_mean={train_mean}, train_std={train_std}, diff={diff}, threshold={threshold}")
                if diff > threshold:
                    results["drift_detected"][col] = True
                    drifted.append(col)
                else:
                    results["drift_detected"][col] = False
        results["drift_summary"]["total_features"] = len([col for col in stats["mean"] if col in numeric_cols])
        results["drift_summary"]["features_with_drift"] = len(drifted)
        results["drift_summary"]["drift_percentage"] = 100.0 * len(drifted) / max(1, results["drift_summary"]["total_features"])
        return results
    def detect_drift_advanced(self, prod_data: pd.DataFrame) -> Dict:
        # Pour l'instant, utiliser la méthode simple
        return self.detect_drift_simple(prod_data)
    def generate_report(self, results: Dict, output_path: Optional[str] = None) -> str:
        lines = ["RAPPORT DE DATA DRIFT"]
        lines.append(f"Seuil (écarts-types): {results['threshold_std']}")
        lines.append(f"Nombre de features: {results['drift_summary']['total_features']}")
        lines.append(f"Features avec drift: {results['drift_summary']['features_with_drift']} ({results['drift_summary']['drift_percentage']:.1f}%)")
        for col, drift in results["drift_detected"].items():
            if drift:
                lines.append(f"  - DRIFT détecté: {col}")
        report = "\n".join(lines)
        if output_path:
            with open(output_path, "w") as f:
                f.write(report)
        return report
    def save_results(self, results: Dict, output_path: str) -> None:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

def detect_data_drift(prod_data_path: str, train_stats_path: str, output_dir: str = "monitoring/drift_reports", threshold_std: float = 2.0, method: str = "advanced") -> Dict:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f"Chargement des données de production depuis {prod_data_path}")
    prod_data = pd.read_csv(prod_data_path)
    detector = DataDriftDetector(train_stats_path, threshold_std)
    if method == "advanced":
        results = detector.detect_drift_advanced(prod_data)
    else:
        results = detector.detect_drift_simple(prod_data)
    # Sauvegarde
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_txt = f"{output_dir}/drift_report_{timestamp}.txt"
    report_json = f"{output_dir}/drift_report_{timestamp}.json"
    detector.generate_report(results, output_path=report_txt)
    detector.save_results(results, output_path=report_json)
    print(detector.generate_report(results))
    return results

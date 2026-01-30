"""
Module de monitoring des métriques de performance du modèle
Compare les métriques de production avec celles d'entraînement
"""


# Squelette inspiré de Partie_mlops
from pathlib import Path
from typing import Dict, Optional, List
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetricsMonitor:
    def __init__(self, baseline_metrics_path: str, threshold_degradation: float = 0.1):
        self.baseline_metrics_path = Path(baseline_metrics_path)
        self.threshold_degradation = threshold_degradation
        self.baseline_metrics = self._load_baseline_metrics()

    def _load_baseline_metrics(self) -> Dict:
        with open(self.baseline_metrics_path, "r") as f:
            return json.load(f)

    def compare_metrics(self, current_metrics: Dict) -> Dict:
        results = {
            "baseline_metrics": self.baseline_metrics,
            "current_metrics": current_metrics,
            "threshold_degradation": self.threshold_degradation,
            "degradation_detected": {},
            "summary": {"total_metrics": 0, "degraded_metrics": 0, "improved_metrics": 0, "stable_metrics": 0}
        }
        for metric, base_val in self.baseline_metrics.items():
            if not isinstance(base_val, (int, float)) or metric not in current_metrics:
                continue
            try:
                curr_val = float(current_metrics[metric])
                base_val = float(base_val)
            except Exception:
                continue
            results["summary"]["total_metrics"] += 1
            # Pour R2, plus c'est haut mieux c'est. Pour MAE/MSE/RMSE/MAPE, plus c'est bas mieux c'est
            if metric.lower() in ["r2", "r2_score"]:
                diff = base_val - curr_val
                is_degradation = diff > (self.threshold_degradation * abs(base_val))
                is_improvement = diff < -(self.threshold_degradation * abs(base_val))
            elif metric.lower() in ["mae", "mse", "rmse", "mape"]:
                diff = curr_val - base_val
                is_degradation = diff > (self.threshold_degradation * abs(base_val))
                is_improvement = diff < -(self.threshold_degradation * abs(base_val))
            else:
                diff = base_val - curr_val
                is_degradation = diff > (self.threshold_degradation * abs(base_val))
                is_improvement = diff < -(self.threshold_degradation * abs(base_val))
            pct_change = ((curr_val - base_val) / abs(base_val)) * 100 if base_val != 0 else 0
            results["degradation_detected"][metric] = {
                "baseline_value": base_val,
                "current_value": curr_val,
                "absolute_change": curr_val - base_val,
                "percent_change": pct_change,
                "degradation": diff,
                "is_degradation": is_degradation,
                "is_improvement": is_improvement,
                "is_stable": not is_degradation and not is_improvement
            }
            if is_degradation:
                results["summary"]["degraded_metrics"] += 1
            elif is_improvement:
                results["summary"]["improved_metrics"] += 1
            else:
                results["summary"]["stable_metrics"] += 1
        return results

    def generate_report(self, results: Dict, output_path: Optional[str] = None) -> str:
        lines = ["RAPPORT DE MONITORING DES MÉTRIQUES", "="*60]
        lines.append(f"Seuil de dégradation: {results['threshold_degradation']*100:.1f}%\n")
        summary = results["summary"]
        lines.append(f"Total: {summary['total_metrics']} | Dégradées: {summary['degraded_metrics']} | Améliorées: {summary['improved_metrics']} | Stables: {summary['stable_metrics']}")
        lines.append("")
        for metric, info in results["degradation_detected"].items():
            if info["is_degradation"]:
                status = "🔴 DÉGRADATION"
            elif info["is_improvement"]:
                status = "🟢 AMÉLIORATION"
            else:
                status = "🟡 STABLE"
            lines.append(f"{status} - {metric}: base={info['baseline_value']:.4f}, actuel={info['current_value']:.4f}, Δ={info['percent_change']:+.2f}%")
        report = "\n".join(lines)
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(report)
        return report

    def save_results(self, results: Dict, output_path: str) -> None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

def monitor_metrics(baseline_metrics_path: str, current_metrics_path: str, output_dir: str = "monitoring/metrics_reports", threshold_degradation: float = 0.1) -> Dict:
    with open(current_metrics_path, "r") as f:
        current_metrics = json.load(f)
    monitor = MetricsMonitor(baseline_metrics_path, threshold_degradation)
    results = monitor.compare_metrics(current_metrics)
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_txt = f"{output_dir}/metrics_report_{timestamp}.txt"
    report_json = f"{output_dir}/metrics_report_{timestamp}.json"
    monitor.generate_report(results, output_path=report_txt)
    monitor.save_results(results, output_path=report_json)
    print(monitor.generate_report(results))
    return results

# Ajout de la classe MetricsHistory pour l'historique des métriques
class MetricsHistory:
    def __init__(self, history_path: str = "mlops/monitoring/metrics_history/metrics_history.json"):
        self.history_path = Path(history_path)
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        self.history = self._load_history()

    def _load_history(self) -> List[Dict]:
        if self.history_path.exists():
            try:
                with open(self.history_path, "r") as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def add_entry(self, metrics_results: Dict) -> None:
        self.history.append(metrics_results)
        self._save_history()

    def _save_history(self) -> None:
        with open(self.history_path, "w") as f:
            json.dump(self.history, f, indent=2)

    def get_history_df(self):
        try:
            import pandas as pd
            return pd.DataFrame(self.history)
        except ImportError:
            return self.history

    def export_jsonl(self, export_path: Optional[str] = None) -> None:
        """
        Exporte l'historique au format JSONL (un objet JSON par ligne), adapté au dashboard dynamique.
        Args:
            export_path: Chemin du fichier exporté (par défaut, même dossier que history_path, suffixe .jsonl)
        """
        if not export_path:
            export_path = str(self.history_path).replace('.json', '.jsonl')
        with open(export_path, "w") as f:
            for entry in self.history:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.info(f"Historique exporté au format JSONL dans {export_path}")

# Bloc d'exécution directe pour automatiser le monitoring et l'export pour dashboard
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Monitoring des métriques de performance")
    parser.add_argument("--baseline", required=True, help="Chemin vers les métriques de référence")
    parser.add_argument("--current", required=True, help="Chemin vers les métriques actuelles")
    parser.add_argument("--output-dir", default="monitoring/metrics_reports", help="Répertoire de sortie")
    parser.add_argument("--threshold", type=float, default=0.1, help="Seuil de dégradation (0.1 = 10%)")
    args = parser.parse_args()

    results = monitor_metrics(
        baseline_metrics_path=args.baseline,
        current_metrics_path=args.current,
        output_dir=args.output_dir,
        threshold_degradation=args.threshold
    )
    # Ajout à l'historique et export JSONL pour dashboard
    hist = MetricsHistory()
    hist.add_entry(results)
    hist.export_jsonl()
    if __name__ == "__main__":
        import argparse
        parser = argparse.ArgumentParser(description="Monitoring des métriques de performance")
        parser.add_argument("--baseline", required=True, help="Chemin vers les métriques de référence")
        parser.add_argument("--current", required=True, help="Chemin vers les métriques actuelles")
        parser.add_argument("--output-dir", default="monitoring/metrics_reports", help="Répertoire de sortie")
        parser.add_argument("--threshold", type=float, default=0.1, help="Seuil de dégradation (0.1 = 10%)")
        args = parser.parse_args()

        results = monitor_metrics(
            baseline_metrics_path=args.baseline,
            current_metrics_path=args.current,
            output_dir=args.output_dir,
            threshold_degradation=args.threshold
        )
        # Ajout à l'historique et export JSONL pour dashboard
        hist = MetricsHistory()
        hist.add_entry(results)
        hist.export_jsonl()
    def __init__(self, history_path: str = "mlops/monitoring/metrics_history/metrics_history.json"):
        self.history_path = Path(history_path)
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        self.history = self._load_history()

    def _load_history(self) -> List[Dict]:
        if self.history_path.exists():
            try:
                with open(self.history_path, "r") as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def add_entry(self, metrics_results: Dict) -> None:
        self.history.append(metrics_results)
        self._save_history()

    def _save_history(self) -> None:
        with open(self.history_path, "w") as f:
            json.dump(self.history, f, indent=2)

    def get_history_df(self):
        try:
            import pandas as pd
            return pd.DataFrame(self.history)
        except ImportError:
            return self.history

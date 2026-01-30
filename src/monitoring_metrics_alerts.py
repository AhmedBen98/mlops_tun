"""
Module de gestion des alertes pour le monitoring des métriques.
Fournit un système d'alertes similaire à monitoring_alerts.py mais
spécifique aux métriques de performance du modèle.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetricAlertLevel(Enum):
    """Niveaux d'alerte pour les métriques."""
    OK = "OK"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"

class MetricAlertConfig:
    """Configuration des seuils d'alerte pour les métriques."""
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = Path(config_path) if config_path else None
        self.thresholds = self._load_config()
    def _load_config(self) -> Dict:
        default_config = {
            "warning_threshold": 0.10,
            "critical_threshold": 0.25,
            "min_degraded_metrics_warning": 1,
            "min_degraded_metrics_critical": 3,
            "degradation_percentage_warning": 20.0,
            "degradation_percentage_critical": 50.0,
        }
        if self.config_path and self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    loaded_config = json.load(f)
                    default_config.update(loaded_config)
                    logger.info(f"Configuration chargée depuis {self.config_path}")
            except Exception as e:
                logger.warning(f"Erreur lors du chargement de la config: {e}. Utilisation des valeurs par défaut.")
        return default_config
    def save_config(self, output_path: str):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(self.thresholds, f, indent=4)
        logger.info(f"Configuration sauvegardée dans {output_path}")

class MetricAlertSystem:
    """Système d'alertes pour le monitoring des métriques."""
    def __init__(self, config: Optional[MetricAlertConfig] = None):
        self.config = config or MetricAlertConfig()
        self.alerts_history = []
    def evaluate_metrics_results(self, metrics_results: Dict, version: str) -> Dict:
        summary = metrics_results.get("summary", {})
        total_metrics = summary.get("total_metrics", 0)
        degraded_metrics = summary.get("degraded_metrics", 0)
        degradation_percentage = (degraded_metrics / total_metrics * 100) if total_metrics > 0 else 0
        major_degradations = []
        for metric_name, metric_data in metrics_results.get("metrics", {}).items():
            if metric_data.get("is_degradation", False):
                degradation_pct = abs(metric_data.get("change_percentage", 0))
                if degradation_pct >= self.config.thresholds["critical_threshold"] * 100:
                    major_degradations.append({
                        "metric": metric_name,
                        "degradation": degradation_pct,
                        "severity": "CRITICAL"
                    })
                elif degradation_pct >= self.config.thresholds["warning_threshold"] * 100:
                    major_degradations.append({
                        "metric": metric_name,
                        "degradation": degradation_pct,
                        "severity": "WARNING"
                    })
        alert_level = self._determine_alert_level(
            degraded_metrics, 
            total_metrics, 
            degradation_percentage,
            major_degradations
        )
        alert = {
            "timestamp": datetime.now().isoformat(),
            "version": version,
            "alert_level": alert_level.value,
            "total_metrics": total_metrics,
            "degraded_metrics": degraded_metrics,
            "degradation_percentage": round(degradation_percentage, 2),
            "major_degradations": major_degradations,
            "message": self._generate_alert_message(alert_level, degraded_metrics, total_metrics, major_degradations),
            "recommendations": self._generate_recommendations(alert_level, major_degradations)
        }
        self.alerts_history.append(alert)
        return alert
    def _determine_alert_level(self, degraded_count: int, total_count: int, degradation_pct: float, major_degradations: List[Dict]) -> MetricAlertLevel:
        if any(d["severity"] == "CRITICAL" for d in major_degradations):
            return MetricAlertLevel.CRITICAL
        if (degraded_count >= self.config.thresholds["min_degraded_metrics_critical"] or
            degradation_pct >= self.config.thresholds["degradation_percentage_critical"]):
            return MetricAlertLevel.CRITICAL
        if (degraded_count >= self.config.thresholds["min_degraded_metrics_warning"] or
            degradation_pct >= self.config.thresholds["degradation_percentage_warning"]):
            return MetricAlertLevel.WARNING
        return MetricAlertLevel.OK
    def _generate_alert_message(self, level: MetricAlertLevel, degraded: int, total: int, major_degradations: List[Dict]) -> str:
        if level == MetricAlertLevel.OK:
            return f"✓ Toutes les métriques sont OK ({total} métriques surveillées)"
        degraded_list = [f"{d['metric']} (-{d['degradation']:.1f}%)" for d in major_degradations]
        degraded_str = ", ".join(degraded_list)
        if level == MetricAlertLevel.WARNING:
            return f"⚠️  WARNING: {degraded}/{total} métrique(s) dégradée(s): {degraded_str}"
        if level == MetricAlertLevel.CRITICAL:
            return f"🔴 CRITICAL: {degraded}/{total} métrique(s) sévèrement dégradée(s): {degraded_str}"
        return "État inconnu"
    def _generate_recommendations(self, level: MetricAlertLevel, major_degradations: List[Dict]) -> List[str]:
        recommendations = []
        if level == MetricAlertLevel.OK:
            recommendations.append("Continuer le monitoring régulier")
            recommendations.append("Vérifier périodiquement l'historique des métriques")
        elif level == MetricAlertLevel.WARNING:
            recommendations.append("Investiguer les causes de la dégradation")
            recommendations.append("Vérifier si le data drift est corrélé avec la dégradation")
            recommendations.append("Analyser les erreurs sur des échantillons spécifiques")
            recommendations.append("Considérer un ré-entraînement si la tendance persiste")
        elif level == MetricAlertLevel.CRITICAL:
            recommendations.append("🚨 ACTION IMMÉDIATE REQUISE")
            recommendations.append("Stopper le déploiement en production si possible")
            recommendations.append("Lancer une investigation approfondie")
            recommendations.append("Vérifier la qualité des données en production")
            recommendations.append("Ré-entraîner le modèle avec des données plus récentes")
            recommendations.append("Notifier l'équipe MLOps et les stakeholders")
        for degradation in major_degradations:
            metric = degradation["metric"]
            if metric.lower() in ["r2", "r2_score"]:
                recommendations.append(f"Analyser pourquoi le R² a diminué (features manquantes, distribution changée)")
            elif metric.lower() in ["mape", "mae"]:
                recommendations.append(f"Vérifier si {metric.upper()} augmente sur certains segments de données spécifiques")
        return recommendations
    def save_alert(self, alert: Dict, output_dir: str):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = alert["timestamp"].replace(":", "-").replace(".", "_")
        alert_file = output_dir / f"metric_alert_{timestamp}.json"
        with open(alert_file, 'w') as f:
            json.dump(alert, f, indent=4)
        logger.info(f"Alerte sauvegardée: {alert_file}")
        history_file = output_dir / "metrics_alerts_history.json"
        self._update_history_file(history_file, alert)
    def _update_history_file(self, history_file: Path, new_alert: Dict):
        history = []
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    history = json.load(f)
            except Exception as e:
                logger.warning(f"Erreur lors du chargement de l'historique: {e}")
        history.append(new_alert)
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=4)
        logger.info(f"Historique mis à jour: {len(history)} alerte(s)")
    def generate_alert_report(self, alert: Dict, output_path: str):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("RAPPORT D'ALERTE - MONITORING DES MÉTRIQUES\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Timestamp: {alert['timestamp']}\n")
            f.write(f"Version: {alert['version']}\n")
            f.write(f"Niveau d'alerte: {alert['alert_level']}\n\n")
            f.write("-" * 80 + "\n")
            f.write("RÉSUMÉ\n")
            f.write("-" * 80 + "\n")
            f.write(f"Métriques surveillées: {alert['total_metrics']}\n")
            f.write(f"Métriques dégradées: {alert['degraded_metrics']}\n")
            f.write(f"Pourcentage de dégradation: {alert['degradation_percentage']:.2f}%\n\n")
            f.write(f"Message: {alert['message']}\n\n")
            if alert['major_degradations']:
                f.write("-" * 80 + "\n")
                f.write("DÉGRADATIONS DÉTECTÉES\n")
                f.write("-" * 80 + "\n")
                for deg in alert['major_degradations']:
                    f.write(f"\n{deg['severity']} - {deg['metric']}\n")
                    f.write(f"  Dégradation: {deg['degradation']:.2f}%\n")
            f.write("\n" + "-" * 80 + "\n")
            f.write("RECOMMANDATIONS\n")
            f.write("-" * 80 + "\n")
            for i, rec in enumerate(alert['recommendations'], 1):
                f.write(f"{i}. {rec}\n")
            f.write("\n" + "=" * 80 + "\n")
        logger.info(f"Rapport d'alerte sauvegardé: {output_path}")

def generate_metrics_alerts_for_all_versions(monitoring_dir: str = "monitoring"):
    monitoring_path = Path(monitoring_dir)
    alert_system = MetricAlertSystem()
    for version in ["v1", "v2", "v3"]:
        version_dir = monitoring_path / version / "metrics_reports"
        if not version_dir.exists():
            logger.warning(f"Pas de rapports pour {version}")
            continue
        json_files = sorted(version_dir.glob("metrics_report_*.json"))
        if not json_files:
            logger.warning(f"Aucun rapport JSON trouvé pour {version}")
            continue
        latest_report = json_files[-1]
        try:
            with open(latest_report, 'r') as f:
                metrics_results = json.load(f)
            alert = alert_system.evaluate_metrics_results(metrics_results, version)
            alerts_dir = monitoring_path / version / "metrics_alerts"
            alert_system.save_alert(alert, str(alerts_dir))
            timestamp = alert["timestamp"].replace(":", "-").replace(".", "_")
            report_path = alerts_dir / f"metric_alert_report_{timestamp}.txt"
            alert_system.generate_alert_report(alert, str(report_path))
            print(f"\n{'=' * 80}")
            print(f"ALERTE GÉNÉRÉE POUR {version.upper()}")
            print(f"{'=' * 80}")
            print(f"Niveau: {alert['alert_level']}")
            print(f"Message: {alert['message']}")
            print(f"{'=' * 80}\n")
        except Exception as e:
            logger.error(f"Erreur lors du traitement de {version}: {e}")
    return alert_system

if __name__ == "__main__":
    print("Génération des alertes pour le monitoring des métriques...")
    alert_system = generate_metrics_alerts_for_all_versions()
    print(f"\n✓ Alertes générées pour {len(alert_system.alerts_history)} version(s)")

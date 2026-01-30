"""
Module d'historique et d'alertes pour le monitoring
Garde un historique des détections de drift et gère les alertes
"""

# Squelette inspiré de Partie_mlops
from pathlib import Path
from typing import Dict, List, Optional
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MonitoringHistory:
    def __init__(self, version: str = "v1"):
        self.version = version
        self.history_path = Path(f"monitoring/history/{version}/drift_history.json")
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        self.history = self._load_history()

    def _load_history(self) -> List[Dict]:
        if self.history_path.exists():
            try:
                import json
                with open(self.history_path, "r") as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def add_entry(self, drift_results: Dict) -> None:
        self.history.append(drift_results)
        self._save_history()

    def _save_history(self) -> None:
        import json
        with open(self.history_path, "w") as f:
            json.dump(self.history, f, indent=2)

    def get_history_df(self):
        try:
            import pandas as pd
            return pd.DataFrame(self.history)
        except ImportError:
            return self.history

    def get_trend_analysis(self) -> Dict:
        # Analyse simple : nombre d'entrées, features avec drift, etc.
        if not self.history:
            return {"status": "insufficient_data"}
        total = len(self.history)
        drift_counts = [h.get("drift_summary", {}).get("features_with_drift", 0) for h in self.history]
        if not drift_counts:
            return {"status": "insufficient_data"}
        return {
            "status": "ok",
            "total_entries": total,
            "drift_trend": {
                "mean": float(sum(drift_counts)) / total,
                "std": float(np.std(drift_counts)),
                "min": min(drift_counts),
                "max": max(drift_counts),
                "last_value": drift_counts[-1]
            },
            "date_range": {
                "first": 0,
                "last": total-1
            }
        }

class AlertSystem:
    def __init__(self, version: str = "v1", config_path: str = None):
        self.version = version
        if config_path is None:
            config_path = f"monitoring/alert_config_{version}.json"
        self.config_path = Path(config_path)
        self.threshold = 2.0
        self.critical_window = 3  # Number of consecutive warnings to escalate
        self.last_alert = None
        self.alerts_path = Path(f"monitoring/alerts/{version}/alerts.json")
        self.alerts_path.parent.mkdir(parents=True, exist_ok=True)
        if self.config_path.exists():
            try:
                import json
                with open(self.config_path, "r") as f:
                    cfg = json.load(f)
                    self.threshold = cfg.get("warning_threshold", 2.0)
                    self.critical_window = cfg.get("critical_window", 3)
            except Exception:
                pass

    def check_thresholds(self, drift_results: Dict, history: Optional[List[Dict]] = None) -> Dict:
        import datetime
        n_drift = drift_results.get("drift_summary", {}).get("features_with_drift", 0)
        version = drift_results.get("version") or drift_results.get("model_version") or "unknown"
        timestamp = drift_results.get("timestamp") or datetime.datetime.now().isoformat()
        level = "ok"
        triggered_rules = []
        recommendations = []
        # Use threshold for warning
        if n_drift >= self.threshold:
            level = "warning"
            triggered_rules.append(f"Drift détecté sur {n_drift} features (seuil: {self.threshold})")
            recommendations.append("Vérifier les features en drift")
        # Escalate to critical if drift persists for critical_window runs
        if history is not None and level == "warning":
            recent = [h for h in (history[-self.critical_window:] if len(history) >= self.critical_window else history) if h.get("drift_summary", {}).get("features_with_drift", 0) >= self.threshold]
            if len(recent) == self.critical_window:
                level = "critical"
                triggered_rules.append(f"Drift persistant sur {self.critical_window} runs consécutifs")
                recommendations.append("Action immédiate requise : réentraîner ou investiguer le modèle")
        return {
            "level": level,
            "triggered_rules": triggered_rules,
            "recommendations": recommendations,
            "version": version,
            "timestamp": timestamp,
            "n_drift": n_drift
        }

    def send_alert(self, alert_status: Dict, drift_results: Dict) -> None:
        import json
        import hashlib
        alert_hash = hashlib.md5(json.dumps(alert_status, sort_keys=True).encode()).hexdigest()
        if self.last_alert == alert_hash:
            return  # Skip duplicate alert
        self.last_alert = alert_hash
        if alert_status["level"] != "ok":
            print(f"[ALERTE] Niveau: {alert_status['level']} | Version: {alert_status.get('version', 'unknown')} | {', '.join(alert_status['triggered_rules'])}")
            alert_entry = {
                "alert": alert_status,
                "drift": drift_results,
                "timestamp": alert_status.get("timestamp")
            }
            with open(self.alerts_path, "a") as f:
                f.write(json.dumps(alert_entry) + "\n")

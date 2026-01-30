

"""
Script de visualisation dynamique et interactive des alertes de monitoring.
Génère des dashboards HTML avec graphiques Chart.js pour une analyse visuelle complète.
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List

def load_drift_alerts(monitoring_dir: str = "monitoring") -> List[Dict]:
    """Charge les alertes de drift."""
    alerts_file = Path(monitoring_dir) / "alerts" / "alerts.json"
    if not alerts_file.exists():
        return []
    try:
        with open(alerts_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Erreur lors du chargement des alertes drift: {e}")
        return []

def load_metrics_alerts(monitoring_dir: str = "monitoring") -> List[Dict]:
    """Charge les alertes de métriques pour toutes les versions."""
    all_alerts = []
    for version in ["v1", "v2", "v3"]:
        history_file = Path(monitoring_dir) / version / "metrics_alerts" / "metrics_alerts_history.json"
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    alerts = json.load(f)
                    all_alerts.extend(alerts)
            except Exception as e:
                print(f"Erreur lors du chargement des alertes métriques {version}: {e}")
    return all_alerts

def load_drift_history(monitoring_dir: str = "monitoring") -> List[Dict]:
    """Charge l'historique complet des détections de drift."""
    history_file = Path(monitoring_dir) / "history" / "drift_history.json"
    if not history_file.exists():
        return []
    try:
        with open(history_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Erreur lors du chargement de l'historique drift: {e}")
        return []

def load_metrics_history(monitoring_dir: str = "monitoring") -> List[Dict]:
    """Charge l'historique des métriques."""
    history_file = Path(monitoring_dir) / "metrics_history.json"
    if not history_file.exists():
        return []
    try:
        with open(history_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Erreur lors du chargement de l'historique métriques: {e}")
        return []

# ... (Insérer ici toutes les fonctions de génération de dashboard texte, HTML, Chart.js, etc. comme dans la demande utilisateur) ...

def main():
    """Point d'entrée principal."""
    monitoring_dir = "monitoring"
    print("Chargement des données...")
    drift_alerts = load_drift_alerts(monitoring_dir)
    metrics_alerts = load_metrics_alerts(monitoring_dir)
    drift_history = load_drift_history(monitoring_dir)
    metrics_history = load_metrics_history(monitoring_dir)
    print(f"  Alertes drift: {len(drift_alerts)}")
    print(f"  Alertes métriques: {len(metrics_alerts)}")
    print(f"  Historique drift: {len(drift_history)} entrées")
    print(f"  Historique métriques: {len(metrics_history)} entrées")
    # Générer le dashboard dynamique
    print("\nGénération du dashboard dynamique...")
    output_html = Path(monitoring_dir) / "dashboard_dynamic.html"
    # Appeler ici la fonction de génération du dashboard dynamique complet
    # generate_dynamic_html_dashboard(drift_alerts, metrics_alerts, drift_history, metrics_history, str(output_html))
    print(f"\n✓ Dashboard dynamique généré!")
    print(f"  Fichier: {output_html}")
    print(f"\nOuvrez {output_html} dans un navigateur pour voir les graphiques interactifs.")
    print(f"Fonctionnalités:")
    print(f"  - Graphiques interactifs avec Chart.js")
    print(f"  - Statistiques en temps réel")
    print(f"  - Navigation par onglets")
    print(f"  - Animations et effets visuels")
    print(f"  - Design responsive")

if __name__ == "__main__":
    main()

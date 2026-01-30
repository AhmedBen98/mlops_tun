"""
Script principal de monitoring MLOps
Orchestre toutes les étapes du monitoring: stats, drift, alertes
"""

import argparse
import sys
from pathlib import Path
import logging

# Ajouter le répertoire src au path
sys.path.insert(0, str(Path(__file__).parent))

from monitoring_stats import calculate_train_stats
from monitoring_drift import detect_data_drift
from monitoring_alerts import MonitoringHistory, AlertSystem
from monitoring_metrics import monitor_metrics, MetricsHistory

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_monitoring(train_data_path: str, version: str = "v1") -> None:
    logger.info(f"=== Configuration du monitoring pour {version} ===")
    output_path = f"monitoring/{version}/train_stats.json"
    stats = calculate_train_stats(
        train_data_path=train_data_path,
        output_path=output_path,
        exclude_cols=["target"]
    )
    logger.info(f"✓ Statistiques calculées pour {len(stats.get('statistics', {}).get('mean', {}))} features")
    logger.info(f"✓ Fichier sauvegardé: {output_path}")
    print("\n" + "="*80)
    print(f"SETUP MONITORING TERMINÉ - {version}")
    print("="*80)
    print(f"Statistiques disponibles dans: {output_path}")
    print(f"Vous pouvez maintenant utiliser 'monitor' pour détecter le drift")
    print("="*80 + "\n")


def run_monitoring(prod_data_path: str, version: str = "v1", threshold_std: float = 2.0, method: str = "advanced") -> dict:
    logger.info(f"=== Monitoring en cours pour {version} ===")
    train_stats_path = f"monitoring/{version}/train_stats.json"
    output_dir = f"monitoring/{version}/drift_reports"
    if not Path(train_stats_path).exists():
        logger.error(f"Statistiques d'entraînement non trouvées: {train_stats_path}")
        logger.error("Veuillez d'abord exécuter 'setup' pour cette version")
        sys.exit(1)
    # Détection de drift simple
    results = detect_data_drift(
        prod_data_path=prod_data_path,
        train_stats_path=train_stats_path,
        output_dir=output_dir,
        threshold_std=threshold_std,
        method=method
    )
    # Ajout historique et alertes par version
    from monitoring_alerts import MonitoringHistory, AlertSystem
    import datetime
    # Ajoute timestamp/version si absent
    if "timestamp" not in results:
        results["timestamp"] = datetime.datetime.now().isoformat()
    results["version"] = version
    # Historique par version
    history = MonitoringHistory(version)
    history.add_entry(results)
    # Alertes par version
    alert_system = AlertSystem(version)
    alert_status = alert_system.check_thresholds(results, history.history)
    alert_system.send_alert(alert_status, results)
    results["alert_status"] = alert_status
    return results


def show_history(version: str = "v1") -> None:
    history_path = f"monitoring/history/drift_history.json"
    history = MonitoringHistory(history_path)
    df = history.get_history_df()
    if df is None or df.empty:
        print("Aucun historique disponible")
        return
    print("\n" + "="*80)
    print("HISTORIQUE DU MONITORING")
    print("="*80)
    print(df.to_string())
    print("="*80 + "\n")
    # Analyse de tendances
    trends = history.get_trend_analysis()
    if trends.get("status") != "insufficient_data":
        print("\nANALYSE DES TENDANCES")
        print("-"*80)
        print(f"Nombre d'entrées: {trends.get('total_entries', 0)}")
        if "date_range" in trends:
            print(f"Période: {trends['date_range'].get('first')} → {trends['date_range'].get('last')}")
        if "drift_trend" in trends:
            drift = trends["drift_trend"]
            print(f"\nFeatures avec drift:")
            print(f"  Moyenne: {drift.get('mean', 0):.2f}")
            print(f"  Écart-type: {drift.get('std', 0):.2f}")
            print(f"  Min: {drift.get('min', 0)}, Max: {drift.get('max', 0)}")
            print(f"  Dernière valeur: {drift.get('last_value', 0)}")
        print("="*80 + "\n")


def monitor_model_metrics(baseline_metrics_path: str, current_metrics_path: str, version: str = "v1", threshold: float = 0.1) -> dict:
    logger.info(f"=== Monitoring des métriques pour {version} ===")
    output_dir = f"monitoring/{version}/metrics_reports"
    results = monitor_metrics(
        baseline_metrics_path=baseline_metrics_path,
        current_metrics_path=current_metrics_path,
        output_dir=output_dir,
        threshold_degradation=threshold
    )
    history = MetricsHistory()
    history.add_entry(results)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Système de monitoring MLOps pour la détection de data drift",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:

  # 1. Configuration initiale (calculer les statistiques d'entraînement)
  python src/monitoring.py setup --train-data data/processed/v1/train.csv --version v1

  # 2. Détecter le drift sur les données de test
  python src/monitoring.py monitor --prod-data data/processed/v1/test.csv --version v1

  # 3. Détecter le drift avec un seuil personnalisé
  python src/monitoring.py monitor --prod-data data/processed/v1/test.csv --version v1 --threshold 1.5

  # 4. Afficher l'historique
  python src/monitoring.py history --version v1

  # 5. Monitorer les métriques de performance
  python src/monitoring.py metrics --baseline metrics/v1/train_metrics.json --current metrics/v1/test_metrics.json --version v1
        """
    )
    subparsers = parser.add_subparsers(dest="command", help="Commande à exécuter")
    setup_parser = subparsers.add_parser("setup", help="Configurer le monitoring (calculer les stats)")
    setup_parser.add_argument("--train-data", required=True, help="Chemin vers le dataset d'entraînement")
    setup_parser.add_argument("--version", default="v1", help="Version du modèle (v1, v2, v3)")
    monitor_parser = subparsers.add_parser("monitor", help="Exécuter le monitoring de drift")
    monitor_parser.add_argument("--prod-data", required=True, help="Chemin vers les données de production")
    monitor_parser.add_argument("--version", default="v1", help="Version du modèle")
    monitor_parser.add_argument("--threshold", type=float, default=2.0, help="Seuil en écarts-types")
    monitor_parser.add_argument("--method", choices=["simple", "advanced"], default="advanced")
    history_parser = subparsers.add_parser("history", help="Afficher l'historique du monitoring")
    history_parser.add_argument("--version", default="v1", help="Version du modèle")
    metrics_parser = subparsers.add_parser("metrics", help="Monitorer les métriques de performance")
    metrics_parser.add_argument("--baseline", required=True, help="Chemin vers les métriques de référence")
    metrics_parser.add_argument("--current", required=True, help="Chemin vers les métriques actuelles")
    metrics_parser.add_argument("--version", default="v1", help="Version du modèle")
    metrics_parser.add_argument("--threshold", type=float, default=0.1, help="Seuil de dégradation (0.1 = 10%)")
    args = parser.parse_args()
    if args.command == "setup":
        setup_monitoring(args.train_data, args.version)
    elif args.command == "monitor":
        results = run_monitoring(
            args.prod_data,
            args.version,
            args.threshold,
            args.method
        )
        print("\n" + "="*80)
        print("RÉSUMÉ DU MONITORING")
        print("="*80)
        alert = results.get("alert_status", {})
        print(f"Niveau d'alerte: {alert.get('level', 'N/A').upper()}")
        if alert.get('triggered_rules'):
            print("\nRègles déclenchées:")
            for rule in alert['triggered_rules']:
                print(f"  - {rule}")
        if alert.get('recommendations'):
            print("\nRecommandations:")
            for rec in alert['recommendations']:
                print(f"  - {rec}")
        print("="*80 + "\n")
    elif args.command == "history":
        show_history(args.version)
    elif args.command == "metrics":
        results = monitor_model_metrics(
            args.baseline,
            args.current,
            args.version,
            args.threshold
        )
        print("\n" + "="*80)
        print("RÉSUMÉ DU MONITORING DES MÉTRIQUES")
        print("="*80)
        summary = results.get("summary", {})
        print(f"Métriques analysées: {summary.get('total_metrics', 0)}")
        print(f"Métriques dégradées: {summary.get('degraded_metrics', 0)}")
        print(f"Métriques améliorées: {summary.get('improved_metrics', 0)}")
        print(f"Métriques stables: {summary.get('stable_metrics', 0)}")
        if summary.get('degraded_metrics', 0) > 0:
            print(f"\n  ATTENTION: {summary['degraded_metrics']} métrique(s) dégradée(s)!")
            print("Voir le rapport détaillé pour plus d'informations.")
        else:
            print("\n✓ Aucune dégradation des métriques détectée")
        print("="*80 + "\n")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

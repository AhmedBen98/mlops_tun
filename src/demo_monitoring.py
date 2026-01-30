"""
Script de démonstration du système de monitoring MLOps
Teste toutes les fonctionnalités: calcul de stats, détection de drift, alertes
"""

import sys
from pathlib import Path

# Ajouter le répertoire src au path
sys.path.insert(0, str(Path(__file__).parent))

from monitoring import setup_monitoring, run_monitoring, show_history

def main():
    print("""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    DÉMONSTRATION DU MONITORING MLOps                          ║
║                      Détection de Data Drift                                  ║
╚═══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Version à tester
    version = "v1"
    
    # Étape 1: Configuration du monitoring
    print("\n" + "="*80)
    print("ÉTAPE 1: CALCUL DES STATISTIQUES D'ENTRAÎNEMENT")
    print("="*80)
    
    train_data_path = f"../data/processed/{version}/train.csv"
    setup_monitoring(train_data_path, version)
    
    input("\nAppuyez sur Entrée pour continuer vers la détection de drift...")
    
    # Étape 2: Détection de drift sur les données de test
    print("\n" + "="*80)
    print("ÉTAPE 2: DÉTECTION DE DRIFT SUR LES DONNÉES DE TEST")
    print("="*80)
    
    test_data_path = f"../data/processed/{version}/test.csv"
    print(f"\nComparaison: {test_data_path} vs statistiques d'entraînement")
    
    results = run_monitoring(test_data_path, version, threshold_std=2.0)
    
    input("\nAppuyez sur Entrée pour voir l'historique...")
    
    # Étape 3: Afficher l'historique
    print("\n" + "="*80)
    print("ÉTAPE 3: HISTORIQUE DU MONITORING")
    print("="*80)
    
    show_history(version)
    
    # Étape 4: Test avec différents seuils
    print("\n" + "="*80)
    print("ÉTAPE 4: TEST AVEC UN SEUIL PLUS SENSIBLE (threshold=1.5)")
    print("="*80)
    
    results = run_monitoring(test_data_path, version, threshold_std=1.5)
    
    # Résumé final
    print("\n" + "="*80)
    print("DÉMONSTRATION TERMINÉE")
    print("="*80)
    print("\nFichiers générés:")
    print(f"  - monitoring/{version}/train_stats.json")
    print(f"  - monitoring/{version}/drift_reports/")
    print(f"  - monitoring/history/drift_history.json")
    print(f"  - monitoring/alerts/alerts.json")
    print(f"  - monitoring/alert_config.json")
    print("\nCommandes disponibles:")
    print(f"  - Setup:   python src/monitoring.py setup --train-data {train_data_path} --version {version}")
    print(f"  - Monitor: python src/monitoring.py monitor --prod-data {test_data_path} --version {version}")
    print(f"  - History: python src/monitoring.py history --version {version}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()

"""
Script de démonstration automatique
Exécute tout le pipeline MLOps pour les 3 versions du dataset
"""
import subprocess
import time
from pathlib import Path

def run_command(cmd, description):
    print(f"{'='*80}")
    print(f"{description}")
    print(f"Command: {cmd}")
    print(f"{'='*80}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Erreur: {result.stderr}")
        return False
    print(result.stdout)
    return True

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def step_1_generate_datasets():
    print("\n" + "="*80)
    print("ÉTAPE 1: GÉNÉRATION DES DATASETS")
    print("="*80 + "\n")
    cmd = "python src/generate_diabetes_versions.py --version all"
    return run_command(cmd, "Génération des 3 versions du dataset Diabetes")

def step_2_init_git_dvc():
    print("\n" + "="*80)
    print("ÉTAPE 2: INITIALISATION GIT ET DVC")
    print("="*80 + "\n")
    commands = [
        ("git init", "Initialisation du repository Git"),
        ("git config user.name 'MLOps Demo'", "Configuration Git user"),
        ("git config user.email 'demo@mlops.com'", "Configuration Git email"),
        ("dvc init", "Initialisation de DVC"),
        ("dvc remote add -d myremote /tmp/dvc-storage", "Ajout du remote DVC"),
    ]
    for cmd, desc in commands:
        if not run_command(cmd, desc):
            return False
    return True

def step_3_process_version(version_num, dataset_file):
    print("\n" + "="*80)
    print(f"ÉTAPE 3.{version_num}: TRAITEMENT VERSION {version_num}")
    print("="*80 + "\n")
    v = f"v{version_num}"
    ensure_dir(f"data/processed/{v}")
    ensure_dir(f"models/{v}")
    ensure_dir(f"metrics/{v}")
    ensure_dir(f"plots/{v}")
    commands = [
        (f"python src/data_processing.py --input {dataset_file} --output data/processed/{v}",
         f"Preprocessing - Version {version_num}"),
        (f"python src/feature_engineering.py --train data/processed/{v}/train.csv --val data/processed/{v}/val.csv --test data/processed/{v}/test.csv --output data/processed/{v} --dataset diabetes",
         f"Feature Engineering - Version {version_num}"),
        (f"python src/train.py --train data/processed/{v}/train_engineered.csv --val data/processed/{v}/val_engineered.csv --dataset diabetes --output models/{v}",
         f"Training - Version {version_num}"),
        (f"python src/evaluate.py --test data/processed/{v}/test_engineered.csv --model models/{v}/model.pkl --dataset diabetes",
         f"Evaluation - Version {version_num}"),
    ]
    for cmd, desc in commands:
        if not run_command(cmd, desc):
            return False
    ensure_dir(f"metrics/{v}")
    subprocess.run(f"cp metrics/train_metrics.json metrics/{v}/", shell=True)
    subprocess.run(f"cp metrics/test_metrics.json metrics/{v}/", shell=True)
    subprocess.run(f"cp -r plots/* plots/{v}/ 2>/dev/null || true", shell=True)
    return True

def step_4_version_control(version_num, dataset_file):
    print("\n" + "="*80)
    print(f"ÉTAPE 4.{version_num}: VERSIONING - VERSION {version_num}")
    print("="*80 + "\n")
    v = f"v{version_num}"
    commands = [
        (f"dvc add {dataset_file}", f"DVC add dataset - Version {version_num}"),
        (f"dvc add models/{v}/model.pkl", f"DVC add model - Version {version_num}"),
        (f"git add {dataset_file}.dvc models/{v}/model.pkl.dvc .gitignore", f"Git add DVC files - Version {version_num}"),
        (f"git add metrics/{v}/ plots/{v}/", f"Git add metrics - Version {version_num}"),
        (f"git commit -m 'Version {version_num}: Dataset and model'", f"Git commit - Version {version_num}"),
        (f"git tag {v}.0", f"Git tag - Version {version_num}"),
    ]
    for cmd, desc in commands:
        if not run_command(cmd, desc):
            return False
    return True

def step_5_compare_versions():
    print("\n" + "="*80)
    print("ÉTAPE 5: COMPARAISON DES VERSIONS")
    print("="*80 + "\n")
    cmd = ("python src/compare_versions.py " +
           "--v1-data data/raw/diabetes_v1_original.csv " +
           "--v2-data data/raw/diabetes_v2_augmented.csv " +
           "--v3-data data/raw/diabetes_v3_varied.csv " +
           "--v1-metrics metrics/v1/test_metrics.json " +
           "--v2-metrics metrics/v2/test_metrics.json " +
           "--v3-metrics metrics/v3/test_metrics.json " +
           "--output reports")
    return run_command(cmd, "Génération du rapport de comparaison")

def main():
    print("\n" + "="*80)
    print("DÉMARRAGE DE LA DÉMONSTRATION MLOPS COMPLÈTE")
    print("Diabetes Dataset - 3 Versions")
    print("="*80 + "\n")
    if not step_1_generate_datasets():
        return
    if not step_2_init_git_dvc():
        return
    versions = [
        (1, "data/raw/diabetes_v1_original.csv"),
        (2, "data/raw/diabetes_v2_augmented.csv"),
        (3, "data/raw/diabetes_v3_varied.csv")
    ]
    for version_num, dataset_file in versions:
        if not step_3_process_version(version_num, dataset_file):
            return
        if not step_4_version_control(version_num, dataset_file):
            return
    step_5_compare_versions()
    print("\n" + "="*80)
    print("DÉMO TERMINÉE")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()

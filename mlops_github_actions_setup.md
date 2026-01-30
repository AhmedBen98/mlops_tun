# Configuration CI/CD MLOps avec GitHub Actions, DVC et MLflow

## 1. Préparer le dépôt GitHub
- Vérifier que le dépôt contient :
  - Le code source (train.py, etc.)
  - Les fichiers DVC (.dvc, .dvcignore, dvc.yaml, dvc.lock, *.dvc)
  - Les scripts utiles (add_target_column.py, etc.)
  - Le dataset versionné par DVC (pas le CSV lui-même, mais le .dvc)
- Pousser tout sur GitHub :
  ```bash
  git add .
  git commit -m "Mise à jour MLOps avec DVC et pipeline"
  git push origin master
  ```

## 2. Créer le workflow GitHub Actions
- Fichier : `.github/workflows/ml-pipeline.yml`
- Exemple de contenu :

```yaml
name: MLOps Pipeline

on:
  push:
    branches: [ master ]
  pull_request:

jobs:
  build-and-train:
    runs-on: ubuntu-latest

    steps:
    - name: Checkout code
      uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install dvc[s3] mlflow scikit-learn joblib

    - name: Pull DVC data
      run: dvc pull

    - name: Run DVC pipeline
      run: dvc repro
```

- Adapter la ligne `pip install dvc[s3]` selon ton remote DVC.

## 3. Vérification du pipeline
- Modifier le dataset ou le code, puis pousser sur GitHub.
- Observer l’exécution automatique du pipeline dans l’onglet Actions.
- À chaque push, le workflow :
  - Installe Python et les dépendances
  - Récupère les données DVC
  - Exécute dvc repro (entraînement, génération du modèle, etc.)

## 4. Traçabilité avec MLflow
- S’assurer que le script d’entraînement loggue bien les métriques et le modèle dans MLflow.
- Après chaque exécution du pipeline, un nouveau run MLflow doit être créé (si le tracking MLflow est bien configuré).
- Pour comparer les métriques, lancer MLflow UI localement :
  ```bash
  mlflow ui
  ```
  puis ouvrir http://localhost:5000

---

**Ce document résume toutes les étapes pour configurer et exploiter un pipeline MLOps automatisé avec GitHub Actions, DVC et MLflow.**

# Guide de configuration MLOps : DVC, GitHub Actions, MLflow

## 1. Préparation du dépôt GitHub
- Créez un dépôt GitHub (ou utilisez un existant)
- Ajoutez tout le projet (code, .dvc, dvc.yaml, dvc.lock, .github, etc.)
- Ne versionnez pas les gros fichiers de données, seulement les .dvc
- Exemple de commandes :

```bash
git add .
git commit -m "Initialisation du projet MLOps avec DVC"
git push origin master
```

## 2. Configuration du workflow GitHub Actions
- Créez le fichier `.github/workflows/ml-pipeline.yml` (déjà généré)
- Ce workflow :
  - S'exécute à chaque push
  - Installe Python, DVC, scikit-learn, MLflow
  - Fait un `dvc pull` pour récupérer les données
  - Exécute `dvc repro` pour entraîner le modèle
  - Upload le modèle entraîné en artifact

## 3. Utilisation de DVC
- Ajoutez vos datasets avec `dvc add`
- Créez un pipeline avec `dvc stage add ...`
- Exécutez le pipeline avec `dvc repro`
- Les fichiers `dvc.yaml` et `dvc.lock` assurent la traçabilité

## 4. Intégration MLflow
- Le script `train.py` logue chaque run dans MLflow
- Les métriques et modèles sont sauvegardés dans le dossier `mlruns/` (ou sur un serveur MLflow si configuré)
- Après chaque exécution (locale ou CI), un nouveau run MLflow est créé

## 5. Bonnes pratiques
- Ne versionnez pas les gros fichiers de données, seulement les .dvc
- Utilisez des branches pour vos évolutions
- Documentez chaque étape dans le README
- Pour la CI/CD, vérifiez les logs GitHub Actions après chaque push

---

**Pour toute modification du dataset ou du code, poussez sur GitHub : le pipeline CI/CD s'exécutera automatiquement et la traçabilité sera assurée via DVC et MLflow.**

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle

import mlflow
import mlflow.sklearn

# Charger le dataset
DATA_PATH = "data/raw/mlops_dataset_v1_original.csv"
df = pd.read_csv(DATA_PATH)

# Cible : prédire le nombre de visiteurs
y = df["num_users"]

# Colonnes à utiliser comme features (on enlève city_name, num_users, description)
features = [
    'country', 'location_id', 'avg_age', 'std_age', 'min_age', 'max_age', 'admin_ratio',
    'region', 'desc_length', 'is_touristique', 'is_capitale', 'is_historique', 'is_moderne'
]
X = df[features]

# Colonnes catégorielles à encoder
cat_features = ['country', 'region']

# Pipeline d'encodage + modèle
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
], remainder="passthrough")

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(random_state=42))
])

with mlflow.start_run():
    pipeline.fit(X, y)
    mlflow.sklearn.log_model(pipeline, "model")
    score = pipeline.score(X, y)
    mlflow.log_metric("score", score)

    # Sauvegarder le modèle localement
    with open("model.pkl", "wb") as f:
        pickle.dump(pipeline, f)

print("Modèle entraîné, sauvegardé dans model.pkl et tracé avec MLflow.")

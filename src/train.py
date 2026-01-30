
"""
Training Script with MLflow Integration
Trains ML models and logs experiments to MLflow
Adapté pour la prédiction du nombre de visiteurs par ville
"""

import argparse
import pandas as pd
import time
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import numpy as np
from src.utils import setup_logging, load_params, ensure_dir

logger = setup_logging()

def get_features(df):
    # Exclure les colonnes non pertinentes pour l'entraînement
    exclude = ["city_name", "country", "description", "region", "visiteurs"]
    return [col for col in df.columns if col not in exclude]

def calculate_regression_metrics(y_true, y_pred):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred)
    }

def print_metrics(metrics, title):
    logger.info(f"--- {title} ---")
    for k, v in metrics.items():
        logger.info(f"{k}: {v:.4f}")

class ModelTrainer:
    """ML Model training with MLflow tracking"""
    def __init__(self, params: dict):
        self.params = params
        self.model = None
        self.model_type = 'regression'

    def get_model(self):
        model_params = self.params.get('model', {})
        self.model = RandomForestRegressor(
            n_estimators=model_params.get('n_estimators', 100),
            max_depth=model_params.get('max_depth', None),
            min_samples_split=model_params.get('min_samples_split', 2),
            min_samples_leaf=model_params.get('min_samples_leaf', 1),
            random_state=self.params['dataset']['random_state']
        )
        logger.info("Using RandomForestRegressor")
        return self.model

    def train(self, train_file: str, val_file: str, output_dir: str):
        logger.info("Loading training and validation data...")
        train_df = pd.read_csv(train_file)
        val_df = pd.read_csv(val_file)

        target_col = 'visiteurs'
        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]
        X_val = val_df.drop(columns=[target_col])
        y_val = val_df[target_col]

        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Validation data shape: {X_val.shape}")

        mlflow.set_tracking_uri(self.params.get('mlflow', {}).get('tracking_uri', 'mlruns'))
        experiment_name = "visiteurs-prediction-experiment"
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name="baseline-model"):
            logger.info("Starting MLflow run...")
            mlflow.log_param("n_samples_train", len(X_train))
            mlflow.log_param("n_samples_val", len(X_val))
            mlflow.log_param("n_features", X_train.shape[1])

            model = self.get_model()
            mlflow.log_params(model.get_params())

            logger.info("Training model...")
            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            logger.info(f"Training completed in {training_time:.2f} seconds")
            mlflow.log_metric("training_time_seconds", training_time)

            logger.info("Performing cross-validation...")
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=self.params.get('evaluation', {}).get('cv_folds', 5),
                scoring=self.params.get('evaluation', {}).get('scoring', 'neg_root_mean_squared_error')
            )
            mlflow.log_metric("cv_mean", cv_scores.mean())
            mlflow.log_metric("cv_std", cv_scores.std())
            logger.info(f"CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)

            train_metrics = calculate_regression_metrics(y_train, y_train_pred)
            val_metrics = calculate_regression_metrics(y_val, y_val_pred)


            for metric_name, metric_value in train_metrics.items():
                mlflow.log_metric(f"train_{metric_name}", metric_value)
            for metric_name, metric_value in val_metrics.items():
                mlflow.log_metric(f"val_{metric_name}", metric_value)

            print_metrics(train_metrics, "Training Metrics")
            print_metrics(val_metrics, "Validation Metrics")

            # Save train metrics to metrics/{version}/train_metrics.json
            import os
            version = os.path.basename(os.path.normpath(output_dir))
            metrics_dir = f"metrics/{version}"
            ensure_dir(metrics_dir)
            from src.utils import save_metrics
            save_metrics(train_metrics, f"{metrics_dir}/train_metrics.json")

            # Feature importance
            if hasattr(model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'feature': X_train.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                logger.info("Top 10 important features:")
                print(feature_importance.head(10))
                # Déduire la version à partir du dossier output_dir (ex: models/v1 → v1)
                import os
                version = os.path.basename(os.path.normpath(output_dir))
                plots_dir = f"plots/{version}"
                ensure_dir(plots_dir)
                feature_importance_path = f"{plots_dir}/feature_importance.csv"
                feature_importance.to_csv(feature_importance_path, index=False)
                mlflow.log_artifact(feature_importance_path)

            # Log model with signature
            signature = infer_signature(X_train, y_train_pred)
            mlflow.sklearn.log_model(
                model,
                "model",
                signature=signature,
                registered_model_name="visiteurs_model"
            )
            logger.info("Model logged to MLflow")

            # Save model locally
            ensure_dir(output_dir)
            model_path = f"{output_dir}/model.pkl"
            joblib.dump(model, model_path)
            logger.info(f"Model saved to {model_path}")

            # Get run ID
            run_id = mlflow.active_run().info.run_id
            logger.info(f"MLflow Run ID: {run_id}")
            return model

def main():
    parser = argparse.ArgumentParser(description='Train ML model')
    parser.add_argument('--train', type=str, default='data/processed/train_engineered.csv')
    parser.add_argument('--val', type=str, default='data/processed/val_engineered.csv')
    parser.add_argument('--output', type=str, default='models')
    parser.add_argument('--params', type=str, default='params.yaml')
    args = parser.parse_args()

    params = load_params(args.params)
    trainer = ModelTrainer(params)
    trainer.train(args.train, args.val, args.output)

if __name__ == "__main__":
    main()

"""
Hyperparameter Optimization with Optuna
Advanced feature: Automated hyperparameter tuning with MLflow integration
"""

import argparse
import pandas as pd
import numpy as np
import time
import joblib

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from src.utils import (
    setup_logging, load_params, ensure_dir, save_metrics,
    calculate_regression_metrics, print_metrics
)

logger = setup_logging()

class OptunaOptimizer:
    """Hyperparameter optimization using Optuna with MLflow tracking"""
    def __init__(self, params: dict, dataset_name: str):
        self.params = params
        self.dataset_name = dataset_name
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.best_model = None

    def load_data(self, train_file: str, val_file: str):
        logger.info("Loading data...")
        train_df = pd.read_csv(train_file)
        val_df = pd.read_csv(val_file)
        target_col = 'visiteurs' if 'visiteurs' in train_df.columns else train_df.columns[-1]
        self.X_train = train_df.drop(columns=[target_col])
        self.y_train = train_df[target_col]
        self.X_val = val_df.drop(columns=[target_col])
        self.y_val = val_df[target_col]
        logger.info(f"Loaded data - Train: {self.X_train.shape}, Val: {self.X_val.shape}")

    def objective(self, trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'random_state': 42
        }
        model = GradientBoostingRegressor(**params)
        cv_scores = cross_val_score(
            model, self.X_train, self.y_train,
            cv=self.params.get('evaluation', {}).get('cv_folds', 5),
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        rmse_scores = np.sqrt(-cv_scores)
        score = -rmse_scores.mean()  # Negative because Optuna maximizes
        with mlflow.start_run(nested=True):
            mlflow.log_params(params)
            mlflow.log_metric("cv_rmse", rmse_scores.mean())
            mlflow.log_metric("cv_std", rmse_scores.std())
        return score

    def optimize(self, train_file: str, val_file: str, output_dir: str):
        self.load_data(train_file, val_file)
        mlflow.set_tracking_uri(self.params['mlflow']['tracking_uri'])
        experiment_name = f"{self.dataset_name}-optuna-optimization"
        mlflow.set_experiment(experiment_name)
        direction = 'maximize'  # Maximizing negative RMSE
        with mlflow.start_run(run_name=f"{self.dataset_name}-optuna-study"):
            logger.info("Starting Optuna optimization...")
            sampler = TPESampler(seed=42)
            pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
            study = optuna.create_study(
                direction=direction,
                sampler=sampler,
                pruner=pruner
            )
            n_trials = self.params.get('optuna', {}).get('n_trials', 30)
            timeout = self.params.get('optuna', {}).get('timeout', None)
            start_time = time.time()
            study.optimize(
                self.objective,
                n_trials=n_trials,
                timeout=timeout,
                show_progress_bar=True
            )
            optimization_time = time.time() - start_time
            best_params = study.best_params
            best_score = study.best_value
            logger.info(f"Optimization completed in {optimization_time:.2f} seconds")
            logger.info(f"Best score: {best_score:.6f}")
            logger.info(f"Best parameters: {best_params}")
            mlflow.log_param("n_trials", len(study.trials))
            mlflow.log_param("optimization_time", optimization_time)
            mlflow.log_metric("best_score", best_score)
            mlflow.log_params(best_params)
            logger.info("Training final model with best parameters...")
            self.best_model = GradientBoostingRegressor(**best_params)
            self.best_model.fit(self.X_train, self.y_train)
            y_train_pred = self.best_model.predict(self.X_train)
            y_val_pred = self.best_model.predict(self.X_val)
            train_metrics = calculate_regression_metrics(self.y_train, y_train_pred)
            val_metrics = calculate_regression_metrics(self.y_val, y_val_pred)
            for metric_name, metric_value in train_metrics.items():
                mlflow.log_metric(f"train_{metric_name}", metric_value)
            for metric_name, metric_value in val_metrics.items():
                mlflow.log_metric(f"val_{metric_name}", metric_value)
            print_metrics(train_metrics, "Training Metrics (Best Model)")
            print_metrics(val_metrics, "Validation Metrics (Best Model)")
            ensure_dir(output_dir)
            model_path = f"{output_dir}/model_optimized.pkl"
            joblib.dump(self.best_model, model_path)
            logger.info(f"Best model saved to {model_path}")
            signature = infer_signature(self.X_train, y_train_pred)
            mlflow.sklearn.log_model(
                self.best_model,
                "optimized_model",
                signature=signature,
                registered_model_name=f"{self.dataset_name}_optimized_model"
            )
            optimization_results = {
                'dataset': self.dataset_name,
                'n_trials': len(study.trials),
                'best_score': float(best_score),
                'best_params': best_params,
                'optimization_time': optimization_time,
                **{f'train_{k}': v for k, v in train_metrics.items()},
                **{f'val_{k}': v for k, v in val_metrics.items()}
            }
            # Save results in versioned metrics directory (e.g., metrics/v1/optuna_results.json)
            version_metrics_dir = f"metrics/{output_dir}"
            ensure_dir(version_metrics_dir)
            optuna_results_path = f"{version_metrics_dir}/optuna_results.json"
            save_metrics(optimization_results, optuna_results_path)
            mlflow.log_artifact(optuna_results_path)
            study_path = f"{output_dir}/optuna_study.pkl"
            joblib.dump(study, study_path)
            mlflow.log_artifact(study_path)
            logger.info("Optimization complete!")
            return self.best_model, optimization_results

def main():
    parser = argparse.ArgumentParser(description='Optimize hyperparameters with Optuna')
    parser.add_argument('--train', type=str, default='data/processed/v1/train_engineered.csv')
    parser.add_argument('--val', type=str, default='data/processed/v1/val_engineered.csv')
    parser.add_argument('--output', type=str, default='v1')
    parser.add_argument('--params', type=str, default='params.yaml')
    parser.add_argument('--dataset', type=str, default='visiteurs')
    args = parser.parse_args()
    params = load_params(args.params)
    optimizer = OptunaOptimizer(params, args.dataset)
    optimizer.optimize(args.train, args.val, args.output)

if __name__ == "__main__":
    main()

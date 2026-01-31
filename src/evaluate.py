"""
Evaluation Script
Evaluates trained model on test set for visitor prediction project
"""

import argparse
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import utils
setup_logging = utils.setup_logging
load_params = utils.load_params
ensure_dir = utils.ensure_dir
save_metrics = utils.save_metrics
calculate_regression_metrics = utils.calculate_regression_metrics
print_metrics = utils.print_metrics

logger = setup_logging()

class ModelEvaluator:
    """Model evaluation with visualization for regression"""
    def __init__(self, params: dict):
        self.params = params

    def load_model(self, model_path: str):
        logger.info(f"Loading model from {model_path}")
        model = joblib.load(model_path)
        return model

    def plot_feature_importance(self, model, feature_names, output_path: str):
        if not hasattr(model, 'feature_importances_'):
            logger.warning("Model does not have feature_importances_ attribute")
            return
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(20)
        plt.figure(figsize=(10, 8))
        sns.barplot(data=feature_importance, x='importance', y='feature')
        plt.title('Top 20 Feature Importances')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Feature importance plot saved to {output_path}")
        # Save CSV as well
        csv_path = output_path.replace('.png', '.csv')
        feature_importance.to_csv(csv_path, index=False)
        logger.info(f"Feature importance CSV saved to {csv_path}")

    def plot_regression_results(self, y_true, y_pred, output_path: str):
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        # Scatter plot: Predicted vs Actual
        axes[0].scatter(y_true, y_pred, alpha=0.5)
        axes[0].plot([y_true.min(), y_true.max()],
                     [y_true.min(), y_true.max()],
                     'r--', lw=2)
        axes[0].set_xlabel('Actual Values')
        axes[0].set_ylabel('Predicted Values')
        axes[0].set_title('Predicted vs Actual')
        axes[0].grid(True, alpha=0.3)
        # Residual plot
        residuals = y_true - y_pred
        axes[1].scatter(y_pred, residuals, alpha=0.5)
        axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('Predicted Values')
        axes[1].set_ylabel('Residuals')
        axes[1].set_title('Residual Plot')
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Regression plots saved to {output_path}")

    def plot_prediction_distribution(self, y_true, y_pred, output_path: str):
        plt.figure(figsize=(10, 6))
        plt.hist(y_true, bins=30, alpha=0.5, label='Actual', color='blue')
        plt.hist(y_pred, bins=30, alpha=0.5, label='Predicted', color='red')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title('Distribution: Actual vs Predicted')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Distribution plot saved to {output_path}")

    def evaluate(self, test_file: str, model_path: str, output_dir: str = None):
        logger.info(f"Loading test data from {test_file}")
        import os
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        test_path = test_file if os.path.isabs(test_file) else os.path.join(project_root, test_file)
        model_path_abs = model_path if os.path.isabs(model_path) else os.path.join(project_root, model_path)
        test_df = pd.read_csv(test_path)
        # Set output directories
        if output_dir:
            plots_dir = f"plots/{output_dir}"
            metrics_dir = f"metrics/{output_dir}"
        else:
            plots_dir = "plots"
            metrics_dir = "metrics"
        ensure_dir(plots_dir)
        ensure_dir(metrics_dir)
        # Target column
        target_col = 'visiteurs' if 'visiteurs' in test_df.columns else test_df.columns[-1]
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]
        logger.info(f"Test data shape: {X_test.shape}")
        # Load model
        model = self.load_model(model_path_abs)
        # Make predictions
        logger.info("Making predictions...")
        y_pred = model.predict(X_test)
        # Calculate metrics
        test_metrics = calculate_regression_metrics(y_test, y_pred)
        print_metrics(test_metrics, "Test Set Metrics")
        # Generate plots
        self.plot_regression_results(y_test.values, y_pred, f"{plots_dir}/regression_results.png")
        self.plot_feature_importance(model, X_test.columns, f"{plots_dir}/feature_importance.png")
        self.plot_prediction_distribution(y_test.values, y_pred, f"{plots_dir}/prediction_distribution.png")
        # Save predictions
        predictions_df = pd.DataFrame({
            'actual': y_test.values,
            'predicted': y_pred
        })
        predictions_df.to_csv(f"{metrics_dir}/predictions.csv", index=False)
        logger.info(f"Predictions saved to {metrics_dir}/predictions.csv")
        # Save metrics
        metrics_dict = {
            'n_samples_test': len(X_test),
            **test_metrics
        }
        save_metrics(metrics_dict, f"{metrics_dir}/test_metrics.json")
        logger.info("Evaluation complete!")
        return test_metrics

def main():
    parser = argparse.ArgumentParser(description='Evaluate ML model')
    parser.add_argument('--test', type=str, default='data/processed/test_engineered.csv')
    parser.add_argument('--model', type=str, default='models/model.pkl')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for plots and metrics (e.g., v1, v2, v3)')
    parser.add_argument('--params', type=str, default='mlops/params.yaml')
    args = parser.parse_args()
    import os
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    params_path = args.params if os.path.isabs(args.params) else os.path.join(project_root, args.params)
    params = load_params(params_path)
    evaluator = ModelEvaluator(params)
    evaluator.evaluate(args.test, args.model, args.output)

if __name__ == "__main__":
    main()

"""
Feature Engineering Script
Creates new features and performs feature selection
"""

import argparse
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression

import utils
setup_logging = utils.setup_logging
load_params = utils.load_params
ensure_dir = utils.ensure_dir

logger = setup_logging()

class FeatureEngineer:
    """Feature engineering pipeline"""

    def __init__(self, params: dict):
        self.params = params
        self.poly_features = None
        self.feature_selector = None
        self.selected_features = None

    def create_polynomial_features(self, X_train, X_val, X_test):
        if not self.params['feature_engineering']['polynomial_features']:
            logger.info("Polynomial features disabled")
            return X_train, X_val, X_test
        degree = self.params['feature_engineering']['degree']
        interaction_only = self.params['feature_engineering']['interaction_only']
        logger.info(f"Creating polynomial features (degree={degree}, interaction_only={interaction_only})")
        self.poly_features = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=False)
        X_train_poly = self.poly_features.fit_transform(X_train)
        X_val_poly = self.poly_features.transform(X_val)
        X_test_poly = self.poly_features.transform(X_test)
        feature_names = self.poly_features.get_feature_names_out(X_train.columns)
        X_train_poly = pd.DataFrame(X_train_poly, columns=feature_names, index=X_train.index)
        X_val_poly = pd.DataFrame(X_val_poly, columns=feature_names, index=X_val.index)
        X_test_poly = pd.DataFrame(X_test_poly, columns=feature_names, index=X_test.index)
        logger.info(f"Polynomial features created: {X_train_poly.shape[1]} features")
        return X_train_poly, X_val_poly, X_test_poly

    def create_domain_features(self, X):
        X_new = X.copy()
        # Exemple de feature de domaine pour ton projet
        if 'num_users' in X.columns and 'desc_length' in X.columns:
            X_new['users_desc_interaction'] = X['num_users'] * X['desc_length']
        if 'avg_age' in X.columns and 'admin_ratio' in X.columns:
            X_new['age_admin_interaction'] = X['avg_age'] * X['admin_ratio']
        logger.info(f"Created {len(X_new.columns) - len(X.columns)} domain-specific features")
        return X_new

    def select_features(self, X_train, y_train, X_val, X_test):
        if not self.params['feature_engineering']['feature_selection']:
            logger.info("Feature selection disabled")
            return X_train, X_val, X_test
        n_features = min(self.params['feature_engineering']['n_features_to_select'], X_train.shape[1])
        logger.info(f"Selecting top {n_features} features")
        self.feature_selector = SelectKBest(score_func=f_regression, k=n_features)
        X_train_selected = self.feature_selector.fit_transform(X_train, y_train)
        X_val_selected = self.feature_selector.transform(X_val)
        X_test_selected = self.feature_selector.transform(X_test)
        selected_mask = self.feature_selector.get_support()
        self.selected_features = X_train.columns[selected_mask].tolist()
        logger.info(f"Selected features: {self.selected_features}")
        X_train_selected = pd.DataFrame(X_train_selected, columns=self.selected_features, index=X_train.index)
        X_val_selected = pd.DataFrame(X_val_selected, columns=self.selected_features, index=X_val.index)
        X_test_selected = pd.DataFrame(X_test_selected, columns=self.selected_features, index=X_test.index)
        return X_train_selected, X_val_selected, X_test_selected

    def engineer(self, train_file, val_file, test_file, output_dir):
        logger.info("Loading processed data...")
        import os
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        train_path = train_file if os.path.isabs(train_file) else os.path.join(project_root, train_file)
        val_path = val_file if os.path.isabs(val_file) else os.path.join(project_root, val_file)
        test_path = test_file if os.path.isabs(test_file) else os.path.join(project_root, test_file)
        output_dir_abs = output_dir if os.path.isabs(output_dir) else os.path.join(project_root, output_dir)
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)
        target_col = 'visiteurs'
        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]
        X_val = val_df.drop(columns=[target_col])
        y_val = val_df[target_col]
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]
        X_train = self.create_domain_features(X_train)
        X_val = self.create_domain_features(X_val)
        X_test = self.create_domain_features(X_test)
        X_train, X_val, X_test = self.create_polynomial_features(X_train, X_val, X_test)
        X_train, X_val, X_test = self.select_features(X_train, y_train, X_val, X_test)
        ensure_dir(output_dir_abs)
        train_engineered = pd.concat([X_train, y_train], axis=1)
        val_engineered = pd.concat([X_val, y_val], axis=1)
        test_engineered = pd.concat([X_test, y_test], axis=1)
        train_engineered.to_csv(os.path.join(output_dir_abs, "train_engineered.csv"), index=False)
        val_engineered.to_csv(os.path.join(output_dir_abs, "val_engineered.csv"), index=False)
        test_engineered.to_csv(os.path.join(output_dir_abs, "test_engineered.csv"), index=False)
        logger.info(f"Feature engineering complete. Final shape: {X_train.shape}")
        logger.info(f"Engineered data saved to {output_dir_abs}")

def main():
    parser = argparse.ArgumentParser(description='Feature engineering')
    parser.add_argument('--train', type=str, default='mlops/data/processed/train.csv')
    parser.add_argument('--val', type=str, default='mlops/data/processed/val.csv')
    parser.add_argument('--test', type=str, default='mlops/data/processed/test.csv')
    parser.add_argument('--output', type=str, default='mlops/data/processed')
    parser.add_argument('--params', type=str, default='mlops/params.yaml')
    args = parser.parse_args()
    import os
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    params_path = args.params if os.path.isabs(args.params) else os.path.join(project_root, args.params)
    params = load_params(params_path)
    engineer = FeatureEngineer(params)
    engineer.engineer(args.train, args.val, args.test, args.output)

if __name__ == "__main__":
    main()

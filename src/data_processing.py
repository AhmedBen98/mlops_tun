"""
Data Processing Script
Handles data cleaning, preprocessing, and splitting
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
import joblib

import utils
setup_logging = utils.setup_logging
load_params = utils.load_params
ensure_dir = utils.ensure_dir

logger = setup_logging()

class DataProcessor:
    """Data preprocessing pipeline"""

    def __init__(self, params: dict):
        self.params = params
        self.scaler = None
        self.label_encoder = None

    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load dataset from CSV"""
        logger.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df)} samples with {len(df.columns)} columns")
        return df

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values"""
        missing_count = df.isnull().sum().sum()

        if missing_count > 0:
            logger.info(f"Found {missing_count} missing values")
            strategy = self.params['preprocessing']['handle_missing']

            numeric_cols = df.select_dtypes(include=[np.number]).columns

            if strategy == 'mean':
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
            elif strategy == 'median':
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            elif strategy == 'drop':
                df = df.dropna()

            logger.info(f"Missing values handled using {strategy} strategy")

        return df

    def remove_outliers(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Remove outliers using IQR method or Z-score"""
        original_size = len(df)
        threshold = self.params['preprocessing']['outlier_threshold']

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != target_col]

        # Z-score method
        z_scores = np.abs((df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std())
        df = df[(z_scores < threshold).all(axis=1)]

        removed = original_size - len(df)
        logger.info(f"Removed {removed} outliers ({removed/original_size*100:.2f}%)")

        return df

    def encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables"""
        categorical_cols = df.select_dtypes(include=['object']).columns

        if len(categorical_cols) > 0:
            logger.info(f"Encoding categorical columns: {list(categorical_cols)}")

            for col in categorical_cols:
                # Label encoding for categorical variables
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))

        return df

    def split_features_target(self, df: pd.DataFrame) -> tuple:
        """Split features and target variable"""
        # Determine target column based on dataset
        if 'quality' in df.columns:
            target_col = 'quality'
        elif 'target' in df.columns:
            target_col = 'target'
        elif 'MedHouseVal' in df.columns:
            target_col = 'MedHouseVal'
        else:
            # Assume last column is target
            target_col = df.columns[-1]

        logger.info(f"Target column: {target_col}")

        X = df.drop(columns=[target_col])
        y = df[target_col]

        return X, y, target_col

    def scale_features(self, X_train: pd.DataFrame, X_val: pd.DataFrame,
                       X_test: pd.DataFrame) -> tuple:
        """Scale features using specified scaler"""
        scaler_type = self.params['preprocessing']['scaler']

        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        elif scaler_type == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")

        logger.info(f"Scaling features using {scaler_type} scaler")

        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )

        X_val_scaled = pd.DataFrame(
            self.scaler.transform(X_val),
            columns=X_val.columns,
            index=X_val.index
        )

        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )

        return X_train_scaled, X_val_scaled, X_test_scaled

    def process(self, input_file: str, output_dir: str):
        """Main processing pipeline"""
        import os
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        input_path = input_file if os.path.isabs(input_file) else os.path.join(project_root, input_file)
        output_dir_abs = output_dir if os.path.isabs(output_dir) else os.path.join(project_root, output_dir)
        # Load data
        df = self.load_data(input_path)

        # Handle missing values
        df = self.handle_missing_values(df)

        # Encode categorical variables
        df = self.encode_categorical(df)

        # Split features and target
        X, y, target_col = self.split_features_target(df)

        # Remove outliers (before splitting to avoid data leakage in some cases)
        df_clean = pd.concat([X, y], axis=1)
        df_clean = self.remove_outliers(df_clean, target_col)
        X, y, _ = self.split_features_target(df_clean)

        # Split data
        test_size = self.params['dataset']['test_size']
        val_size = self.params['dataset']['val_size']
        random_state = self.params['dataset']['random_state']

        # First split: train+val and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Second split: train and val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state
        )

        logger.info(f"Train set: {len(X_train)} samples")
        logger.info(f"Validation set: {len(X_val)} samples")
        logger.info(f"Test set: {len(X_test)} samples")

        # Scale features
        X_train_scaled, X_val_scaled, X_test_scaled = self.scale_features(
            X_train, X_val, X_test
        )

        # Save processed data
        ensure_dir(output_dir)

        train_df = pd.concat([X_train_scaled, y_train], axis=1)
        val_df = pd.concat([X_val_scaled, y_val], axis=1)
        test_df = pd.concat([X_test_scaled, y_test], axis=1)

        train_df.to_csv(f"{output_dir}/train.csv", index=False)
        val_df.to_csv(f"{output_dir}/val.csv", index=False)
        test_df.to_csv(f"{output_dir}/test.csv", index=False)

        # Save scaler
        joblib.dump(self.scaler, os.path.join(output_dir_abs, "scaler.pkl"))

        logger.info(f"Processed data saved to {output_dir_abs}")


def main():
    """Main function"""

    parser = argparse.ArgumentParser(description='Process dataset')
    parser.add_argument('--input', type=str, default='mlops/data/raw/mlops_dataset_v3.csv',
                        help='Input dataset file')
    parser.add_argument('--output', type=str, default='mlops/data/processed',
                        help='Output directory')
    parser.add_argument('--params', type=str, default='mlops/params.yaml',
                        help='Parameters file')

    args = parser.parse_args()

    import os
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    params_path = args.params if os.path.isabs(args.params) else os.path.join(project_root, args.params)
    params = load_params(params_path)
    processor = DataProcessor(params)
    processor.process(args.input, args.output)


if __name__ == "__main__":
    main()

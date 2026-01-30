"""
Test suite for data processing module
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

import pytest
import pandas as pd
import numpy as np
from data_processing import DataProcessor


@pytest.fixture
def sample_data():
    """Create sample dataset for testing"""
    np.random.seed(42)
    data = {
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'feature3': np.random.randn(100),
        'category': np.random.choice(['A', 'B', 'C'], 100),
        'target': np.random.randint(0, 10, 100)
    }
    return pd.DataFrame(data)


@pytest.fixture
def params():
    """Default parameters for testing"""
    return {
        'dataset': {
            'test_size': 0.2,
            'val_size': 0.1,
            'random_state': 42
        },
        'preprocessing': {
            'scaler': 'standard',
            'handle_missing': 'mean',
            'outlier_threshold': 3
        }
    }


def test_data_processor_init(params):
    """Test DataProcessor initialization"""
    processor = DataProcessor(params)
    assert processor.params == params
    assert processor.scaler is None


def test_handle_missing_values(sample_data, params):
    """Test missing value handling"""
    # Add missing values
    sample_data.loc[0:5, 'feature1'] = np.nan
    
    processor = DataProcessor(params)
    result = processor.handle_missing_values(sample_data)
    
    assert result['feature1'].isnull().sum() == 0


def test_encode_categorical(sample_data, params):
    """Test categorical encoding"""
    processor = DataProcessor(params)
    result = processor.encode_categorical(sample_data)
    
    assert result['category'].dtype in [np.int32, np.int64]


def test_split_features_target(sample_data, params):
    """Test feature-target split"""
    processor = DataProcessor(params)
    X, y, target_col = processor.split_features_target(sample_data)
    
    assert target_col == 'target'
    assert 'target' not in X.columns
    assert len(X) == len(y)


def test_scale_features(sample_data, params):
    """Test feature scaling"""
    processor = DataProcessor(params)
    
    # Split data first
    X, y, _ = processor.split_features_target(sample_data)
    X = X.select_dtypes(include=[np.number])
    
    # Create simple train/val/test split
    X_train = X[:60]
    X_val = X[60:80]
    X_test = X[80:]
    
    X_train_scaled, X_val_scaled, X_test_scaled = processor.scale_features(
        X_train, X_val, X_test
    )
    
    # Check that scaling was applied (mean ~0, std ~1 for train)
    assert abs(X_train_scaled.mean().mean()) < 0.1
    assert abs(X_train_scaled.std().mean() - 1) < 0.2

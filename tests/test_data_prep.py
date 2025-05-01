import pytest
from src.data_preparation import DataPreprocessor
import pandas as pd
import os
import numpy as np

@pytest.fixture
def sample_data(tmpdir):
    # Create test data
    data = {
        'Time': [0, 86400, 172800],  # 0, 24h, 48h
        'V1': [1.0, -1.0, 0.5],
        'V2': [0.5, -0.5, 0.2],
        'Amount': [10.0, 100.0, 50.0],
        'Class': [0, 1, 0]
    }
    df = pd.DataFrame(data)
    
    # Save to temp file
    file_path = os.path.join(tmpdir, "test_data.csv")
    df.to_csv(file_path, index=False)
    return file_path

def test_data_loading(sample_data):
    preprocessor = DataPreprocessor(sample_data)
    df = preprocessor.load_data()
    
    assert not df.empty
    assert 'hour_of_day' in df.columns
    assert 'day_of_week' in df.columns
    assert 'amount_log' in df.columns
    assert df.shape[0] == 3

def test_data_splitting(sample_data):
    preprocessor = DataPreprocessor(sample_data)
    preprocessor.load_data()
    X_train, X_test, y_train, y_test = preprocessor.split_data(test_size=0.33)
    
    assert len(X_train) == 2
    assert len(X_test) == 1
    assert isinstance(y_train, pd.Series)

def test_handle_imbalance(sample_data):
    preprocessor = DataPreprocessor(sample_data)
    preprocessor.load_data()
    X_train, X_test, y_train, y_test = preprocessor.split_data()
    X_res, y_res = preprocessor.handle_imbalance(X_train, y_train)
    
    assert len(X_res) > len(X_train)  # SMOTE should increase samples
    assert sum(y_res == 1) == sum(y_res == 0)  # Balanced classes
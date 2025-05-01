import pytest
import numpy as np
from src.prediction_pipeline import FraudDetectionPipeline
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
import joblib
import os
from xgboost import XGBClassifier

@pytest.fixture
def sample_model(tmpdir):
    # Create test model and scaler
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    scaler = StandardScaler().fit(X)
    model = XGBClassifier().fit(X, y)
    
    # Save to temp files
    model_path = os.path.join(tmpdir, "test_model.pkl")
    scaler_path = os.path.join(tmpdir, "test_scaler.pkl")
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    return model_path, scaler_path

def test_pipeline_initialization(sample_model):
    model_path, scaler_path = sample_model
    pipeline = FraudDetectionPipeline(model_path, scaler_path)
    
    assert pipeline.model is not None
    assert pipeline.scaler is not None

def test_single_prediction(sample_model):
    model_path, scaler_path = sample_model
    pipeline = FraudDetectionPipeline(model_path, scaler_path)
    
    input_data = {
        'Time': 1000,
        'V1': -1.2,
        'V2': 0.5,
        'V3': 1.0,
        'Amount': 250.0
    }
    
    result = pipeline.predict(input_data)
    
    assert 'prediction' in result
    assert 'probability' in result
    assert 0 <= result['probability'] <= 1

def test_batch_prediction(sample_model):
    model_path, scaler_path = sample_model
    pipeline = FraudDetectionPipeline(model_path, scaler_path)
    
    # Create test dataframe
    data = {
        'Time': [1000, 2000],
        'V1': [-1.2, 0.3],
        'V2': [0.5, -0.5],
        'V3': [1.0, 0.0],
        'Amount': [250.0, 50.0]
    }
    df = pd.DataFrame(data)
    
    results = pipeline.batch_predict(df)
    
    assert len(results) == 2
    assert 'probability' in results.columns
    assert 'is_fraud' in results.columns
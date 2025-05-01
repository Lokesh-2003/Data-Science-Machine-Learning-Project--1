import pytest
import numpy as np
from src.model_training import FraudDetectionModel
from sklearn.datasets import make_classification
import os
import joblib

@pytest.fixture
def sample_data():
    X, y = make_classification(
        n_samples=100,
        n_features=10,
        n_classes=2,
        weights=[0.9, 0.1],
        random_state=42
    )
    return X, y

def test_model_training(sample_data, tmpdir):
    X, y = sample_data
    model = FraudDetectionModel()
    trained_model = model.train_model(X, y)
    
    assert trained_model is not None
    assert hasattr(trained_model, 'predict')
    
    # Test model saving
    model_path = os.path.join(tmpdir, "test_model.pkl")
    model.save_model(model_path)
    assert os.path.exists(model_path)
    
    loaded_model = joblib.load(model_path)
    assert hasattr(loaded_model, 'predict')

def test_model_evaluation(sample_data):
    X, y = sample_data
    model = FraudDetectionModel()
    model.train_model(X, y)
    
    metrics = model.evaluate_model(X, y, save_plots=False)
    
    assert 'f1_score' in metrics
    assert 0 <= metrics['f1_score'] <= 1
    assert isinstance(metrics['confusion_matrix'], np.ndarray)
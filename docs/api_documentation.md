# Fraud Detection API Documentation

## Prediction Pipeline

### Class: `FraudDetectionPipeline`
```python
def __init__(self, model_path='models/fraud_detection_model.pkl', 
             scaler_path='models/scaler.pkl')
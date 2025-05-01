import joblib
import pandas as pd
import numpy as np
import yaml
import os
from typing import Union, Dict

class FraudDetectionPipeline:
    def __init__(self, 
                 model_path: str = 'models/fraud_detection_model.pkl', 
                 scaler_path: str = 'models/scaler.pkl',
                 config_path: str = 'config/params.yaml'):
        
        # Verify paths exist
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler file not found at {scaler_path}")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")
            
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
    def preprocess_input(self, input_data: Union[Dict, pd.DataFrame]) -> pd.DataFrame:
        """Preprocess input data for prediction"""
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        elif isinstance(input_data, pd.DataFrame):
            df = input_data.copy()
        else:
            raise ValueError("Input must be either a dictionary or pandas DataFrame")
            
        # Apply same transformations as training
        if 'Time' in df.columns:
            df['hour_of_day'] = df['Time'] % (24 * 3600) // 3600
            df['day_of_week'] = (df['Time'] // (24 * 3600)) % 7
        
        if 'Amount' in df.columns:
            df['amount_log'] = np.log1p(df['Amount'])
            df['amount_to_mean'] = df['Amount'] / df['Amount'].mean()
            
        # Ensure all expected features are present
        expected_features = self.model.feature_names_in_
        missing_features = set(expected_features) - set(df.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
            
        return df[expected_features]  # Return with correct feature order
    
    def predict(self, input_data: Union[Dict, pd.DataFrame], threshold: float = None) -> Dict:
        """Make fraud prediction with probability"""
        if threshold is None:
            threshold = self.config.get('prediction_threshold', 0.5)
        
        # Preprocess input
        processed_data = self.preprocess_input(input_data)
        
        # Scale features
        scaled_data = self.scaler.transform(processed_data)
        
        # Make prediction
        probability = self.model.predict_proba(scaled_data)[0, 1]
        prediction = 1 if probability >= threshold else 0
        
        return {
            'prediction': prediction,
            'probability': float(probability),
            'is_fraud': bool(prediction),
            'threshold': threshold,
            'features': processed_data.to_dict('records')[0]
        }
    
    def batch_predict(self, input_data: pd.DataFrame, threshold: float = None) -> pd.DataFrame:
        """Make predictions on multiple transactions"""
        if threshold is None:
            threshold = self.config.get('prediction_threshold', 0.5)
            
        processed_data = self.preprocess_input(input_data)
        scaled_data = self.scaler.transform(processed_data)
        probabilities = self.model.predict_proba(scaled_data)[:, 1]
        predictions = (probabilities >= threshold).astype(int)
        
        results = pd.DataFrame({
            'prediction': predictions,
            'probability': probabilities,
            'is_fraud': predictions.astype(bool)
        })
        
        return pd.concat([input_data.reset_index(drop=True), results], axis=1)
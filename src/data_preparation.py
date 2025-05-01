import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import joblib
import yaml
import os

class DataPreprocessor:
    def __init__(self, data_path='data/raw/creditcard.csv'):
        self.data_path = data_path
        self.df = None
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load and preprocess the data"""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found at {self.data_path}")
            
        self.df = pd.read_csv(self.data_path)
        
        # Basic preprocessing
        if 'Time' in self.df.columns:
            self.df['hour_of_day'] = self.df['Time'] % (24 * 3600) // 3600
            self.df['day_of_week'] = (self.df['Time'] // (24 * 3600)) % 7
        
        if 'Amount' in self.df.columns:
            self.df['amount_log'] = np.log1p(self.df['Amount'])
            self.df['amount_to_mean'] = self.df['Amount'] / self.df['Amount'].mean()
        
        return self.df
    
    def split_data(self, test_size=0.2, random_state=42):
        """Split data into train and test sets"""
        if self.df is None:
            self.load_data()
            
        X = self.df.drop('Class', axis=1)
        y = self.df['Class']
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    def handle_imbalance(self, X_train, y_train):
        """Apply SMOTE to handle class imbalance"""
        smote = SMOTE(random_state=42)
        return smote.fit_resample(X_train, y_train)
    
    def scale_features(self, X_train, X_test):
        """Scale features using StandardScaler"""
        self.scaler.fit(X_train)
        os.makedirs('models', exist_ok=True)
        joblib.dump(self.scaler, 'models/scaler.pkl')
        return self.scaler.transform(X_train), self.scaler.transform(X_test)
    
    def save_processed_data(self, X_train, X_test, y_train, y_test):
        """Save processed data for future use"""
        os.makedirs('data/processed', exist_ok=True)
        pd.concat([X_train, y_train], axis=1).to_csv('data/processed/train.csv', index=False)
        pd.concat([X_test, y_test], axis=1).to_csv('data/processed/test.csv', index=False)
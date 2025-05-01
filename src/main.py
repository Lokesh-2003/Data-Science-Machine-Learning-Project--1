from data_preparation import DataPreprocessor
from model_training import FraudDetectionModel
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import yaml

def main():
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('docs/plots', exist_ok=True)
    
    # Step 1: Data Preparation
    print("Step 1: Preparing data...")
    preprocessor = DataPreprocessor()
    
    try:
        df = preprocessor.load_data()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the data file exists at data/raw/creditcard.csv")
        return
    
    # Basic EDA
    print("\nClass Distribution:")
    print(df['Class'].value_counts())
    
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Class', data=df)
    plt.title('Class Distribution')
    plt.savefig('docs/plots/class_distribution.png')
    plt.close()
    
    # Split data
    X_train, X_test, y_train, y_test = preprocessor.split_data()
    
    # Handle imbalance
    X_train_smote, y_train_smote = preprocessor.handle_imbalance(X_train, y_train)
    
    # Scale features
    X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train_smote, X_test)
    
    # Save processed data
    preprocessor.save_processed_data(X_train_scaled, X_test_scaled, y_train_smote, y_test)
    
    # Step 2: Model Training
    print("\nStep 2: Training model...")
    model = FraudDetectionModel()
    model.train_model(X_train_scaled, y_train_smote)
    
    # Hyperparameter tuning
    print("\nPerforming hyperparameter tuning...")
    model.hyperparameter_tuning(X_train_scaled, y_train_smote)
    
    # Evaluation
    print("\nModel Evaluation:")
    metrics = model.evaluate_model(X_test_scaled, y_test)
    model.save_model()
    
    # Save metrics to config
    with open('config/params.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    config['metrics'] = {
        'f1_score': float(metrics['f1_score']),
        'roc_auc': float(metrics['roc_auc'])
    }
    
    with open('config/params.yaml', 'w') as f:
        yaml.safe_dump(config, f)
    
    print("\nFraud
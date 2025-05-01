# Fraud Detection System - Project Report

## Overview
This system detects fraudulent credit card transactions using machine learning.

## Architecture
1. **Data Pipeline**: 
   - Ingests transaction data
   - Handles class imbalance
   - Creates temporal features

2. **Model**:
   - XGBoost classifier
   - Hyperparameter tuned
   - Threshold adjustable

3. **Interface**:
   - Streamlit web app
   - Single and batch processing

## Performance
| Metric       | Score  |
|--------------|--------|
| F1 Score     | 0.85   |
| ROC-AUC      | 0.97   |
| Precision    | 0.82   |
| Recall       | 0.88   |

## Usage
```bash
# Train model
python src/main.py

# Run web app
streamlit run app/app.py
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, f1_score, roc_auc_score, confusion_matrix
import joblib
import yaml
import os
import matplotlib.pyplot as plt
import seaborn as sns

class FraudDetectionModel:
    def __init__(self):
        self.model = None
        self.best_params = None
        
    def train_model(self, X_train, y_train):
        """Train XGBoost model with default parameters"""
        with open('config/params.yaml') as f:
            params = yaml.safe_load(f)
        
        scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])
        
        self.model = XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            eval_metric='logloss',
            use_label_encoder=False,
            **params.get('model_params', {})
        )
        self.model.fit(X_train, y_train)
        return self.model
    
    def hyperparameter_tuning(self, X_train, y_train):
        """Perform hyperparameter tuning using GridSearchCV"""
        param_grid = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'gamma': [0, 0.1, 0.2]
        }
        
        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            cv=3,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        self.best_params = grid_search.best_params_
        self.model = grid_search.best_estimator_
        return self.model
    
    def evaluate_model(self, X_test, y_test, save_plots=True):
        """Evaluate model performance with visualizations"""
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Print metrics
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        print(f"\nF1 Score: {f1_score(y_test, y_pred):.4f}")
        print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
        
        # Generate plots
        if save_plots:
            os.makedirs('docs/plots', exist_ok=True)
            
            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(6,6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.savefig('docs/plots/confusion_matrix.png')
            plt.close()
            
            # Feature Importance
            feat_importances = pd.Series(self.model.feature_importances_, index=X_test.columns)
            plt.figure(figsize=(10,6))
            feat_importances.nlargest(10).plot(kind='barh')
            plt.title('Top 10 Feature Importance')
            plt.savefig('docs/plots/feature_importance.png')
            plt.close()
        
        return {
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba),
            'confusion_matrix': cm
        }
    
    def save_model(self, path='models/fraud_detection_model.pkl'):
        """Save the trained model"""
        os.makedirs('models', exist_ok=True)
        joblib.dump(self.model, path)
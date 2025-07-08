"""
Classification Models for Myeloproliferative Neoplasm (MPN) Diagnosis

This module implements classification models for diagnosing MPN subtypes
including Essential Thrombocythemia (ET), Polycythemia Vera (PV), and 
Primary Myelofibrosis (PMF) based on clinical and morphological features.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from typing import Dict, Any, Optional, List, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class MPNClassifier:
    """
    Myeloproliferative Neoplasm Classifier
    
    This class provides both binary classification (Lymphoma vs Non-Lymphoma)
    and multi-class classification (ET, PV, PMF) capabilities using various
    machine learning algorithms.
    """
    
    # Label mappings
    BINARY_LABELS = {0: "Control", 1: "MPN"}
    MPN_LABELS = {0: "ET", 1: "PV", 2: "PMF"}
    
    def __init__(self, 
                 model_type: str = "decision_tree",
                 binary_mode: bool = False):
        """
        Initialize the MPN classifier
        
        Args:
            model_type (str): Type of classifier to use 
                ('decision_tree', 'random_forest', 'logistic_regression', 'svm', 'xgboost')
            binary_mode (bool): Whether to use binary classification mode
        """
        self.model_type = model_type
        self.binary_mode = binary_mode
        self.model = None
        self.feature_names = None
        self.is_trained = False
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the specified model type"""
        if self.model_type == "decision_tree":
            self.model = DecisionTreeClassifier(
                random_state=42,
                max_depth=5,
                min_samples_split=5,
                min_samples_leaf=2,
                criterion='entropy'
            )
        elif self.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2
            )
        elif self.model_type == "logistic_regression":
            self.model = LogisticRegression(
                random_state=42,
                max_iter=1000,
                solver='liblinear'
            )
        elif self.model_type == "svm":
            self.model = SVC(
                random_state=42,
                probability=True,
                kernel='rbf',
                C=1.0
            )
        elif self.model_type == "xgboost":
            self.model = xgb.XGBClassifier(
                random_state=42,
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def train(self, 
              X: Union[np.ndarray, pd.DataFrame], 
              y: Union[np.ndarray, pd.Series],
              feature_names: Optional[List[str]] = None,
              test_size: float = 0.2,
              use_grid_search: bool = False) -> Dict[str, Any]:
        """
        Train the classifier
        
        Args:
            X: Feature matrix
            y: Target labels
            feature_names: Names of features
            test_size: Proportion of test set
            use_grid_search: Whether to use grid search for hyperparameter tuning
            
        Returns:
            Dict containing training results and metrics
        """
        try:
            # Convert to numpy arrays if needed
            if isinstance(X, pd.DataFrame):
                if feature_names is None:
                    feature_names = X.columns.tolist()
                X = X.values
            if isinstance(y, pd.Series):
                y = y.values
            
            self.feature_names = feature_names
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            
            # Train model
            if use_grid_search:
                param_grid = self._get_param_grid()
                grid_search = GridSearchCV(
                    self.model, param_grid, cv=5, scoring='accuracy', n_jobs=-1
                )
                grid_search.fit(X_train, y_train)
                self.model = grid_search.best_estimator_
                best_params = grid_search.best_params_
            else:
                self.model.fit(X_train, y_train)
                best_params = None
            
            # Evaluate model
            y_pred = self.model.predict(X_test)
            y_pred_proba = self.model.predict_proba(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            class_report = classification_report(y_test, y_pred, output_dict=True)
            
            # Calculate AUC for binary classification
            auc_score = None
            if self.binary_mode and hasattr(self.model, 'predict_proba'):
                auc_score = roc_auc_score(y_test, y_pred_proba[:, 1])
            
            self.is_trained = True
            
            results = {
                'accuracy': accuracy,
                'classification_report': class_report,
                'auc_score': auc_score,
                'best_params': best_params,
                'feature_names': self.feature_names,
                'n_train_samples': len(X_train),
                'n_test_samples': len(X_test)
            }
            
            logger.info(f"Model trained successfully. Test accuracy: {accuracy:.3f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise
    
    def _get_param_grid(self) -> Dict[str, List]:
        """Get parameter grid for grid search based on model type"""
        if self.model_type == "decision_tree":
            return {
                'max_depth': [3, 5, 10, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'criterion': ['gini', 'entropy']
            }
        elif self.model_type == "random_forest":
            return {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        elif self.model_type == "logistic_regression":
            return {
                'C': [0.1, 1.0, 10.0],
                'solver': ['liblinear', 'lbfgs'],
                'max_iter': [1000, 2000]
            }
        elif self.model_type == "svm":
            return {
                'C': [0.1, 1.0, 10.0],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']
            }
        elif self.model_type == "xgboost":
            return {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 10],
                'learning_rate': [0.01, 0.1, 0.2]
            }
        else:
            return {}
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> Dict[str, Any]:
        """
        Make predictions on input data
        
        Args:
            X: Feature matrix
            
        Returns:
            Dict containing predictions and probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        try:
            # Convert to numpy array if needed
            if isinstance(X, pd.DataFrame):
                X = X.values
            
            # Ensure correct shape
            if X.ndim == 1:
                X = X.reshape(1, -1)
            
            # Make predictions
            predictions = self.model.predict(X)
            probabilities = self.model.predict_proba(X)
            
            # Convert predictions to labels
            if self.binary_mode:
                predicted_labels = [self.BINARY_LABELS[pred] for pred in predictions]
            else:
                predicted_labels = [self.MPN_LABELS[pred] for pred in predictions]
            
            results = {
                'predictions': predictions.tolist(),
                'predicted_labels': predicted_labels,
                'probabilities': probabilities.tolist(),
                'max_probabilities': np.max(probabilities, axis=1).tolist()
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise
    
    def predict_single(self, features: Union[List, np.ndarray]) -> Dict[str, Any]:
        """
        Make prediction for a single sample
        
        Args:
            features: Feature vector for single sample
            
        Returns:
            Dict containing prediction result
        """
        if not isinstance(features, (list, np.ndarray)):
            raise ValueError("Features must be a list or numpy array")
        
        features = np.array(features).reshape(1, -1)
        result = self.predict(features)
        
        return {
            'prediction': result['predictions'][0],
            'predicted_label': result['predicted_labels'][0],
            'probability': result['max_probabilities'][0],
            'all_probabilities': result['probabilities'][0]
        }
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance scores
        
        Returns:
            Dict mapping feature names to importance scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        
        if hasattr(self.model, 'feature_importances_'):
            importance_scores = self.model.feature_importances_
            
            if self.feature_names:
                return dict(zip(self.feature_names, importance_scores))
            else:
                return dict(enumerate(importance_scores))
        else:
            logger.warning(f"Model type {self.model_type} does not support feature importance")
            return None
    
    def save(self, filepath: str):
        """Save the trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'binary_mode': self.binary_mode,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'MPNClassifier':
        """Load a saved model"""
        try:
            model_data = joblib.load(filepath)
            
            classifier = cls(
                model_type=model_data['model_type'],
                binary_mode=model_data['binary_mode']
            )
            
            classifier.model = model_data['model']
            classifier.feature_names = model_data['feature_names']
            classifier.is_trained = model_data['is_trained']
            
            logger.info(f"Model loaded from {filepath}")
            return classifier
            
        except Exception as e:
            logger.error(f"Error loading model from {filepath}: {str(e)}")
            raise


def create_mpn_classifier(model_type: str = "decision_tree", binary_mode: bool = False) -> MPNClassifier:
    """
    Convenience function to create an MPN classifier
    
    Args:
        model_type: Type of classifier to use
        binary_mode: Whether to use binary classification
        
    Returns:
        Initialized MPNClassifier instance
    """
    return MPNClassifier(model_type=model_type, binary_mode=binary_mode)
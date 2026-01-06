# src/classifier.py - FIXED VERSION

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC  # Fixed import
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
from typing import Dict, List, Tuple
import os

class DeepfakeClassifier:
    def __init__(self, model_type: str = 'random_forest'):
        """
        Initialize classifier
        model_type: 'random_forest', 'svm', or 'logistic_regression'
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the selected model"""
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'svm':
            self.model = SVC(  # Fixed: was SVM, should be SVC
                kernel='rbf',
                C=1.0,
                gamma='scale',
                random_state=42,
                probability=True
            )
        elif self.model_type == 'logistic_regression':
            self.model = LogisticRegression(
                random_state=42,
                max_iter=1000
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def prepare_features(self, features_list: List[Dict]) -> np.ndarray:
        """Convert list of feature dictionaries to numpy array"""
        if not features_list:
            raise ValueError("Features list is empty")
        
        # Get feature names from first sample
        if self.feature_names is None:
            self.feature_names = list(features_list[0].keys())
        
        # Convert to DataFrame for easier handling
        df = pd.DataFrame(features_list)
        
        # Ensure all required features are present
        for feature in self.feature_names:
            if feature not in df.columns:
                df[feature] = 0.0
        
        # Select only the required features in correct order
        df = df[self.feature_names]
        
        # Handle missing values and inf values
        df = df.fillna(0)
        df = df.replace([np.inf, -np.inf], 0)
        
        return df.values
    
    def train(self, features_list: List[Dict], labels: List[int]) -> Dict:
        """
        Train the classifier
        labels: 0 for real, 1 for fake
        """
        print(f"Training {self.model_type} classifier...")
        
        # Prepare features
        X = self.prepare_features(features_list)
        y = np.array(labels)
        
        print(f"Training data shape: {X.shape}")
        print(f"Class distribution - Real: {np.sum(y == 0)}, Fake: {np.sum(y == 1)}")
        
        # Check if we have enough samples
        if len(X) < 4:
            raise ValueError("Need at least 4 samples for training")
        
        # Split data
        test_size = min(0.2, 2.0 / len(X))  # Adjust test size for small datasets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate on test set
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation (adjust cv for small datasets)
        cv_folds = min(5, len(X_train))
        if cv_folds < 2:
            cv_scores = np.array([accuracy])  # Use test accuracy if too few samples
        else:
            cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=cv_folds)
        
        results = {
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        print(f"Training completed!")
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"CV Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return results
    
    def predict(self, features: Dict) -> Tuple[int, float]:
        """
        Predict if video is fake or real
        Returns: (prediction, confidence)
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Prepare features
        X = self.prepare_features([features])
        X_scaled = self.scaler.transform(X)
        
        # Make prediction
        prediction = self.model.predict(X_scaled)[0]
        probabilities = self.model.predict_proba(X_scaled)[0]
        confidence = np.max(probabilities)
        
        return int(prediction), float(confidence)
    
    def save_model(self, filepath: str):
        """Save trained model and scaler"""
        if self.model is None:
            raise ValueError("No trained model to save")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_type': self.model_type
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model and scaler"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.model_type = model_data['model_type']
        
        print(f"Model loaded from {filepath}")
    
    def get_feature_importance(self) -> Dict:
        """Get feature importance (for tree-based models)"""
        if self.model_type == 'random_forest' and self.model is not None:
            importance = self.model.feature_importances_
            feature_importance = dict(zip(self.feature_names, importance))
            return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        else:
            return {}
"""
Random Forest Phishing Detector Model
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib
from feature_extractor import EnhancedURLFeatureExtractor


class RandomForestPhishingDetector:
    """Random Forest with Red Flag Override System"""
    
    def __init__(self, n_estimators=200, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features='sqrt', random_state=42,
                 red_flag_threshold=0.25, n_jobs=-1):
        """
        Parameters:
        -----------
        n_estimators : int
            Number of trees in the forest (default: 200)
        max_depth : int or None
            Maximum depth of trees (None = unlimited)
        min_samples_split : int
            Minimum samples required to split an internal node
        min_samples_leaf : int
            Minimum samples required at a leaf node
        max_features : str or int
            Number of features to consider for best split
        red_flag_threshold : float
            Red flag percentage for override (default: 0.25 = 25%)
        n_jobs : int
            Number of jobs to run in parallel (-1 = use all processors)
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.red_flag_threshold = red_flag_threshold
        self.n_jobs = n_jobs
        
        self.scaler = StandardScaler()
        self.classifier = None
        self.feature_names = None
        self.feature_extractor = EnhancedURLFeatureExtractor()
        
    def fit(self, X, y):
        """Train Random Forest model"""
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X_values = X.values
        else:
            self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            X_values = X
        
        # Standardize
        X_scaled = self.scaler.fit_transform(X_values)
        
        # Train Random Forest
        self.classifier = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            bootstrap=True,
            oob_score=True,
            verbose=0
        )
        
        self.classifier.fit(X_scaled, y)
        
        return self
    
    def _calculate_red_flag_score(self, features_dict):
        """Calculate red flag score"""
        red_flag_features = [k for k in features_dict.keys() if k.startswith('red_flag_')]
        if not red_flag_features:
            return 0.0
        
        triggered = sum(1 for k in red_flag_features if features_dict[k] == 1)
        return triggered / len(red_flag_features)
    
    def predict(self, X):
        """Predict phishing (1) or legitimate (0)"""
        if isinstance(X, pd.DataFrame):
            X_values = X.values
        else:
            X_values = X
        X_scaled = self.scaler.transform(X_values)
        return self.classifier.predict(X_scaled)
    
    def predict_proba(self, X):
        """Predict probabilities"""
        if isinstance(X, pd.DataFrame):
            X_values = X.values
        else:
            X_values = X
        X_scaled = self.scaler.transform(X_values)
        return self.classifier.predict_proba(X_scaled)
    
    def predict_url(self, url, verbose=False):
        """Test any URL with Random Forest + Red Flags"""
        # Extract features
        features_dict = self.feature_extractor.extract_features(url)
        
        # Create DataFrame with features in correct order
        features_df = pd.DataFrame([features_dict])
        
        try:
            features_df = features_df[self.feature_names]
        except KeyError:
            # If feature names don't match, create properly ordered array
            feature_array = np.array([features_dict.get(name, 0) for name in self.feature_names]).reshape(1, -1)
            features_df = pd.DataFrame(feature_array, columns=self.feature_names)
        
        # Check red flags
        red_flags = self.feature_extractor.get_red_flag_explanation(features_dict)
        red_flag_score = self._calculate_red_flag_score(features_dict)
        
        # Get Random Forest prediction
        rf_prediction = self.predict(features_df.values)[0]
        rf_probabilities = self.predict_proba(features_df.values)[0]
        rf_confidence = float(rf_probabilities[1]) * 100
        
        # Combined decision
        if red_flag_score >= self.red_flag_threshold:
            final_prediction = 1
            final_confidence = max(rf_confidence, red_flag_score * 100)
            decision_method = "Red Flag Override"
        else:
            final_prediction = rf_prediction
            final_confidence = (rf_confidence * 0.7) + (red_flag_score * 100 * 0.3)
            decision_method = "Random Forest" if red_flag_score < 0.15 else "Random Forest + Red Flags"
        
        result = {
            'url': url,
            'prediction': 'PHISHING' if final_prediction == 1 else 'LEGITIMATE',
            'is_safe': final_prediction == 0,
            'confidence': {
                'legitimate': float(rf_probabilities[0]),
                'phishing': float(rf_probabilities[1])
            },
            'rf_confidence': rf_confidence,
            'red_flag_score': red_flag_score * 100,
            'final_confidence': final_confidence,
            'risk_score': final_confidence,
            'red_flags': red_flags,
            'red_flag_count': len(red_flags),
            'decision_method': decision_method
        }
        
        return result
    
    def save(self, filepath):
        """Save model to file"""
        model_data = {
            'classifier': self.classifier,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'red_flag_threshold': self.red_flag_threshold,
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'max_features': self.max_features,
            'random_state': self.random_state
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """Load model from file"""
        model_data = joblib.load(filepath)
        
        model = cls(
            n_estimators=model_data['n_estimators'],
            max_depth=model_data['max_depth'],
            min_samples_split=model_data['min_samples_split'],
            min_samples_leaf=model_data['min_samples_leaf'],
            max_features=model_data['max_features'],
            random_state=model_data['random_state'],
            red_flag_threshold=model_data['red_flag_threshold']
        )
        
        model.classifier = model_data['classifier']
        model.scaler = model_data['scaler']
        model.feature_names = model_data['feature_names']
        
        return model


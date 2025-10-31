"""
Train and Save Phishing Detection Models
"""

import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from random_forest_model import RandomForestPhishingDetector
from xgboost_model import XGBoostPhishingDetector
from data_loader import load_phiusiil_dataset


def train_models(dataset_path='url3.csv', models_dir='models'):
    """
    Train both Random Forest and XGBoost models and save them
    
    Parameters:
    -----------
    dataset_path : str
        Path to the dataset CSV file
    models_dir : str
        Directory to save trained models
    """
    # Create models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    
    # Load dataset
    print("Loading dataset...")
    X, y = load_phiusiil_dataset(dataset_path)
    
    if X is None or y is None:
        print("Error: Could not load dataset. Please check the file path.")
        return
    
    # Flip labels if needed (adjust based on your dataset)
    # Uncomment the line below if labels are reversed
    y = 1 - y
    
    # Split data
    print("\n" + "="*70)
    print("SPLITTING DATA")
    print("="*70)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {X_train.shape[1]}")
    
    # Train Random Forest Model
    print("\n" + "="*70)
    print("TRAINING RANDOM FOREST MODEL")
    print("="*70)
    
    rf_model = RandomForestPhishingDetector(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=42,
        red_flag_threshold=0.25,
        n_jobs=-1
    )
    
    rf_model.fit(X_train, y_train)
    
    # Evaluate Random Forest
    y_pred_rf = rf_model.predict(X_test.values)
    y_proba_rf = rf_model.predict_proba(X_test.values)[:, 1]
    
    print("\nRandom Forest Performance:")
    print(f"  Accuracy:  {accuracy_score(y_test, y_pred_rf):.4f}")
    print(f"  Precision: {precision_score(y_test, y_pred_rf):.4f}")
    print(f"  Recall:    {recall_score(y_test, y_pred_rf):.4f}")
    print(f"  F1-Score:  {f1_score(y_test, y_pred_rf):.4f}")
    print(f"  ROC-AUC:   {roc_auc_score(y_test, y_proba_rf):.4f}")
    
    # Save Random Forest model
    rf_path = os.path.join(models_dir, 'random_forest_model.pkl')
    rf_model.save(rf_path)
    
    # Train XGBoost Model
    print("\n" + "="*70)
    print("TRAINING XGBOOST MODEL")
    print("="*70)
    
    xgb_model = XGBoostPhishingDetector(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        red_flag_threshold=0.25
    )
    
    xgb_model.fit(X_train, y_train)
    
    # Evaluate XGBoost
    y_pred_xgb = xgb_model.predict(X_test.values)
    y_proba_xgb = xgb_model.predict_proba(X_test.values)[:, 1]
    
    print("\nXGBoost Performance:")
    print(f"  Accuracy:  {accuracy_score(y_test, y_pred_xgb):.4f}")
    print(f"  Precision: {precision_score(y_test, y_pred_xgb):.4f}")
    print(f"  Recall:    {recall_score(y_test, y_pred_xgb):.4f}")
    print(f"  F1-Score:  {f1_score(y_test, y_pred_xgb):.4f}")
    print(f"  ROC-AUC:   {roc_auc_score(y_test, y_proba_xgb):.4f}")
    
    # Save XGBoost model
    xgb_path = os.path.join(models_dir, 'xgboost_model.pkl')
    xgb_model.save(xgb_path)
    
    print("\n" + "="*70)
    print("✓ MODELS TRAINED AND SAVED SUCCESSFULLY!")
    print("="*70)
    print(f"Random Forest model saved to: {rf_path}")
    print(f"XGBoost model saved to: {xgb_path}")
    print("="*70)


if __name__ == "__main__":
    # Update this path to your dataset file
    dataset_path = 'url3.csv'
    
    # Train and save models
    train_models(dataset_path=dataset_path, models_dir='models')


"""
Train and Save Phishing Detection Models (Including Fuzzy Hybrid)
"""
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from skfuzzy import cmeans
from random_forest_model import RandomForestPhishingDetector
from xgboost_model import XGBoostPhishingDetector
from data_loader import load_phiusiil_dataset
import pickle
import time


class FuzzyRandomForestHybrid:
    """Fuzzy C-Means + Random Forest Hybrid Model"""

    def __init__(self, n_clusters=3, m=2.0, max_iter=150, error=1e-5,
                 n_estimators=200, max_depth=None, random_state=42,
                 red_flag_threshold=0.25):
        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.error = error
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.red_flag_threshold = red_flag_threshold

        self.scaler = StandardScaler()
        self.fuzzy_centers = None
        self.fuzzy_model = None
        self.classifier = None
        self.feature_names = None
        self.training_time = 0

    def fit(self, X, y):
        """Train the hybrid model"""
        print("=" * 70)
        print("TRAINING FUZZY C-MEANS + RANDOM FOREST HYBRID")
        print("=" * 70)

        start_time = time.time()

        if hasattr(X, 'columns'):
            self.feature_names = X.columns.tolist()
            X = X.values
        else:
            self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]

        print("\n[1/4] Standardizing features...")
        X_scaled = self.scaler.fit_transform(X)
        print(f"    ✓ Features standardized: {X.shape[1]} features")

        print(f"\n[2/4] Applying Fuzzy C-Means clustering (k={self.n_clusters})...")
        cntr, u, u0, d, jm, p, fpc = cmeans(
            X_scaled.T,
            c=self.n_clusters,
            m=self.m,
            error=self.error,
            maxiter=self.max_iter,
            init=None,
            seed=self.random_state
        )

        self.fuzzy_centers = cntr
        self.fuzzy_model = {'u': u, 'cntr': cntr, 'fpc': fpc}

        print(f"    ✓ Fuzzy Partition Coefficient (FPC): {fpc:.4f}")
        print(f"    ✓ Higher FPC (closer to 1.0) indicates better clustering")

        print("\n    Cluster Analysis:")
        for i in range(self.n_clusters):
            cluster_labels = y[u.argmax(axis=0) == i]
            if len(cluster_labels) > 0:
                phishing_pct = (cluster_labels == 1).mean() * 100
                print(f"    Cluster {i}: {len(cluster_labels)} samples, "
                      f"{phishing_pct:.1f}% phishing")

        print("\n[3/4] Adding fuzzy membership features...")
        X_fuzzy = self._add_fuzzy_features(X_scaled, u)

        fuzzy_feature_names = [f'fuzzy_membership_c{i}'
                               for i in range(self.n_clusters)]
        self.enhanced_feature_names = self.feature_names + fuzzy_feature_names

        print(f"    ✓ Original features: {X.shape[1]}")
        print(f"    ✓ Enhanced features: {X_fuzzy.shape[1]} "
              f"(+{self.n_clusters} fuzzy memberships)")

        print(f"\n[4/4] Training Random Forest on enhanced features...")

        from sklearn.ensemble import RandomForestClassifier

        self.classifier = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            random_state=self.random_state,
            n_jobs=-1
        )

        self.classifier.fit(X_fuzzy, y)

        y_pred = self.classifier.predict(X_fuzzy)
        train_acc = accuracy_score(y, y_pred)
        print(f"    ✓ Training accuracy: {train_acc:.4f}")

        self.training_time = time.time() - start_time

        print("\n" + "=" * 70)
        print("✓ FUZZY HYBRID TRAINING COMPLETE!")
        print(f"✓ Training Time: {self.training_time:.2f} seconds")
        print("=" * 70)

        return self

    def _add_fuzzy_features(self, X_scaled, u):
        """Combine original features with fuzzy memberships"""
        fuzzy_memberships = u.T
        return np.hstack([X_scaled, fuzzy_memberships])

    def _predict_fuzzy_membership(self, X_scaled):
        """Calculate fuzzy membership for new samples"""
        cntr = self.fuzzy_centers
        n_samples = X_scaled.shape[0]
        u = np.zeros((self.n_clusters, n_samples))

        for i in range(n_samples):
            distances = np.linalg.norm(X_scaled[i] - cntr, axis=1)
            distances = np.maximum(distances, 1e-10)

            for j in range(self.n_clusters):
                u[j, i] = 1.0 / np.sum((distances[j] / distances) ** (2 / (self.m - 1)))

        return u

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        u = self._predict_fuzzy_membership(X_scaled)
        X_fuzzy = self._add_fuzzy_features(X_scaled, u)
        return self.classifier.predict(X_fuzzy)

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        u = self._predict_fuzzy_membership(X_scaled)
        X_fuzzy = self._add_fuzzy_features(X_scaled, u)
        return self.classifier.predict_proba(X_fuzzy)

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"✓ Fuzzy Hybrid model saved to: {filepath}")

    @staticmethod
    def load(filepath):
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"✓ Fuzzy Hybrid model loaded from: {filepath}")
        return model


def train_models(dataset_path='url3.csv', models_dir='models'):
    """Train all three models: Random Forest, XGBoost, and Fuzzy+RF Hybrid"""

    os.makedirs(models_dir, exist_ok=True)

    print("Loading dataset...")
    X, y = load_phiusiil_dataset(dataset_path)

    if X is None or y is None:
        print("Error: Could not load dataset. Please check the file path.")
        return

    # Uncomment if labels need flipping
    # y = 1 - y

    print("\n" + "=" * 70)
    print("SPLITTING DATA (70:30)")
    print("=" * 70)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print(f"Training samples: {len(X_train):,}")
    print(f"Test samples: {len(X_test):,}")
    print(f"Features: {X_train.shape[1]}")

    all_results = {}

    # Train Random Forest
    print("\n" + "=" * 70)
    print("MODEL 1/3: TRAINING PURE RANDOM FOREST")
    print("=" * 70)

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

    y_pred_rf = rf_model.predict(X_test.values)
    y_proba_rf = rf_model.predict_proba(X_test.values)[:, 1]

    rf_results = {
        'accuracy': accuracy_score(y_test, y_pred_rf),
        'precision': precision_score(y_test, y_pred_rf),
        'recall': recall_score(y_test, y_pred_rf),
        'f1_score': f1_score(y_test, y_pred_rf),
        'roc_auc': roc_auc_score(y_test, y_proba_rf)
    }

    print("\nRandom Forest Performance:")
    print(f"  Accuracy:  {rf_results['accuracy']:.4f}")
    print(f"  Precision: {rf_results['precision']:.4f}")
    print(f"  Recall:    {rf_results['recall']:.4f}")
    print(f"  F1-Score:  {rf_results['f1_score']:.4f}")
    print(f"  ROC-AUC:   {rf_results['roc_auc']:.4f}")

    all_results['Random Forest'] = rf_results

    rf_path = os.path.join(models_dir, 'random_forest_model.pkl')
    rf_model.save(rf_path)

    # Train XGBoost
    print("\n" + "=" * 70)
    print("MODEL 2/3: TRAINING PURE XGBOOST")
    print("=" * 70)

    xgb_model = XGBoostPhishingDetector(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        red_flag_threshold=0.25
    )

    xgb_model.fit(X_train, y_train)

    y_pred_xgb = xgb_model.predict(X_test.values)
    y_proba_xgb = xgb_model.predict_proba(X_test.values)[:, 1]

    xgb_results = {
        'accuracy': accuracy_score(y_test, y_pred_xgb),
        'precision': precision_score(y_test, y_pred_xgb),
        'recall': recall_score(y_test, y_pred_xgb),
        'f1_score': f1_score(y_test, y_pred_xgb),
        'roc_auc': roc_auc_score(y_test, y_proba_xgb)
    }

    print("\nXGBoost Performance:")
    print(f"  Accuracy:  {xgb_results['accuracy']:.4f}")
    print(f"  Precision: {xgb_results['precision']:.4f}")
    print(f"  Recall:    {xgb_results['recall']:.4f}")
    print(f"  F1-Score:  {xgb_results['f1_score']:.4f}")
    print(f"  ROC-AUC:   {xgb_results['roc_auc']:.4f}")

    all_results['XGBoost'] = xgb_results

    xgb_path = os.path.join(models_dir, 'xgboost_model.pkl')
    xgb_model.save(xgb_path)

    # Train Fuzzy Hybrid
    print("\n" + "=" * 70)
    print("MODEL 3/3: TRAINING FUZZY + RANDOM FOREST HYBRID")
    print("=" * 70)

    fuzzy_rf_model = FuzzyRandomForestHybrid(
        n_clusters=3,
        m=2.0,
        n_estimators=200,
        max_depth=None,
        random_state=42,
        red_flag_threshold=0.25
    )

    fuzzy_rf_model.fit(X_train, y_train)

    y_pred_fuzzy = fuzzy_rf_model.predict(X_test.values)
    y_proba_fuzzy = fuzzy_rf_model.predict_proba(X_test.values)[:, 1]

    fuzzy_results = {
        'accuracy': accuracy_score(y_test, y_pred_fuzzy),
        'precision': precision_score(y_test, y_pred_fuzzy),
        'recall': recall_score(y_test, y_pred_fuzzy),
        'f1_score': f1_score(y_test, y_pred_fuzzy),
        'roc_auc': roc_auc_score(y_test, y_proba_fuzzy)
    }

    print("\nFuzzy + RF Hybrid Performance:")
    print(f"  Accuracy:  {fuzzy_results['accuracy']:.4f}")
    print(f"  Precision: {fuzzy_results['precision']:.4f}")
    print(f"  Recall:    {fuzzy_results['recall']:.4f}")
    print(f"  F1-Score:  {fuzzy_results['f1_score']:.4f}")
    print(f"  ROC-AUC:   {fuzzy_results['roc_auc']:.4f}")

    all_results['Fuzzy + RF'] = fuzzy_results

    fuzzy_path = os.path.join(models_dir, 'fuzzy_rf_hybrid_model.pkl')
    fuzzy_rf_model.save(fuzzy_path)

    # Comparison
    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)

    import pandas as pd

    comparison_df = pd.DataFrame(all_results).T
    comparison_df = comparison_df.round(4)

    print("\n", comparison_df.to_string())

    print("\nBest Model by Metric:")
    for metric in comparison_df.columns:
        best_model = comparison_df[metric].idxmax()
        best_score = comparison_df[metric].max()
        print(f"  {metric:12s}: {best_model:20s} ({best_score:.4f})")

    ranks = comparison_df.rank(ascending=False)
    avg_rank = ranks.mean(axis=1)
    overall_winner = avg_rank.idxmin()

    print(f"\nOverall Best Model: {overall_winner}")
    print(f"Average Rank: {avg_rank[overall_winner]:.2f}")

    print("\n" + "=" * 70)
    print("✓ TRAINING COMPLETE!")
    print("=" * 70)

    return all_results


if __name__ == "__main__":
    dataset_path = 'url3.csv'
    results = train_models(dataset_path=dataset_path, models_dir='models')
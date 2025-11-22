# Phishing URL Detection System

A machine learning-based system for detecting phishing URLs using Random Forest, XGBoost, and a novel Fuzzy C-Means + Random Forest hybrid approach.

## Features

- **Multiple Detection Models**
  - Random Forest with red flag override system
  - XGBoost classifier
  - Fuzzy C-Means + Random Forest hybrid
  
- **Advanced Feature Extraction**
  - 30+ URL-based features
  - 14 red flag indicators for suspicious patterns
  
- **Interactive Web Interface**
  - Built with Streamlit
  - Real-time URL analysis
  - Side-by-side model comparison

## Project Structure

```
mini-project/
├── train_models.py          # Model training script
├── app.py                   # Streamlit web interface
├── random_forest_model.py   # Random Forest implementation
├── xgboost_model.py         # XGBoost implementation
├── feature_extractor.py     # URL feature extraction
├── data_loader.py           # Dataset loading utilities
├── requirements.txt         # Python dependencies
├── models/                  # Trained models (generated)
└── url3.csv                 # Training dataset
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

**1. Train Models**
```bash
python train_models.py
```
Trains all three models on `url3.csv` and saves them to `models/` directory. Outputs performance metrics and comparison summary.

**2. Run Web Interface**
```bash
streamlit run app.py
```
Analyze URLs in real-time with risk scores, confidence levels, and red flag indicators.


## Models

**Random Forest**: 200 trees with red flag override  
**XGBoost**: Gradient boosting with 200 estimators  
**Fuzzy Hybrid**: Fuzzy C-Means clustering + Random Forest on enhanced features

All models use a red flag override system: if red flag score ≥ 25%, URL is classified as phishing regardless of ML prediction.

## Features Extracted

**Structural (30)**: URL length, domain length, special characters, protocol, TLD analysis  
**Red Flags (14)**: Suspicious keywords, IP addresses, suspicious TLDs, HTTPS misuse

## Dataset

Uses the [PhiUSIIL Phishing URL Dataset](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset) from UCI Machine Learning Repository.

**Format**: CSV file with URL and label columns. Supported labels:
- Binary: `0/1` (auto-detects which is phishing)
- Text: `legitimate/phishing`, `benign/malicious`, `good/bad`

## Performance Metrics

Models evaluated on: Accuracy, Precision, Recall, F1-Score, ROC-AUC

## Usage Example

```python
from random_forest_model import RandomForestPhishingDetector

model = RandomForestPhishingDetector.load('models/random_forest_model.pkl')
result = model.predict_url('https://example.com')
print(result['prediction'], result['risk_score'])
```

## Requirements

Python 3.8+, scikit-learn 1.7.2, xgboost 3.1.1, streamlit 1.51.0, pandas, numpy, scikit-fuzzy

See `requirements.txt` for complete list.
"""
Data Loading Utility for Phishing Dataset
"""

import pandas as pd
import time
from feature_extractor import EnhancedURLFeatureExtractor


def load_phiusiil_dataset(filepath):
    """
    Load PhiUSIIL dataset and extract features from URLs
    
    The dataset format is typically:
    - Column with URLs (string)
    - Column with labels (0/1 or legitimate/phishing)
    """
    print("="*70)
    print("LOADING PhiUSIIL DATASET")
    print("="*70)
    
    # Load dataset
    try:
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filepath.endswith('.xlsx') or filepath.endswith('.xls'):
            df = pd.read_excel(filepath)
        elif filepath.endswith('.txt'):
            df = pd.read_csv(filepath, sep='\t')
        else:
            df = pd.read_csv(filepath)
    except Exception as e:
        print(f"Error loading file: {e}")
        return None, None
    
    print(f"\n✓ File loaded successfully!")
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {df.columns.tolist()}")
    
    # Identify URL and label columns
    url_column = None
    label_column = None
    
    for col in df.columns:
        if 'url' in col.lower() or 'link' in col.lower():
            url_column = col
            break
    
    for col in df.columns:
        if 'label' in col.lower() or 'class' in col.lower() or 'target' in col.lower():
            label_column = col
            break
    
    if url_column is None:
        url_column = df.columns[0]
    if label_column is None:
        label_column = df.columns[-1]
    
    print(f"\n✓ Detected columns:")
    print(f"   URL column: {url_column}")
    print(f"   Label column: {label_column}")
    
    # Extract features from URLs
    print(f"\n[1/3] Extracting features from {len(df)} URLs...")
    print("   This may take a few minutes...")
    
    extractor = EnhancedURLFeatureExtractor()
    feature_list = []
    
    start_time = time.time()
    for idx, url in enumerate(df[url_column]):
        if idx % 1000 == 0 and idx > 0:
            elapsed = time.time() - start_time
            rate = idx / elapsed
            remaining = (len(df) - idx) / rate
            print(f"   Progress: {idx}/{len(df)} ({idx/len(df)*100:.1f}%) - ETA: {remaining:.0f}s")
        
        features = extractor.extract_features(str(url))
        feature_list.append(features)
    
    print(f"   ✓ Feature extraction complete! Time: {time.time() - start_time:.2f}s")
    
    # Convert to DataFrame
    print(f"\n[2/3] Converting to DataFrame...")
    X = pd.DataFrame(feature_list)
    
    # Process labels
    print(f"\n[3/3] Processing labels...")
    y = df[label_column]
    
    unique_labels = y.unique()
    print(f"   Unique labels found: {unique_labels}")
    
    # Standardize labels to 0 (legitimate) and 1 (phishing)
    if set(unique_labels).issubset({0, 1}):
        # Auto-detect label encoding
        label_0_urls = df[y == 0][url_column].head(20).tolist()
        label_1_urls = df[y == 1][url_column].head(20).tolist()
        
        def count_suspicious_patterns(urls):
            suspicious_count = 0
            for url in urls:
                url_lower = str(url).lower()
                if any(pattern in url_lower for pattern in ['login', 'verify', 'update', 'secure', 'account', 'signin']):
                    suspicious_count += 1
                if any(tld in url_lower for tld in ['.tk', '.ml', '.ga', '.xyz']):
                    suspicious_count += 1
                if 'http://' in url_lower and any(kw in url_lower for kw in ['paypal', 'bank', 'secure']):
                    suspicious_count += 1
            return suspicious_count
        
        susp_0 = count_suspicious_patterns(label_0_urls)
        susp_1 = count_suspicious_patterns(label_1_urls)
        
        print(f"   Auto-detection:")
        print(f"   Label 0 suspicious patterns: {susp_0}/20")
        print(f"   Label 1 suspicious patterns: {susp_1}/20")
        
        if susp_0 > susp_1:
            print(f"   ✓ DETECTED: Labels are REVERSED (0=phishing, 1=legitimate)")
            y = 1 - y
        else:
            print(f"   ✓ DETECTED: Labels are CORRECT (0=legitimate, 1=phishing)")
            y = y.astype(int)
            
    elif set(unique_labels).issubset({'legitimate', 'phishing'}):
        y = (y == 'phishing').astype(int)
    elif set(unique_labels).issubset({'benign', 'malicious'}):
        y = (y == 'malicious').astype(int)
    elif set(unique_labels).issubset({'good', 'bad'}):
        y = (y == 'bad').astype(int)
    else:
        print(f"   ⚠️  Unknown label format: {unique_labels}")
        y = (y == unique_labels[1]).astype(int)
    
    print(f"\n{'='*70}")
    print("DATASET SUMMARY")
    print(f"{'='*70}")
    print(f"Total URLs: {len(df)}")
    print(f"Features extracted: {X.shape[1]} (30 original + 14 red flags)")
    print(f"Legitimate: {(y == 0).sum()} ({(y == 0).sum()/len(y)*100:.1f}%)")
    print(f"Phishing: {(y == 1).sum()} ({(y == 1).sum()/len(y)*100:.1f}%)")
    print(f"{'='*70}\n")
    
    return X, y


import pandas as pd

def extract_qa_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts features from QA mismatches and flags.
    """
    features = pd.DataFrame(index=df.index)
    
    # In our synthetic data, we already have some flags built-in.
    # We will pass them through.
    if 'qa_mismatch' in df.columns:
        features['qa_mismatch'] = df['qa_mismatch']
    else:
        features['qa_mismatch'] = 0
        
    if 'wrong_number_flag' in df.columns:
        features['wrong_number_flag'] = df['wrong_number_flag']
    else:
        features['wrong_number_flag'] = 0
        
    if 'medical_advice_flag' in df.columns:
        features['medical_advice_flag'] = df['medical_advice_flag']
    else:
        features['medical_advice_flag'] = 0
        
    # Anomaly indicator
    features['has_any_flag'] = ((features['qa_mismatch'] == 1) | 
                                (features['wrong_number_flag'] == 1) | 
                                (features['medical_advice_flag'] == 1)).astype(int)
    
    return features

import pandas as pd
import numpy as np

def extract_metadata_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts features from metadata like duration and num_questions_asked.
    """
    features = pd.DataFrame(index=df.index)
    
    # Existing features passed through directly
    features['duration'] = df['duration']
    features['num_questions_asked'] = df['num_questions_asked']
    
    # Derived boolean flags
    features['is_short_call'] = (df['duration'] < 60).astype(int)
    features['is_long_call'] = (df['duration'] > 1800).astype(int)
    features['no_questions_asked'] = (df['num_questions_asked'] == 0).astype(int)
    
    # Interaction / ratio features
    # Add a small epsilon to avoid division by zero
    features['questions_per_minute'] = df['num_questions_asked'] / ((df['duration'] / 60) + 1e-5)
    
    return features

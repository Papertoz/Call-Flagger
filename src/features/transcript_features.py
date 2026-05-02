import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import re

def extract_transcript_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts NLP features from the transcript:
    1. Keyword-based flags.
    2. SentenceTransformer embeddings.
    """
    print(f"Extracting transcript features for {len(df)} rows...")
    features = pd.DataFrame(index=df.index)
    
    # 1. Keyword-based features
    medical_keywords = r'\b(ibuprofen|medication|diabetes|prescribe|symptoms|pain)\b'
    angry_keywords = r'\b(ridiculous|useless|sue|cancel)\b'
    
    # Fill NaN with empty string just in case
    transcripts = df['transcript'].fillna("")
    
    features['has_medical_keywords'] = transcripts.str.contains(medical_keywords, flags=re.IGNORECASE, regex=True).astype(int)
    features['has_angry_keywords'] = transcripts.str.contains(angry_keywords, flags=re.IGNORECASE, regex=True).astype(int)
    features['transcript_length'] = transcripts.apply(lambda x: len(str(x).split()))
    
    # 2. Embeddings
    print("Generating SentenceTransformer embeddings (this might take a moment)...")
    # Using a small, fast model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # We use a smaller subset of dimensions to keep it fast, or all 384
    # XGBoost handles 384 dimensions well.
    embeddings = model.encode(transcripts.tolist(), show_progress_bar=True)
    
    emb_df = pd.DataFrame(embeddings, columns=[f'emb_{i}' for i in range(embeddings.shape[1])], index=df.index)
    
    # Combine
    features = pd.concat([features, emb_df], axis=1)
    
    return features

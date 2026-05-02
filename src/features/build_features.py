import os
import pandas as pd
from src.data.preprocess import clean_data
from src.features.metadata_features import extract_metadata_features
from src.features.transcript_features import extract_transcript_features
from src.features.qa_features import extract_qa_features

def process_and_build_features(input_path: str, output_path: str):
    """
    Reads raw data, cleans it, extracts all features, and saves the processed data.
    """
    print(f"Processing {input_path}...")
    df = pd.read_csv(input_path)
    
    # 1. Clean
    df_clean = clean_data(df)
    
    # 2. Extract Features
    meta_feats = extract_metadata_features(df_clean)
    qa_feats = extract_qa_features(df_clean)
    text_feats = extract_transcript_features(df_clean)
    
    # 3. Combine
    features = pd.concat([meta_feats, qa_feats, text_feats], axis=1)
    
    # Add target variable
    if 'has_ticket' in df.columns:
        features['has_ticket'] = df['has_ticket']
        
    # 4. Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    features.to_csv(output_path, index=False)
    print(f"Saved processed features to {output_path} (Shape: {features.shape})")

if __name__ == "__main__":
    process_and_build_features("data/raw/train.csv", "data/processed/train_processed.csv")
    process_and_build_features("data/raw/val.csv", "data/processed/val_processed.csv")
    process_and_build_features("data/raw/test.csv", "data/processed/test_processed.csv")

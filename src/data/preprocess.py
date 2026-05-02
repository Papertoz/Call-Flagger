import pandas as pd

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the raw data.
    - Handles missing values.
    - Standardizes text columns.
    """
    df_clean = df.copy()
    
    # Fill missing transcripts with empty string
    if 'transcript' in df_clean.columns:
        df_clean['transcript'] = df_clean['transcript'].fillna("")
        
    # Fill numerical NaNs with median/mode or 0 depending on the logic
    if 'duration' in df_clean.columns:
        df_clean['duration'] = df_clean['duration'].fillna(df_clean['duration'].median())
        
    if 'num_questions_asked' in df_clean.columns:
        df_clean['num_questions_asked'] = df_clean['num_questions_asked'].fillna(0)
        
    return df_clean

if __name__ == "__main__":
    # Small test
    df = pd.read_csv("data/raw/train.csv")
    cleaned_df = clean_data(df)
    print("Data cleaned successfully. Shape:", cleaned_df.shape)

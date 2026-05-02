import pandas as pd
import numpy as np

def calculate_scale_pos_weight(df: pd.DataFrame, target_col='has_ticket') -> float:
    """
    Calculates scale_pos_weight for XGBoost to handle class imbalance.
    formula: count(negative examples) / count(positive examples)
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")
        
    num_negatives = (df[target_col] == 0).sum()
    num_positives = (df[target_col] == 1).sum()
    
    if num_positives == 0:
        return 1.0
        
    scale_pos_weight = num_negatives / num_positives
    print(f"Calculated scale_pos_weight: {scale_pos_weight:.2f} (Negatives: {num_negatives}, Positives: {num_positives})")
    
    return scale_pos_weight

if __name__ == "__main__":
    df = pd.read_csv("data/processed/train_processed.csv")
    calculate_scale_pos_weight(df)

import pandas as pd
import numpy as np
import pickle
import json
import os
from sklearn.metrics import f1_score, precision_score, recall_score

def tune_threshold(val_path: str, model_path: str, output_path: str):
    """
    Finds the optimal probability threshold to maximize F1 score on the validation set.
    """
    print("Loading validation data and model for threshold tuning...")
    val_df = pd.read_csv(val_path)
    
    target_col = 'has_ticket'
    X_val = val_df.drop(columns=[target_col])
    y_val = val_df[target_col]
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
        
    # Get predicted probabilities
    y_probs = model.predict_proba(X_val)[:, 1]
    
    # Search for optimal threshold
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_f1 = 0
    best_thresh = 0.5
    
    print("\nTuning Threshold:")
    for thresh in thresholds:
        y_pred = (y_probs >= thresh).astype(int)
        f1 = f1_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, zero_division=0)
        recall = recall_score(y_val, y_pred, zero_division=0)
        
        print(f"Threshold: {thresh:.2f} -> F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            
    print(f"\nOptimal Threshold: {best_thresh:.2f} (F1: {best_f1:.4f})")
    
    # Save the optimal threshold
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump({'optimal_threshold': float(best_thresh), 'best_f1': float(best_f1)}, f)
        
    print(f"Threshold saved to {output_path}")
    return best_thresh

if __name__ == "__main__":
    tune_threshold(
        "data/processed/val_processed.csv",
        "models/xgboost_model.pkl",
        "models/threshold.json"
    )

import pandas as pd
import pickle
import json
import os
from src.rules.rule_engine import apply_rules

def predict_calls(features_path: str, model_path: str, threshold_path: str, output_path: str):
    """
    Makes final predictions combining ML probabilities and the Rule Engine.
    """
    print("Loading data for prediction...")
    df = pd.read_csv(features_path)
    
    # Drop target column if it exists to avoid leakage
    X = df.drop(columns=['has_ticket'], errors='ignore')
    
    print("Loading model and threshold...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
        
    with open(threshold_path, 'r') as f:
        threshold_data = json.load(f)
        threshold = threshold_data['optimal_threshold']
        
    # 1. ML Prediction
    print("Running ML model...")
    ml_probs = model.predict_proba(X)[:, 1]
    ml_preds = (ml_probs >= threshold).astype(int)
    
    # 2. Rule Engine Prediction
    print("Running Rule Engine...")
    rule_preds = apply_rules(df).values
    
    # 3. Combine (Logical OR)
    print("Combining predictions...")
    final_preds = ((ml_preds == 1) | (rule_preds == 1)).astype(int)
    
    # Generate Output
    output_df = pd.DataFrame({
        'ml_probability': ml_probs,
        'ml_prediction': ml_preds,
        'rule_prediction': rule_preds,
        'final_prediction': final_preds
    })
    
    if 'has_ticket' in df.columns:
        output_df['actual'] = df['has_ticket']
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
    
    return output_df

if __name__ == "__main__":
    predict_calls(
        "data/processed/test_processed.csv",
        "models/xgboost_model.pkl",
        "models/threshold.json",
        "outputs/predictions.csv"
    )

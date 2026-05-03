import json
import os
from sklearn.metrics import classification_report
import pandas as pd
from src.data.load_data import load_and_split_data
from src.features.build_features import process_and_build_features
from src.models.train import train_model
from src.models.threshold_tuning import tune_threshold
from src.models.predict import predict_calls

def run_pipeline():
    """
    Executes the entire machine learning pipeline end-to-end.
    """
    print("="*50)
    print("Starting Call Quality Auto-Flagger Pipeline")
    print("="*50)

    # 1. Generate & Load Data
    print("\n--- PHASE 1: Data Setup ---")
    load_and_split_data(num_samples=5000)

    # 2. Build Features
    print("\n--- PHASE 2: Feature Engineering ---")
    process_and_build_features("data/raw/train.csv", "data/processed/train_processed.csv")
    process_and_build_features("data/raw/val.csv", "data/processed/val_processed.csv")
    process_and_build_features("data/raw/test.csv", "data/processed/test_processed.csv")

    # 3. Model Training & Tuning
    print("\n--- PHASE 3: Training & Tuning ---")
    train_model(
        "data/processed/train_processed.csv",
        "data/processed/val_processed.csv",
        "models/xgboost_model.pkl"
    )
    tune_threshold(
        "data/processed/val_processed.csv",
        "models/xgboost_model.pkl",
        "models/threshold.json"
    )

    # 4. Evaluation on Test Set
    print("\n--- PHASE 4: Final Evaluation (Test Set) ---")
    results = predict_calls(
        "data/processed/test_processed.csv",
        "models/xgboost_model.pkl",
        "models/threshold.json",
        "outputs/predictions.csv"
    )
    
    # Calculate metrics
    print("\nFinal Test Metrics (ML + Rule Engine):")
    report = classification_report(results['actual'], results['final_prediction'], output_dict=True)
    print(classification_report(results['actual'], results['final_prediction']))
    
    # Save metrics
    with open("outputs/classification_metrics.json", "w") as f:
        json.dump(report, f, indent=4)
        
    print("\nPipeline Completed Successfully!")

if __name__ == "__main__":
    run_pipeline()

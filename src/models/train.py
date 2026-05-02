import pandas as pd
import xgboost as xgb
import pickle
import os
from sklearn.metrics import classification_report, f1_score
from src.imbalance.class_weights import calculate_scale_pos_weight

def train_model(train_path: str, val_path: str, model_output_path: str):
    """
    Trains the XGBoost model for call flagging.
    """
    print("Loading data for training...")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    # Separate features and target
    target_col = 'has_ticket'
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    
    X_val = val_df.drop(columns=[target_col])
    y_val = val_df[target_col]
    
    # Handle Imbalance
    scale_pos_weight = calculate_scale_pos_weight(train_df, target_col)
    
    # Initialize XGBoost Classifier
    print("Initializing XGBoost...")
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        eval_metric='aucpr', # Area under PR curve is better for imbalanced data
        early_stopping_rounds=20,
        random_state=42
    )
    
    # Train
    print("Training model...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=10
    )
    
    # Evaluate on validation set
    y_val_pred = model.predict(X_val)
    print("\nValidation Performance (Default Threshold = 0.5):")
    print(classification_report(y_val, y_val_pred))
    
    # Save the model
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    with open(model_output_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\nModel saved to {model_output_path}")
    
    return model

if __name__ == "__main__":
    train_model(
        "data/processed/train_processed.csv",
        "data/processed/val_processed.csv",
        "models/xgboost_model.pkl"
    )

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from src.data.generate_synthetic_data import generate_synthetic_data

def load_and_split_data(output_dir="data/raw", num_samples=5000):
    """
    Generates synthetic data and splits it into train, val, and test sets.
    Preserves the class imbalance using stratified splitting.
    """
    print("Loading and splitting data...")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Generate the full dataset
    df = generate_synthetic_data(num_samples=num_samples)
    
    # 2. Split into Train (70%), Val (15%), Test (15%)
    # Use stratify to maintain the 9% positive class ratio across all splits
    train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df['has_ticket'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df['has_ticket'], random_state=42)
    
    # 3. Save to data/raw/
    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "val.csv")
    test_path = os.path.join(output_dir, "test.csv")
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Train set saved to {train_path} (Shape: {train_df.shape})")
    print(f"Val set saved to {val_path} (Shape: {val_df.shape})")
    print(f"Test set saved to {test_path} (Shape: {test_df.shape})")
    
    # Verify stratification
    print("\nTarget distribution in Train:")
    print(train_df['has_ticket'].value_counts(normalize=True))
    
    return train_df, val_df, test_df

if __name__ == "__main__":
    load_and_split_data()

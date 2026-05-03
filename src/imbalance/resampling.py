import pandas as pd
from imblearn.over_sampling import SMOTE

def apply_smote(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Applies Synthetic Minority Over-sampling Technique (SMOTE) to the training data.
    Note: For high-dimensional embeddings (like SentenceTransformers),
    using scale_pos_weight in XGBoost is often preferred over SMOTE.
    This function is provided as an alternative approach.
    """
    print("Applying SMOTE to handle class imbalance...")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    print(f"Original target distribution:\n{y_train.value_counts()}")
    print(f"Resampled target distribution:\n{pd.Series(y_resampled).value_counts()}")
    
    return X_resampled, y_resampled

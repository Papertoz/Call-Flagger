import pandas as pd

def apply_rules(df: pd.DataFrame) -> pd.Series:
    """
    Applies deterministic rules to flag a call.
    Returns a pandas Series of binary flags (1 if rule flagged, 0 otherwise).
    """
    # A call is flagged by rules if it has medical advice or angry keywords,
    # or if it was a wrong number
    
    rule_flags = (
        (df.get('has_medical_keywords', 0) == 1) |
        (df.get('has_angry_keywords', 0) == 1) |
        (df.get('wrong_number_flag', 0) == 1) |
        (df.get('medical_advice_flag', 0) == 1)
    ).astype(int)
    
    return rule_flags

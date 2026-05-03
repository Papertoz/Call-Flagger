import pandas as pd
import json
import os
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, confusion_matrix

def evaluate_model(predictions_path: str, report_output_path: str):
    """
    Evaluates the final predictions and generates a detailed validation report.
    """
    print(f"Evaluating predictions from {predictions_path}...")
    df = pd.read_csv(predictions_path)
    
    if 'actual' not in df.columns or 'final_prediction' not in df.columns:
        raise ValueError("Predictions file must contain 'actual' and 'final_prediction' columns.")
        
    y_true = df['actual']
    y_pred = df['final_prediction']
    
    # Calculate metrics
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    report_text = f"=== Call Quality Auto-Flagger Validation Report ===\n"
    report_text += f"Precision: {precision:.4f}\n"
    report_text += f"Recall:    {recall:.4f}\n"
    report_text += f"F1 Score:  {f1:.4f}\n\n"
    
    report_text += "Confusion Matrix:\n"
    report_text += f"True Negatives:  {cm[0][0]}\n"
    report_text += f"False Positives: {cm[0][1]}\n"
    report_text += f"False Negatives: {cm[1][0]}\n"
    report_text += f"True Positives:  {cm[1][1]}\n\n"
    
    report_text += "Detailed Classification Report:\n"
    report_text += classification_report(y_true, y_pred)
    
    os.makedirs(os.path.dirname(report_output_path), exist_ok=True)
    with open(report_output_path, 'w') as f:
        f.write(report_text)
        
    print(f"Validation report saved to {report_output_path}")

if __name__ == "__main__":
    evaluate_model("outputs/predictions.csv", "outputs/validation_report.txt")

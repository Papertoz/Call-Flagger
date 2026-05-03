# 🚀 Call Quality Auto-Flagger

## 📌 Overview
The Call Quality Auto-Flagger is an end-to-end machine learning system designed to automatically identify call center interactions that require human review. The system analyzes call transcripts, metadata, and Q&A responses to detect subtle anomalies and compliance violations, enabling scalable and efficient quality monitoring.

---

## 🎯 Objective
To build a binary classification model that predicts whether a call should be flagged for human review (`has_ticket = True`), with a focus on:

- Maximizing recall (catching problematic calls)
- Maintaining a strong F1 score
- Handling highly imbalanced data (~9% positive cases)

---

## 🧩 Problem Scope
The system detects multiple categories of anomalies, including:

- Outcome misclassification  
- Speech-to-text (STT) errors  
- Skipped or missing required questions  
- Wrong number misclassification  
- Medical advice violations (compliance breach)  
- Data capture inconsistencies  

---

## 🏗️ System Architecture

```text
Raw Call Data (Transcript + Metadata + Q&A)
        ↓
Data Preprocessing
        ↓
Feature Engineering (Metadata + Text + Q&A)
        ↓
Text Embeddings (SentenceTransformers)
        ↓
Imbalance Handling (Class Weights / Resampling)
        ↓
Machine Learning Model (XGBoost)
        ↓
Threshold Optimization
        ↓
Rule-Based Validation Layer
        ↓
Final Prediction (Flag / No Flag)
```

---

## ⚙️ Tech Stack

- Python  
- Pandas, NumPy  
- Scikit-learn  
- XGBoost  
- SentenceTransformers

---

## 🚀 How to Run the Pipeline

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the End-to-End System**
   This single script will generate mock data, extract NLP embeddings, train the XGBoost model, tune the threshold, and output final predictions.
   ```bash
   python main.py
   ```

3. **Check Outputs**
   - Model saved to: `models/xgboost_model.pkl`
   - Optimal Threshold saved to: `models/threshold.json`
   - Final Predictions saved to: `outputs/predictions.csv`
   - Performance Metrics saved to: `outputs/classification_metrics.json`

---

## 🏆 Key Highlights

- Built a hybrid ML + rule-based anomaly detection system.
- Effectively handled 91% vs 9% imbalanced data scenarios using `scale_pos_weight` and threshold tuning.
- Applied practical NLP using `all-MiniLM-L6-v2` SentenceTransformer embeddings.
- Modular, production-ready python architecture.

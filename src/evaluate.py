"""
Model Evaluation Module
-----------------------
Evaluates trained model on test data and exports metrics.json for DVC tracking.
"""

import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from src import load_params


def evaluate_model(params: dict):
    test_path = params["evaluate"]["test_data_path"]
    model_path = params["evaluate"]["model_path"]
    metrics_path = params["evaluate"]["metrics_path"]
    target_col = params["preprocess"]["target_column"]

    print(f"[INFO] Loading test data from: {test_path}")
    test_df = pd.read_csv(test_path)
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    print(f"[INFO] Loading model from: {model_path}")
    model = joblib.load(model_path)

    y_pred = model.predict(X_test)
    unique_classes = np.unique(y_test)

    if len(unique_classes) == 2:
        y_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
    else:
        y_proba = model.predict_proba(X_test)
        roc_auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
        precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
        recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
    }

    os.makedirs(os.path.dirname(metrics_path) if os.path.dirname(metrics_path) else ".", exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"[INFO] Evaluation completed. Metrics logged to {metrics_path}:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    params = load_params()
    evaluate_model(params)
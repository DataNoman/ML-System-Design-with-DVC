"""
Model Evaluation Module
-----------------------
Evaluates the trained XGBoost model on the test dataset.

Outputs:
1. metrics.json
2. reports/confusion_matrix.csv
3. reports/classification_report.txt
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src import load_params


def evaluate_model(params: dict):
    test_path = params["evaluate"]["test_data_path"]
    model_path = params["evaluate"]["model_path"]
    metrics_path = params["evaluate"]["metrics_path"]

    target_col = params["preprocess"]["target_column"]

    # --------------------------------------------------
    # Load test data
    # --------------------------------------------------

    print(f"[INFO] Loading test data from: {test_path}")

    test_df = pd.read_csv(test_path)

    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    # --------------------------------------------------
    # Load trained model
    # --------------------------------------------------

    print(f"[INFO] Loading model from: {model_path}")

    model = joblib.load(model_path)

    # --------------------------------------------------
    # Predictions
    # --------------------------------------------------

    print("[INFO] Generating predictions...")

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # --------------------------------------------------
    # Overall metrics
    # --------------------------------------------------

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )

    unique_classes = np.unique(y_test)

    if len(unique_classes) == 2:
        roc_auc = roc_auc_score(
            y_test,
            y_proba[:, 1],
        )
    else:
        roc_auc = roc_auc_score(
            y_test,
            y_proba,
            multi_class="ovr",
            average="macro",
        )

    # --------------------------------------------------
    # Classification report
    # --------------------------------------------------

    class_report = classification_report(
        y_test,
        y_pred,
        zero_division=0,
    )

    print("\n[INFO] Classification Report:")
    print(class_report)

    # --------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------

    cm = confusion_matrix(
        y_test,
        y_pred,
    )

    print("[INFO] Confusion Matrix:")
    print(cm)

    # --------------------------------------------------
    # Save metrics
    # --------------------------------------------------

    metrics = {
        "accuracy": round(float(accuracy), 4),
        "precision_macro": round(float(precision), 4),
        "recall_macro": round(float(recall), 4),
        "f1_macro": round(float(f1), 4),
        "roc_auc_macro_ovr": round(float(roc_auc), 4),
    }

    metrics_directory = os.path.dirname(metrics_path)

    if metrics_directory:
        os.makedirs(
            metrics_directory,
            exist_ok=True,
        )

    with open(metrics_path, "w") as f:
        json.dump(
            metrics,
            f,
            indent=4,
        )

    # --------------------------------------------------
    # Save confusion matrix
    # --------------------------------------------------

    reports_directory = "reports"

    os.makedirs(
        reports_directory,
        exist_ok=True,
    )

    class_labels = [
        str(label)
        for label in sorted(unique_classes)
    ]

    cm_df = pd.DataFrame(
        cm,
        index=[
            f"Actual_{label}"
            for label in class_labels
        ],
        columns=[
            f"Predicted_{label}"
            for label in class_labels
        ],
    )

    confusion_matrix_path = os.path.join(
        reports_directory,
        "confusion_matrix.csv",
    )

    cm_df.to_csv(
        confusion_matrix_path
    )

    # --------------------------------------------------
    # Save classification report
    # --------------------------------------------------

    classification_report_path = os.path.join(
        reports_directory,
        "classification_report.txt",
    )

    with open(
        classification_report_path,
        "w",
    ) as f:
        f.write(class_report)

    # --------------------------------------------------
    # Final output
    # --------------------------------------------------

    print("\n[INFO] Evaluation completed.")

    print("\n[INFO] Overall Metrics:")

    print(
        json.dumps(
            metrics,
            indent=4,
        )
    )

    print(
        f"\n[INFO] Metrics saved to: "
        f"{metrics_path}"
    )

    print(
        f"[INFO] Confusion matrix saved to: "
        f"{confusion_matrix_path}"
    )

    print(
        f"[INFO] Classification report saved to: "
        f"{classification_report_path}"
    )


if __name__ == "__main__":
    params = load_params()

    evaluate_model(params)
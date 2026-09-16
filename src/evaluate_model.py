"""
Generic Model Evaluation Module
--------------------------------
Evaluates a trained classification model and saves:
- metrics
- confusion matrix
- classification report
"""

import argparse
import json
import os

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from src import load_params


def evaluate_model(
    params: dict,
    model_path: str,
    metrics_path: str,
    confusion_matrix_path: str,
    classification_report_path: str,
):

    test_path = params["preprocess"]["test_data_path"]
    target_col = params["preprocess"]["target_column"]

    print(f"[INFO] Loading test dataset from: {test_path}")
    test_df = pd.read_csv(test_path)

    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    print(f"[INFO] Loading model from: {model_path}")
    model = joblib.load(model_path)

    print("[INFO] Generating predictions...")

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

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

    roc_auc = roc_auc_score(
        y_test,
        y_prob,
        multi_class="ovr",
        average="macro",
    )

    metrics = {
        "accuracy": accuracy,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "roc_auc_macro_ovr": roc_auc,
    }

    # Create directories
    for path in [
        metrics_path,
        confusion_matrix_path,
        classification_report_path,
    ]:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

    # Save metrics
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Save confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    pd.DataFrame(cm).to_csv(
        confusion_matrix_path,
        index=False,
    )

    # Save classification report
    report = classification_report(
        y_test,
        y_pred,
        zero_division=0,
    )

    with open(classification_report_path, "w") as f:
        f.write(report)

    print("\n[RESULTS]")
    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Precision (macro): {precision:.4f}")
    print(f"Recall (macro):    {recall:.4f}")
    print(f"F1 (macro):        {f1:.4f}")
    print(f"ROC-AUC (macro):   {roc_auc:.4f}")

    print(f"\n[INFO] Metrics saved to: {metrics_path}")
    print(f"[INFO] Confusion matrix saved to: {confusion_matrix_path}")
    print(f"[INFO] Classification report saved to: {classification_report_path}")


def main():

    parser = argparse.ArgumentParser(
        description="Evaluate a trained classification model."
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Path to trained model.",
    )

    parser.add_argument(
        "--metrics",
        required=True,
        help="Path to metrics JSON.",
    )

    parser.add_argument(
        "--confusion-matrix",
        required=True,
        help="Path to confusion matrix CSV.",
    )

    parser.add_argument(
        "--classification-report",
        required=True,
        help="Path to classification report TXT.",
    )

    args = parser.parse_args()

    params = load_params()

    evaluate_model(
        params=params,
        model_path=args.model,
        metrics_path=args.metrics,
        confusion_matrix_path=args.confusion_matrix,
        classification_report_path=args.classification_report,
    )


if __name__ == "__main__":
    main()
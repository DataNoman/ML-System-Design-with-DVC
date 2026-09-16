"""
Model Comparison Module
-----------------------
Loads evaluation metrics from all trained models
and creates a comparison table.
"""

import json
import os

import pandas as pd


MODEL_METRICS = {
    "XGBoost": "metrics/xgboost.json",
    "Random Forest": "metrics/random_forest.json",
    "LightGBM": "metrics/lightgbm.json",
    "CatBoost": "metrics/catboost.json",
    "Logistic Regression": "metrics/logistic_regression.json",
}


def load_metrics():

    results = []

    for model_name, metrics_path in MODEL_METRICS.items():

        print(f"[INFO] Loading metrics for {model_name}")

        with open(metrics_path, "r") as f:
            metrics = json.load(f)

        results.append(
            {
                "Model": model_name,
                "Accuracy": metrics["accuracy"],
                "Precision": metrics["precision_macro"],
                "Recall": metrics["recall_macro"],
                "F1": metrics["f1_macro"],
                "ROC-AUC": metrics["roc_auc_macro_ovr"],
            }
        )

    return pd.DataFrame(results)


def main():

    output_path = "reports/model_comparison.csv"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    comparison_df = load_metrics()

    comparison_df.to_csv(
        output_path,
        index=False,
    )

    print("\n[MODEL COMPARISON]")
    print(comparison_df.to_string(index=False))

    print(f"\n[INFO] Comparison saved to: {output_path}")


if __name__ == "__main__":
    main()
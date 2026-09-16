"""
Logistic Regression Training Module
------------------------------------
Trains a Logistic Regression classifier on the preprocessed dataset.
"""

import os
import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression

from src import load_params


def train_logistic(params: dict):

    train_path = params["preprocess"]["train_data_path"]
    target_col = params["preprocess"]["target_column"]

    model_params = params["models"]["logistic_regression"]

    model_output_path = model_params["model_path"]

    print(f"[INFO] Loading training dataset from: {train_path}")

    train_df = pd.read_csv(train_path)

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]

    print("[INFO] Training Logistic Regression model...")

    model = LogisticRegression(
        C=model_params["C"],
        max_iter=model_params["max_iter"],
        random_state=model_params["random_state"]
    )

    model.fit(X_train, y_train)

    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)

    joblib.dump(model, model_output_path)

    print(f"[INFO] Logistic Regression model saved to: {model_output_path}")


if __name__ == "__main__":
    params = load_params()
    train_logistic(params)
"""
XGBoost Training Module
-----------------------
Trains an XGBoost classifier on the preprocessed dataset.
"""

import os
import joblib
import pandas as pd
from xgboost import XGBClassifier

from src import load_params


def train_xgboost(params: dict):

    train_path = params["preprocess"]["train_data_path"]
    target_col = params["preprocess"]["target_column"]

    model_params = params["models"]["xgboost"]

    model_output_path = model_params["model_path"]

    print(f"[INFO] Loading training dataset from: {train_path}")

    train_df = pd.read_csv(train_path)

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]

    print("[INFO] Training XGBoost model...")

    model = XGBClassifier(
        n_estimators=model_params["n_estimators"],
        max_depth=model_params["max_depth"],
        learning_rate=model_params["learning_rate"],
        random_state=model_params["random_state"],
        eval_metric="logloss"
    )

    model.fit(X_train, y_train)

    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)

    joblib.dump(model, model_output_path)

    print(f"[INFO] XGBoost model saved to: {model_output_path}")


if __name__ == "__main__":
    params = load_params()
    train_xgboost(params)
"""
Model Training Module
---------------------
Trains an XGBoost classifier model on preprocessed dataset.
"""

import os
import joblib
import pandas as pd
from xgboost import XGBClassifier
from src import load_params


def train_model(params: dict):
    train_path = params["train"]["train_data_path"]
    model_output_path = params["train"]["model_path"]
    target_col = params["preprocess"]["target_column"]

    # Model Hyperparameters from params.yaml
    n_estimators = params["train"]["n_estimators"]
    max_depth = params["train"]["max_depth"]
    learning_rate = params["train"]["learning_rate"]
    random_state = params["train"]["random_state"]

    print(f"[INFO] Loading training dataset from: {train_path}")
    train_df = pd.read_csv(train_path)

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]

    print(f"[INFO] Training XGBoost model (n_estimators={n_estimators}, max_depth={max_depth})...")
    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
        eval_metric="logloss"
    )
    model.fit(X_train, y_train)

    # Save model artifact
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(model, model_output_path)
    print(f"[INFO] Model successfully trained and saved to: {model_output_path}")


if __name__ == "__main__":
    params = load_params()
    train_model(params)
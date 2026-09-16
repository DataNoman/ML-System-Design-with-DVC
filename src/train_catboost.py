"""
CatBoost Training Module
------------------------
Trains a CatBoost classifier on the preprocessed dataset.
"""

import os
import joblib
import pandas as pd

from catboost import CatBoostClassifier

from src import load_params


def train_catboost(params: dict):

    train_path = params["preprocess"]["train_data_path"]
    target_col = params["preprocess"]["target_column"]

    model_params = params["models"]["catboost"]

    model_output_path = model_params["model_path"]

    print(f"[INFO] Loading training dataset from: {train_path}")

    train_df = pd.read_csv(train_path)

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]

    print("[INFO] Training CatBoost model...")

    model = CatBoostClassifier(
        iterations=model_params["iterations"],
        depth=model_params["depth"],
        learning_rate=model_params["learning_rate"],
        random_seed=model_params["random_state"],
        verbose=model_params["verbose"]
    )

    model.fit(X_train, y_train)

    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)

    joblib.dump(model, model_output_path)

    print(f"[INFO] CatBoost model saved to: {model_output_path}")


if __name__ == "__main__":
    params = load_params()
    train_catboost(params)
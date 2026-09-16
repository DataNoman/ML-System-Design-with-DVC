"""
Random Forest Training Module
-----------------------------
Trains a Random Forest classifier on the preprocessed dataset.
"""

import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from src import load_params


def train_random_forest(params: dict):

    train_path = params["preprocess"]["train_data_path"]
    target_col = params["preprocess"]["target_column"]

    model_params = params["models"]["random_forest"]

    model_output_path = model_params["model_path"]

    print(f"[INFO] Loading training dataset from: {train_path}")

    train_df = pd.read_csv(train_path)

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]

    print("[INFO] Training Random Forest model...")

    model = RandomForestClassifier(
        n_estimators=model_params["n_estimators"],
        max_depth=model_params["max_depth"],
        random_state=model_params["random_state"],
        n_jobs=model_params["n_jobs"]
    )

    model.fit(X_train, y_train)

    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)

    joblib.dump(model, model_output_path)

    print(f"[INFO] Random Forest model saved to: {model_output_path}")


if __name__ == "__main__":
    params = load_params()
    train_random_forest(params)
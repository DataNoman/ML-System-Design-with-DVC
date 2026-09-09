"""
Data Preprocessing Module
-------------------------
Cleans, encodes, handles missing values, applies SMOTE, 
and splits data into training and test sets.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from src import load_params


def _resolve_column(columns, preferred_name: str):
    lookup = {str(col).lower(): col for col in columns}
    normalized = preferred_name.lower()
    if normalized in lookup:
        return lookup[normalized]
    raise KeyError(f"Column '{preferred_name}' not found in dataset. Available columns: {list(columns)}")


def preprocess_data(params: dict):
    raw_path = params["preprocess"]["raw_data_path"]
    train_out = params["preprocess"]["train_data_path"]
    test_out = params["preprocess"]["test_data_path"]
    test_size = params["preprocess"]["test_size"]
    random_state = params["preprocess"]["random_state"]

    print(f"[INFO] Loading raw data from: {raw_path}")
    df = pd.read_csv(raw_path)

    # Drop unique identifiers if present
    for id_col in ["id", "Patient_ID"]:
        if id_col in df.columns:
            df = df.drop(columns=[id_col])

    bmi_col = _resolve_column(df.columns, "BMI")
    if df[bmi_col].isnull().sum() > 0:
        median_bmi = df[bmi_col].median()
        df[bmi_col] = df[bmi_col].fillna(median_bmi)
        print(f"[INFO] Imputed missing '{bmi_col}' values with median: {median_bmi}")

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            median_value = df[col].median()
            df[col] = df[col].fillna(median_value)
            print(f"[INFO] Imputed missing '{col}' values with median: {median_value}")

    target_col = _resolve_column(df.columns, params["preprocess"]["target_column"])
    X = df.drop(columns=[target_col])
    y = df[target_col]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    categorical_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    smote = SMOTE(random_state=random_state)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
    print(f"[INFO] Applied SMOTE. Training samples balance: {pd.Series(y_train_res).value_counts().to_dict()}")

    feature_cols = X.columns.tolist()
    train_df = pd.DataFrame(X_train_res, columns=feature_cols)
    train_df[target_col] = y_train_res

    test_df = pd.DataFrame(X_test_scaled, columns=feature_cols)
    test_df[target_col] = y_test

    os.makedirs(os.path.dirname(train_out), exist_ok=True)
    train_df.to_csv(train_out, index=False)
    test_df.to_csv(test_out, index=False)
    print(f"[INFO] Preprocessed data saved to:\n  - {train_out}\n  - {test_out}")


if __name__ == "__main__":
    params = load_params()
    preprocess_data(params)
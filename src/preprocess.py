"""
Data Preprocessing Module
-------------------------
Prepares the stroke-risk dataset for machine learning.

Steps:
1. Load raw dataset
2. Remove identifier columns
3. Remove target-leakage columns
4. Separate features and target
5. Encode target labels
6. Split into training and testing sets
7. Impute missing values using training data only
8. One-hot encode categorical features
9. Standardize numerical features
10. Apply SMOTE to training data only
11. Save processed train/test datasets
"""

import os

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src import load_params


def _resolve_column(columns, preferred_name: str):
    """
    Find a column name without being sensitive to capitalization.
    """
    lookup = {str(col).lower(): col for col in columns}
    normalized = preferred_name.lower()

    if normalized in lookup:
        return lookup[normalized]

    raise KeyError(
        f"Column '{preferred_name}' not found in dataset. "
        f"Available columns: {list(columns)}"
    )


def preprocess_data(params: dict):
    """
    Main preprocessing pipeline.
    """

    # ---------------------------------------------------------
    # 1. Load parameters
    # ---------------------------------------------------------

    raw_path = params["preprocess"]["raw_data_path"]
    train_out = params["preprocess"]["train_data_path"]
    test_out = params["preprocess"]["test_data_path"]

    test_size = params["preprocess"]["test_size"]
    random_state = params["preprocess"]["random_state"]

    target_col_name = params["preprocess"]["target_column"]

    print(f"[INFO] Loading raw data from: {raw_path}")

    df = pd.read_csv(raw_path)

    print(f"[INFO] Raw dataset shape: {df.shape}")

    # ---------------------------------------------------------
    # 2. Resolve target column
    # ---------------------------------------------------------

    target_col = _resolve_column(
        df.columns,
        target_col_name
    )

    print(f"[INFO] Target column: {target_col}")

    # ---------------------------------------------------------
    # 3. Remove identifier columns
    # ---------------------------------------------------------

    id_columns = []

    for col in df.columns:
        if str(col).lower() in ["id", "patient_id"]:
            id_columns.append(col)

    if id_columns:
        df = df.drop(columns=id_columns)

        print(
            f"[INFO] Removed identifier column(s): "
            f"{id_columns}"
        )

    # ---------------------------------------------------------
    # 4. Remove target leakage
    # ---------------------------------------------------------
    #
    # Stroke_Risk_Score directly determines Stroke_Risk:
    #
    # 2-34   -> Low
    # 35-64  -> Moderate
    # 65-100 -> High
    #
    # Keeping this column would allow the model to simply
    # reproduce the rule used to create the target.
    # ---------------------------------------------------------

    leakage_columns = []

    for col in df.columns:
        if str(col).lower() == "stroke_risk_score":
            leakage_columns.append(col)

    if leakage_columns:
        df = df.drop(columns=leakage_columns)

        print(
            f"[INFO] Removed target leakage column(s): "
            f"{leakage_columns}"
        )

    # ---------------------------------------------------------
    # 5. Separate features and target
    # ---------------------------------------------------------

    X = df.drop(columns=[target_col])
    y = df[target_col]

    print(f"[INFO] Feature shape before preprocessing: {X.shape}")

    print("[INFO] Original target distribution:")
    print(y.value_counts())

    # ---------------------------------------------------------
    # 6. Encode target labels
    # ---------------------------------------------------------

    label_encoder = LabelEncoder()

    y_encoded = label_encoder.fit_transform(y)

    print("[INFO] Target label mapping:")

    for encoded_value, original_value in enumerate(
        label_encoder.classes_
    ):
        print(
            f"    {original_value} -> {encoded_value}"
        )

    # ---------------------------------------------------------
    # 7. Train/Test split
    # ---------------------------------------------------------
    #
    # IMPORTANT:
    # The split happens BEFORE fitting preprocessing
    # operations such as imputation and scaling.
    #
    # This prevents information from the test set leaking
    # into the training process.
    # ---------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded
    )

    print(
        f"[INFO] Training samples before preprocessing: "
        f"{len(X_train)}"
    )

    print(
        f"[INFO] Testing samples before preprocessing: "
        f"{len(X_test)}"
    )

    # ---------------------------------------------------------
    # 8. Identify numerical and categorical columns
    # ---------------------------------------------------------

    numeric_cols = X_train.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_cols = X_train.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    print(
        f"[INFO] Numerical features: "
        f"{len(numeric_cols)}"
    )

    print(
        f"[INFO] Categorical features: "
        f"{len(categorical_cols)}"
    )

    # ---------------------------------------------------------
    # 9. Handle missing numerical values
    # ---------------------------------------------------------
    #
    # Median is calculated ONLY from the training set.
    # The same training median is then applied to the test set.
    # ---------------------------------------------------------

    for col in numeric_cols:

        if X_train[col].isnull().sum() > 0:

            median_value = X_train[col].median()

            X_train[col] = X_train[col].fillna(
                median_value
            )

            X_test[col] = X_test[col].fillna(
                median_value
            )

            print(
                f"[INFO] Imputed '{col}' "
                f"with training median: {median_value}"
            )

    # ---------------------------------------------------------
    # 10. Handle missing categorical values
    # ---------------------------------------------------------

    for col in categorical_cols:

        if X_train[col].isnull().sum() > 0:

            mode_values = X_train[col].mode()

            if len(mode_values) > 0:

                mode_value = mode_values.iloc[0]

                X_train[col] = X_train[col].fillna(
                    mode_value
                )

                X_test[col] = X_test[col].fillna(
                    mode_value
                )

                print(
                    f"[INFO] Imputed '{col}' "
                    f"with training mode: {mode_value}"
                )

    # ---------------------------------------------------------
    # 11. One-hot encode categorical features
    # ---------------------------------------------------------
    #
    # Encoding is performed separately, then columns are aligned
    # so train and test have exactly the same features.
    # ---------------------------------------------------------

    if categorical_cols:

        X_train = pd.get_dummies(
            X_train,
            columns=categorical_cols,
            drop_first=True
        )

        X_test = pd.get_dummies(
            X_test,
            columns=categorical_cols,
            drop_first=True
        )

        # Make sure test has exactly the same columns as train.
        X_test = X_test.reindex(
            columns=X_train.columns,
            fill_value=0
        )

    # ---------------------------------------------------------
    # 12. Standardize numerical features
    # ---------------------------------------------------------
    #
    # StandardScaler is fitted ONLY on training data.
    # ---------------------------------------------------------

    scaler = StandardScaler()

    if numeric_cols:

        X_train[numeric_cols] = scaler.fit_transform(
            X_train[numeric_cols]
        )

        X_test[numeric_cols] = scaler.transform(
            X_test[numeric_cols]
        )

        print("[INFO] Numerical features standardized.")

    # ---------------------------------------------------------
    # 13. Ensure all features are numeric
    # ---------------------------------------------------------

    X_train = X_train.astype(float)
    X_test = X_test.astype(float)

    # ---------------------------------------------------------
    # 14. Apply SMOTE ONLY to training data
    # ---------------------------------------------------------
    #
    # NEVER apply SMOTE to the test set.
    #
    # The test set should represent the real-world class
    # distribution.
    # ---------------------------------------------------------

    print("[INFO] Class distribution before SMOTE:")

    print(
        pd.Series(y_train).value_counts().sort_index()
    )

    smote = SMOTE(
        random_state=random_state
    )

    X_train_resampled, y_train_resampled = (
        smote.fit_resample(
            X_train,
            y_train
        )
    )

    print(
        "[INFO] Applied SMOTE."
    )

    print(
        "[INFO] Training samples after SMOTE: "
        f"{len(X_train_resampled)}"
    )

    print(
        "[INFO] Training class balance after SMOTE:"
    )

    print(
        pd.Series(y_train_resampled)
        .value_counts()
        .sort_index()
        .to_dict()
    )

    # ---------------------------------------------------------
    # 15. Create processed DataFrames
    # ---------------------------------------------------------

    feature_cols = X_train.columns.tolist()

    train_df = pd.DataFrame(
        X_train_resampled,
        columns=feature_cols
    )

    train_df[target_col] = y_train_resampled

    test_df = pd.DataFrame(
        X_test,
        columns=feature_cols
    )

    test_df[target_col] = y_test

    # ---------------------------------------------------------
    # 16. Create output directories
    # ---------------------------------------------------------

    train_directory = os.path.dirname(train_out)
    test_directory = os.path.dirname(test_out)

    if train_directory:
        os.makedirs(
            train_directory,
            exist_ok=True
        )

    if test_directory:
        os.makedirs(
            test_directory,
            exist_ok=True
        )

    # ---------------------------------------------------------
    # 17. Save processed datasets
    # ---------------------------------------------------------

    train_df.to_csv(
        train_out,
        index=False
    )

    test_df.to_csv(
        test_out,
        index=False
    )

    # ---------------------------------------------------------
    # 18. Final information
    # ---------------------------------------------------------

    print(
        f"[INFO] Final training dataset shape: "
        f"{train_df.shape}"
    )

    print(
        f"[INFO] Final testing dataset shape: "
        f"{test_df.shape}"
    )

    print(
        "[INFO] Processed data saved to:"
    )

    print(
        f"    - {train_out}"
    )

    print(
        f"    - {test_out}"
    )


if __name__ == "__main__":

    params = load_params()

    preprocess_data(params)
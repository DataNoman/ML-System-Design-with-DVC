"""
Data Loading Module
-------------------
Loads raw stroke dataset from the data/raw directory.
"""

import os
import pandas as pd
from src import load_params


def load_raw_data(data_path: str) -> pd.DataFrame:
    """Load raw dataset from the given file path."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Raw data file not found at: {data_path}")

    df = pd.read_csv(data_path)
    print(f"[INFO] Raw dataset loaded successfully with shape: {df.shape}")
    return df


if __name__ == "__main__":
    params = load_params()
    raw_data_path = params["data_loader"]["raw_data_path"]
    df = load_raw_data(raw_data_path)
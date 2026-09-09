"""
Stroke Risk Prediction ML Package
---------------------------------
This package contains source modules for data preprocessing,
model training, evaluation, and pipeline utilities.
"""

__version__ = "0.1.0"
__author__ = "DataNoman"

# Convenient top-level imports (Optional)
# Allows importing directly from 'src' (e.g., from src import load_params)
import os
import yaml


def load_params(params_path: str = "params.yaml") -> dict:
    """Helper utility to load pipeline configuration parameters."""
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"Configuration file not found at: {params_path}")
    
    with open(params_path, "r") as file:
        return yaml.safe_load(file)
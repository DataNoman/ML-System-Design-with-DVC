# ML-System-Design-with-DVC

A production-ready machine learning system for stroke risk prediction that uses DVC (Data Version Control) and Git for reproducible, trackable ML workflows.

## Overview

This repository contains an end-to-end ML pipeline implemented with DVC to make experiments reproducible and auditable. The pipeline preprocesses raw data, trains a gradient-boosting model, and evaluates performance. All data artifacts, models, and metrics are tracked with DVC.

Changes in this update
- README rewritten to reflect the current code, dependencies, and measured metrics.
- Installation and configuration instructions clarified.
- Metrics in `metrics.json` updated to the latest evaluation results.

### Dataset
The project uses the Stroke Risk Prediction Dataset. Obtain it from Kaggle:
https://www.kaggle.com/datasets/mobeenfatimah/stroke-risk-prediction-dataset

## Key features
- DVC-managed pipeline (preprocess → train → evaluate)
- Data and model versioning with DVC
- Centralized hyperparameters in `params.yaml`
- Metrics output stored in `metrics.json` (DVC metrics)
- Deterministic runs via fixed random seeds

## Repository layout

```
├── data/
│   ├── raw/                          # Raw dataset (DVC-tracked)
│   └── processed/                    # Preprocessed train/test CSVs
├── src/
│   ├── preprocess.py                 # Data cleaning & preprocessing
│   ├── train.py                      # Model training (XGBoost-compatible)
│   └── evaluate.py                   # Model evaluation & metrics export
├── models/                           # Trained model artifacts (DVC-tracked)
├── dvc.yaml                          # DVC pipeline definition
├── dvc.lock                          # DVC lock file (pinned deps)
├── params.yaml                       # Pipeline parameters and hyperparams
├── metrics.json                      # Latest evaluation metrics (DVC-tracked)
├── requirements.txt                  # Python dependencies (pinned)
├── Project Requirements.pdf          # Project specification
├── .gitignore
└── README.md                         # This file
```

## Pipeline stages

1) Preprocess
- Input: files in `data/raw/` (see `params.yaml`)  
- Output: `data/processed/train.csv`, `data/processed/test.csv`  
- Configurable split (default 80/20) and seed in `params.yaml`

2) Train
- Input: `data/processed/train.csv`  
- Output: model artifact saved to `models/` and tracked by DVC  
- Uses a gradient-boosting estimator (XGBoost or scikit-learn compatible wrapper). Hyperparameters are in `params.yaml`.

3) Evaluate
- Input: `data/processed/test.csv`, trained model  
- Output: `metrics.json` with evaluation results  
- Metrics are printed and saved so DVC can track them across experiments

## Latest metrics

The metrics file (metrics.json) currently contains the following results from the latest evaluation run:

{
  "accuracy": 0.9989,
  "precision": 0.9985,
  "recall": 0.9948,
  "f1_score": 0.9966,
  "roc_auc": 1.0
}

Note: If metrics look unexpectedly high, check for target leakage, data duplication between train/test splits, or evaluation on a non-representative test set.

## Requirements

This project was developed and tested with the following (see `requirements.txt` for exact pinned versions):

- Python >= 3.8
- pandas >= 2.0
- numpy >= 1.24
- scikit-learn >= 1.3
- xgboost >= 2.0
- imbalanced-learn >= 0.11
- joblib >= 1.3
- pyyaml >= 6.0
- dvc >= 3.0
- dvc-gdrive (optional) for Google Drive remotes

If you need reproducibility across environments, use the versions pinned in `requirements.txt`.

## Installation

1. Clone the repo:

```bash
git clone https://github.com/DataNoman/ML-System-Design-with-DVC.git
cd ML-System-Design-with-DVC
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

3. Install Python dependencies:

```bash
pip install -r requirements.txt
```

4. Install DVC and configure a remote (example for Google Drive):

```bash
pip install dvc[dvc_gdrive]
# configure remote per your DVC remote provider
# dvc remote add -d myremote <remote-url>
```

## Usage

Reproduce the full pipeline:

```bash
dvc repro
```

Re-run a specific stage:

```bash
dvc repro dvc.yaml:preprocess
dvc repro dvc.yaml:train
dvc repro dvc.yaml:evaluate
```

Show pipeline DAG:

```bash
dvc dag
```

Display tracked metrics:

```bash
dvc metrics show
```

## Configuration

Edit `params.yaml` to change preprocessing and training settings. Example:

```yaml
preprocess:
  raw_data_path: data/raw/stroke_risk_prediction_dataset.csv
  test_size: 0.2
  random_state: 42

train:
  n_estimators: 100
  max_depth: 5
  learning_rate: 0.05
  random_state: 42

evaluate:
  # add evaluation-specific settings if needed
```

## Reproducibility notes & best practices

- Use `params.yaml` for hyperparameter and run configuration.  
- Keep raw data DVC-tracked and do not commit large data to Git.  
- Use `dvc.lock` and DVC remotes to reproduce experiments across machines.  
- Validate that train/test splits are properly stratified and have no overlap.

## Recommendations

- Use k-fold cross-validation and report mean±std for metrics before deploying.  
- Add automated CI checks that run lightweight pipeline stages on PRs.  
- Add experiment tracking (MLflow/Weights & Biases) if you need more advanced experiment analytics.

## Future work

- Hyperparameter tuning (Optuna or grid search)
- Feature importance and explainability reports
- Additional models and ensemble strategies
- Containerized model serving (Docker + FastAPI)
- CI for linting, unit tests, and lightweight pipeline repro

## Contributing

Contributions welcome — please fork, create a branch, and open a pull request. Include tests and update documentation for new features.

## License

MIT License

## Author

DataNoman

---

**Last Updated**: 2026-09-15

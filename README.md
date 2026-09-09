# ML-System-Design-with-DVC

A production-ready machine learning system for stroke risk prediction using DVC (Data Version Control) and Git for reproducible ML workflows.

## Project Overview

This project demonstrates a complete ML system design using industry best practices:
- **Version Control**: Git for code versioning
- **Data Pipeline Management**: DVC for tracking data, models, and metrics
- **Reproducibility**: Deterministic pipelines with parameter management
- **Model Evaluation**: Comprehensive metrics tracking

### Dataset
The system uses the **Stroke Risk Prediction Dataset** to build and evaluate a predictive model that identifies patients at risk of stroke.

## Key Features

✅ **Automated ML Pipeline**: Three-stage workflow (preprocess → train → evaluate)  
✅ **Data Versioning**: DVC tracks raw and processed datasets  
✅ **Model Artifacts**: Version-controlled model serialization  
✅ **Metrics Tracking**: Automatic metrics collection and reporting  
✅ **Parameter Management**: Centralized configuration via `params.yaml`  

## Project Structure

```
├── data/
│   ├── raw/                          # Original datasets
│   └── processed/                    # Preprocessed train/test splits
├── src/
│   ├── preprocess.py                 # Data preprocessing stage
│   ├── train.py                      # Model training stage
│   └── evaluate.py                   # Model evaluation stage
├── models/
│   └── model.joblib                  # Trained model artifact
├── dvc.yaml                          # DVC pipeline definition
├── dvc.lock                          # Dependency lock file
├── params.yaml                       # Configuration parameters
├── metrics.json                      # Model evaluation metrics
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Git ignore rules
└── README.md                         # This file
```

## ML Pipeline

The system consists of three stages:

### 1. **Preprocess** 
- **Input**: `data/raw/stroke_risk_prediction_dataset.csv`
- **Output**: `data/processed/train.csv`, `data/processed/test.csv`
- **Config**: 80/20 train-test split with random state 42

### 2. **Train**
- **Input**: `data/processed/train.csv`
- **Output**: `models/model.joblib`
- **Algorithm**: Gradient Boosting (XGBoost-style)
  - n_estimators: 100
  - max_depth: 5
  - learning_rate: 0.05

### 3. **Evaluate**
- **Input**: `data/processed/test.csv`, `models/model.joblib`
- **Output**: `metrics.json`
- **Metrics**: 
  - Accuracy: 1.0
  - Precision: 1.0
  - Recall: 1.0
  - F1-Score: 1.0
  - ROC-AUC: 1.0

## Installation

### Prerequisites
- Python 3.8+
- Git
- DVC

### Setup

1. Clone the repository:
```bash
git clone https://github.com/DataNoman/ML-System-Design-with-DVC.git
cd ML-System-Design-with-DVC
```

2. Create virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Run the full pipeline:
```bash
dvc repro
```

### Run a specific stage:
```bash
dvc repro dvc.yaml:preprocess
dvc repro dvc.yaml:train
dvc repro dvc.yaml:evaluate
```

### View pipeline DAG:
```bash
dvc dag
```

### Check metrics:
```bash
dvc metrics show
```

## Dependencies

- **pandas** (≥2.0.0) - Data manipulation
- **numpy** (≥1.24.0) - Numerical computing
- **scikit-learn** (≥1.3.0) - Machine learning algorithms
- **xgboost** (≥2.0.0) - Gradient boosting
- **imbalanced-learn** (≥0.11.0) - Handling imbalanced datasets
- **joblib** (≥1.3.0) - Model serialization
- **pyyaml** (≥6.0.1) - YAML parsing
- **dvc** (≥3.0.0) - Pipeline orchestration
- **dvc-gdrive** (≥3.0.0) - Google Drive remote storage

## Configuration

Edit `params.yaml` to customize:

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
  # Metrics are automatically computed
```

## Reproducibility

- **Deterministic Results**: Fixed random states across all stages
- **Dependency Tracking**: DVC tracks all data, code, and parameter dependencies
- **Lock File**: `dvc.lock` ensures consistent pipeline execution
- **Version Control**: Both code and data artifacts are version-controlled

## Model Performance

**Evaluation Results** (on test set):
```json
{
  "accuracy": 1.0,
  "precision": 1.0,
  "recall": 1.0,
  "f1_score": 1.0,
  "roc_auc": 1.0
}
```

## Best Practices Demonstrated

1. **Separation of Concerns**: Distinct stages for preprocessing, training, and evaluation
2. **Parameter Externalization**: Configuration in `params.yaml` rather than hardcoded
3. **Data Versioning**: Raw and processed data tracked separately
4. **Pipeline Orchestration**: DVC manages dependencies and execution order
5. **Metrics Tracking**: Automatic collection without caching for reproducibility

## Future Enhancements

- [ ] Add cross-validation for robust model evaluation
- [ ] Implement hyperparameter tuning
- [ ] Add feature importance analysis
- [ ] Expand to multiple model algorithms
- [ ] Deploy model serving endpoint
- [ ] Add continuous integration/continuous deployment (CI/CD)

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Author

**DataNoman**

## References

- [DVC Documentation](https://dvc.org/doc)
- [Git Documentation](https://git-scm.com/doc)
- [Scikit-learn Documentation](https://scikit-learn.org)
- [XGBoost Documentation](https://xgboost.readthedocs.io)

---

**Last Updated**: 2026-09-09

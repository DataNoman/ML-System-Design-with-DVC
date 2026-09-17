````markdown
# Stroke Risk Prediction — ML System Design with DVC

An end-to-end machine learning system for predicting **stroke risk levels** using multiple classification models, with a reproducible **DVC pipeline** for data preprocessing, model training, evaluation, and model comparison.

The project is designed to demonstrate practical **Machine Learning System Design**, experiment reproducibility, data versioning, model versioning, and automated ML pipelines using **DVC** and **Git**.

---

## 📌 Project Overview

Stroke is a major health concern, and machine learning can be used to identify patterns associated with different levels of stroke risk.

This project builds a classification system that predicts one of three stroke-risk categories:

- **Low**
- **Moderate**
- **High**
https://www.kaggle.com/datasets/mobeenfatimah/stroke-risk-prediction-dataset

Instead of relying on a single model, the system trains and evaluates **five different classification algorithms** using the same preprocessing and train/test split:

1. XGBoost
2. Random Forest
3. LightGBM
4. CatBoost
5. Logistic Regression

The complete workflow is managed using **DVC**, allowing individual models to be reproduced independently or the entire pipeline to be executed at once.

---

## 🎯 Project Objectives

The main objectives of this project are:

- Build a reproducible machine learning pipeline.
- Version datasets and machine learning artifacts using DVC.
- Separate preprocessing, training, and evaluation into independent pipeline stages.
- Compare multiple machine learning algorithms using the same processed dataset.
- Track model parameters using `params.yaml`.
- Store large model artifacts using DVC rather than Git.
- Generate reproducible evaluation reports.
- Demonstrate DVC commands such as:
  - `dvc init`
  - `dvc add`
  - `dvc repro`
  - `dvc status`
  - `dvc dag`
  - `dvc push`
  - `dvc pull`

---

# 📊 Dataset

The project uses a stroke-risk prediction dataset containing:

- **50,000 records**
- **40 columns**

The target variable is:

```text
Stroke_Risk
````

The target contains three classes:

| Class     | Number of Samples | Percentage |
| --------- | ----------------: | ---------: |
| Moderate  |            34,375 |    68.750% |
| High      |            12,112 |    24.224% |
| Low       |             3,513 |     7.026% |
| **Total** |        **50,000** |   **100%** |

There are no exact duplicate rows in the original dataset.

---

# ⚠️ Data Leakage Prevention

One of the important parts of this project was identifying and removing a potentially direct target-leakage feature:

```text
Stroke_Risk_Score
```

This feature had values that directly corresponded to the target categories:

* High risk: approximately 65–100
* Moderate risk: approximately 35–64
* Low risk: approximately 2–34

Because this feature effectively encoded the target, it was removed before model training.

The following identifier was also removed:

```text
Patient_ID
```

This prevents the model from using an identifier as a predictive feature.

---

# 🔄 Data Preprocessing Pipeline

The preprocessing stage is implemented in:

```text
src/preprocess.py
```

The preprocessing workflow is:

```text
Raw Dataset
     │
     ▼
Remove Patient_ID
     │
     ▼
Remove Stroke_Risk_Score
     │
     ▼
Train/Test Split
     │
     ├───────────────┐
     ▼               ▼
Training Data     Test Data
     │               │
     ▼               │
Imputation          │
     │               │
     ▼               │
Categorical         │
Encoding            │
     │               │
     ▼               │
Feature Scaling     │
     │               │
     ▼               │
SMOTE               │
     │               │
     ▼               │
Balanced Training   │
     │               │
     └───────┬───────┘
             ▼
       Processed Data
```

### Preprocessing Details

The pipeline:

1. Loads the raw dataset.
2. Removes identifier columns.
3. Removes `Stroke_Risk_Score` to prevent target leakage.
4. Splits the data into training and test sets.
5. Encodes the target variable.
6. Identifies numerical and categorical features.
7. Performs missing-value imputation.
8. Performs one-hot encoding on categorical variables.
9. Standardizes numerical features.
10. Aligns training and test feature columns.
11. Applies **SMOTE only to the training dataset**.

The train/test split is performed **before fitting preprocessing transformations**, preventing information from the test set from influencing preprocessing.

---

## 📦 Processed Dataset

The dataset is split using:

```yaml
test_size: 0.2
random_state: 42
```

Result:

```text
Training samples: 40,000
Test samples:     10,000
```

Before preprocessing, there are:

```text
16 numerical features
21 categorical features
```

After preprocessing:

```text
83 features
```

SMOTE increases the training dataset to:

```text
82,500 samples
```

with equal representation of the three classes:

```text
27,500 High
27,500 Low
27,500 Moderate
```

---

# 🤖 Machine Learning Models

The project evaluates five classification algorithms.

## 1. XGBoost

Implementation:

```text
src/train_xgboost.py
```

Main parameters:

```yaml
n_estimators: 100
max_depth: 5
learning_rate: 0.05
random_state: 42
```

---

## 2. Random Forest

Implementation:

```text
src/train_random_forest.py
```

Main parameters:

```yaml
n_estimators: 100
max_depth: 10
random_state: 42
n_jobs: -1
```

---

## 3. LightGBM

Implementation:

```text
src/train_lightgbm.py
```

Main parameters:

```yaml
n_estimators: 100
max_depth: 7
learning_rate: 0.05
random_state: 42
n_jobs: -1
```

---

## 4. CatBoost

Implementation:

```text
src/train_catboost.py
```

Main parameters:

```yaml
iterations: 100
depth: 7
learning_rate: 0.05
random_state: 42
verbose: false
```

---

## 5. Logistic Regression

Implementation:

```text
src/train_logistic.py
```

Main parameters:

```yaml
C: 1.0
max_iter: 1000
random_state: 42
```

---

# 🏗️ System Architecture

The overall system is organized as:

```text
                    ┌──────────────────────┐
                    │    Raw Dataset       │
                    │      50,000 rows     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │     Preprocessing    │
                    │                      │
                    │ • Leakage removal    │
                    │ • Train/test split   │
                    │ • Imputation         │
                    │ • Encoding           │
                    │ • Scaling            │
                    │ • SMOTE              │
                    └──────────┬───────────┘
                               │
                    Processed Training Data
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
          ▼                    ▼                    ▼
     ┌─────────┐          ┌──────────┐        ┌─────────┐
     │ XGBoost │          │ Random   │        │ LightGBM│
     │         │          │ Forest   │        │         │
     └────┬────┘          └────┬─────┘        └────┬────┘
          │                    │                   │
          ▼                    ▼                   ▼
      Evaluation          Evaluation          Evaluation

          │                    │                   │
          └──────────────┬─────┴──────┬────────────┘
                         │
               ┌─────────┴─────────┐
               │                   │
               ▼                   ▼
          ┌─────────┐       ┌───────────────┐
          │CatBoost │       │Logistic Reg.  │
          └────┬────┘       └───────┬───────┘
               │                    │
               ▼                    ▼
          Evaluation           Evaluation
               │                    │
               └──────────┬─────────┘
                          │
                          ▼
                ┌─────────────────────┐
                │  Model Comparison   │
                └─────────────────────┘
                          │
                          ▼
                model_comparison.csv
```

---

# 🔁 DVC Pipeline

The entire ML workflow is managed by:

```text
dvc.yaml
```

The pipeline contains the following stages.

### Preprocessing

```text
preprocess
```

Runs:

```bash
python -m src.preprocess
```

Produces:

```text
data/processed/train.csv
data/processed/test.csv
```

---

### Model Training

Five independent training stages are available:

```text
train_xgboost
train_random_forest
train_lightgbm
train_catboost
train_logistic
```

Each stage depends on the processed training dataset and its model-specific parameters.

---

### Model Evaluation

Each model has a separate evaluation stage:

```text
evaluate_xgboost
evaluate_random_forest
evaluate_lightgbm
evaluate_catboost
evaluate_logistic
```

The common evaluation implementation is:

```text
src/evaluate_model.py
```

Each evaluation stage generates:

* Accuracy
* Macro Precision
* Macro Recall
* Macro F1
* Macro ROC-AUC
* Confusion Matrix
* Classification Report

---

### Model Comparison

The final stage is:

```text
compare_models
```

Implementation:

```text
src/compare_models.py
```

It reads the metrics generated by all five evaluation stages and produces:

```text
reports/model_comparison.csv
```

---

# 🌳 DVC Pipeline DAG

The pipeline can be visualized using:

```bash
dvc dag
```

Conceptually:

```text
                         preprocess
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
          ▼                  ▼                  ▼
    train_xgboost      train_random_forest   train_lightgbm
          │                  │                  │
          ▼                  ▼                  ▼
    evaluate_xgboost   evaluate_random_forest evaluate_lightgbm
          │                  │                  │
          │                  │                  │
          ├──────────────────┼──────────────────┤
          │                  │
          ▼                  ▼
    train_catboost     train_logistic
          │                  │
          ▼                  ▼
    evaluate_catboost evaluate_logistic
          │                  │
          └──────────┬───────┘
                     ▼
              compare_models
```

---

# ⚙️ Running the Project

## 1. Clone the Repository

```bash
git clone https://github.com/DataNoman/ML-System-Design-with-DVC.git
cd ML-System-Design-with-DVC
```

---

## 2. Create a Virtual Environment

Windows PowerShell:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

---

## 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

The project uses:

* Python
* Pandas
* NumPy
* SciPy
* Scikit-learn
* XGBoost
* LightGBM
* CatBoost
* Imbalanced-learn
* Joblib
* PyYAML
* DVC
* DVC Google Drive plugin

---

# 📥 Retrieve DVC Data and Models

After cloning the repository, use:

```bash
dvc pull
```

This retrieves DVC-managed data and model artifacts from the configured remote storage.

---

# ▶️ Running the Complete Pipeline

To reproduce the entire machine learning pipeline:

```bash
dvc repro
```

DVC automatically determines which stages need to be executed based on dependencies, parameters, outputs, and pipeline state.

If nothing has changed:

```text
Data and pipelines are up to date.
```

---

# 🎯 Running an Individual Model

Because each model has its own DVC stages, individual experiments can be reproduced without running the entire pipeline.

For example:

### XGBoost

```bash
dvc repro evaluate_xgboost
```

### Random Forest

```bash
dvc repro evaluate_random_forest
```

### LightGBM

```bash
dvc repro evaluate_lightgbm
```

### CatBoost

```bash
dvc repro evaluate_catboost
```

### Logistic Regression

```bash
dvc repro evaluate_logistic
```

DVC automatically runs the required upstream training stage when necessary.

---

# 🔍 Checking Pipeline Status

Use:

```bash
dvc status
```

Example:

```text
Data and pipelines are up to date.
```

To check synchronization with the DVC remote:

```bash
dvc status -c
```

---

# 🌳 Viewing the Pipeline

Use:

```bash
dvc dag
```

This displays the dependency graph of the ML pipeline.

---

# 📤 Uploading DVC Artifacts

After generating or changing DVC-managed outputs:

```bash
dvc push
```

This uploads the corresponding artifacts to the configured DVC remote storage.

---

# 📥 Downloading DVC Artifacts

On another machine:

```bash
dvc pull
```

This retrieves the required DVC-managed data and model artifacts.

---

# 📈 Evaluation Results

The complete pipeline was executed using DVC and all five models were trained and evaluated.

The resulting comparison was:

| Model               | Accuracy | Precision | Recall |     F1 | ROC-AUC |
| ------------------- | -------: | --------: | -----: | -----: | ------: |
| XGBoost             |   0.9989 |    0.9985 | 0.9948 | 0.9966 |  1.0000 |
| Random Forest       |   0.9981 |    0.9991 | 0.9910 | 0.9950 |  0.9991 |
| LightGBM            |   0.9989 |    0.9985 | 0.9948 | 0.9966 |  1.0000 |
| CatBoost            |   0.9989 |    0.9985 | 0.9948 | 0.9966 |  1.0000 |
| Logistic Regression |   1.0000 |    1.0000 | 1.0000 | 1.0000 |  1.0000 |

The detailed comparison is generated automatically at:

```text
reports/model_comparison.csv
```

Individual evaluation outputs are generated under:

```text
reports/
```

---

# ⚠️ Results Interpretation

The evaluation scores are extremely high across all five models.

These results should be interpreted as **experimental results on the project's held-out test set**, rather than evidence that the models will achieve the same performance on real-world clinical data.

The project specifically removes the identified `Stroke_Risk_Score` leakage feature, but the unusually high performance indicates that further investigation would be appropriate before deploying such a model in a real-world setting.

Possible future validation could include:

* Testing on an independent external dataset
* More extensive cross-validation
* Feature-level leakage investigation
* Calibration analysis
* Additional feature auditing
* Hyperparameter optimization
* External validation on data collected from a different population

---

# 📁 Project Structure

```text
ML-System-Design-with-DVC/
│
├── data/
│   ├── raw/
│   │   └── stroke_risk_prediction_dataset.csv
│   │
│   └── processed/
│       ├── train.csv
│       └── test.csv
│
├── models/
│   ├── .gitignore
│   ├── xgboost.joblib
│   ├── random_forest.joblib
│   ├── lightgbm.joblib
│   ├── catboost.joblib
│   └── logistic_regression.joblib
│
├── reports/
│   ├── .gitignore
│   ├── xgboost_confusion_matrix.csv
│   ├── xgboost_classification_report.txt
│   ├── random_forest_confusion_matrix.csv
│   ├── random_forest_classification_report.txt
│   ├── lightgbm_confusion_matrix.csv
│   ├── lightgbm_classification_report.txt
│   ├── catboost_confusion_matrix.csv
│   ├── catboost_classification_report.txt
│   ├── logistic_regression_confusion_matrix.csv
│   ├── logistic_regression_classification_report.txt
│   └── model_comparison.csv
│
├── metrics/
│   ├── xgboost.json
│   ├── random_forest.json
│   ├── lightgbm.json
│   ├── catboost.json
│   └── logistic_regression.json
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── train_xgboost.py
│   ├── train_random_forest.py
│   ├── train_lightgbm.py
│   ├── train_catboost.py
│   ├── train_logistic.py
│   ├── evaluate_model.py
│   └── compare_models.py
│
├── .dvc/
├── .dvcignore
├── .gitignore
├── dvc.yaml
├── dvc.lock
├── params.yaml
├── requirements.txt
└── README.md
```

---

# 🧩 Configuration

Model and pipeline parameters are stored in:

```text
params.yaml
```

This separates configuration from source code and allows DVC to detect parameter changes.

Example:

```yaml
models:
  xgboost:
    n_estimators: 100
    max_depth: 5
    learning_rate: 0.05
    random_state: 42
```

Changing a tracked parameter can cause DVC to identify the affected pipeline stage and reproduce it.

---

# 🗃️ Data and Model Versioning

Git is used for versioning:

* Source code
* Configuration
* DVC pipeline definitions
* DVC lock file
* Project documentation
* Evaluation metrics

DVC is used for large machine-learning artifacts such as:

* Raw dataset
* Processed datasets
* Trained model files

The actual large model files are therefore not committed directly to Git.

---

# ☁️ DVC Remote Storage

The project uses a **Google Drive DVC remote** for storing DVC-managed artifacts.

The remote can be checked with:

```bash
dvc remote list
```

To synchronize artifacts:

```bash
dvc push
```

and:

```bash
dvc pull
```

Authentication credentials should be configured locally and **must not be committed to the repository**.

---

# 🔬 Reproducibility

The project is designed so that the same pipeline can be reproduced from the repository.

The important reproducibility components are:

```text
Git
 │
 ├── Source Code
 ├── params.yaml
 ├── dvc.yaml
 └── dvc.lock
        │
        ▼
       DVC
        │
        ├── Dataset
        ├── Processed Data
        └── Model Artifacts
```

`dvc.lock` records the exact state of the pipeline dependencies and outputs.

---

# 🧪 Example Workflow

A typical experiment can follow this workflow:

```bash
# Check the repository
git status

# Check DVC
dvc status

# View the pipeline
dvc dag

# Reproduce the complete pipeline
dvc repro

# Compare model results
type reports\model_comparison.csv

# Upload updated DVC artifacts
dvc push

# Commit project changes
git add .
git commit -m "Update ML pipeline and model experiments"

# Push Git changes
git push
```

---

# 🛠️ Technologies Used

| Technology       | Purpose                                  |
| ---------------- | ---------------------------------------- |
| Python           | Main programming language                |
| Pandas           | Data processing                          |
| NumPy            | Numerical computing                      |
| Scikit-learn     | ML preprocessing and Logistic Regression |
| XGBoost          | Gradient boosting model                  |
| LightGBM         | Gradient boosting model                  |
| CatBoost         | Gradient boosting model                  |
| Imbalanced-learn | SMOTE                                    |
| Joblib           | Model serialization                      |
| PyYAML           | Configuration management                 |
| Git              | Source-code version control              |
| DVC              | Data, model, and pipeline versioning     |
| Google Drive     | DVC remote storage                       |

---

# 📚 Key DVC Commands Demonstrated

| Command             | Purpose                            |
| ------------------- | ---------------------------------- |
| `dvc init`          | Initialize DVC                     |
| `dvc add`           | Track data with DVC                |
| `dvc repro`         | Reproduce pipeline                 |
| `dvc repro <stage>` | Reproduce a specific stage         |
| `dvc status`        | Check pipeline status              |
| `dvc status -c`     | Check cache/remote synchronization |
| `dvc dag`           | Display pipeline dependency graph  |
| `dvc push`          | Upload DVC artifacts               |
| `dvc pull`          | Download DVC artifacts             |

---

# 🚀 Future Improvements

Potential improvements to the system include:

* Hyperparameter optimization
* Cross-validation
* Model calibration
* Feature importance analysis
* SHAP-based model explainability
* External dataset validation
* Experiment tracking
* Automated testing
* CI/CD integration
* Model deployment through a web application
* Prediction API
* Monitoring for data drift and model performance

---

# 👨‍💻 Author

**DataNoman**

Machine Learning / Data Science project focused on reproducible ML system design using Python, Git, and DVC.

---

## 📄 License

This project is intended for educational and machine-learning system design purposes.

```
```

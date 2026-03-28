# Handwritten Digit Classification with MLflow Experiment Tracking

## Overview

This lab demonstrates MLflow experiment tracking, model comparison, hyperparameter tuning, and model registry usage with a handwritten digit classification problem.

**Dataset**: Digits (1,797 samples, 64 features, 10 classes — digits 0–9)
**Models**: KNN (K-Nearest Neighbors), SVM (Support Vector Machine)
**MLflow Features**: Experiment Tracking, Model Registry, Artifact Logging, Hyperparameter Tuning

## Improvements Made

### 1. Different Dataset

**Original**: Wine Quality and Diabetes datasets
**Improved**: Digits dataset (1,797 samples of 8x8 pixel handwritten digits)

- Consistent with Docker and FastAPI labs
- Image classification domain
- Multiclass problem (10 classes)

### 2. Multiple Model Comparison

**Original**: Single model per script
**Improved**:

- Trains KNN and SVM as baseline models
- Logs all experiments to MLflow
- Compares models side-by-side in UI
- Creates comparison bar chart visualization

### 3. Hyperparameter Tuning with Tracking

**Original**: Fixed hyperparameters
**Improved**:

- Grid search across KNN parameter space (n_neighbors, weights, metric)
- Logs all 16 parameter combinations as nested runs
- Tracks CV scores for each configuration
- Selects and logs best parameters

### 4. Model Registry Integration

**Original**: Models logged but not registered
**Improved**:

- Registers best model as "DigitsClassifier"
- Versions models automatically
- Transitions model to Production stage

### 5. Artifact Logging

**Original**: Only model artifacts
**Improved**:

- Confusion matrix heatmaps (10x10 for all digit classes)
- Model comparison bar charts
- Classification reports as text artifacts

### 6. Feature Scaling

**Original**: No preprocessing
**Improved**:

- StandardScaler applied before training
- Critical for both KNN (distance-based) and SVM (margin-based)

## Project Structure

```
MLflow_Labs/Lab_1/
├── digits_mlflow.py
├── requirements.txt
├── README.md
├── linear_regression.py      (original, unchanged)
├── linear_regression.ipynb   (original, unchanged)
├── serving.py                (original, unchanged)
├── serving.ipynb             (original, unchanged)
├── starter.ipynb             (original, unchanged)
└── mlruns/                   (created after running)
```

## How to Run

### Step 1: Install Dependencies

```bash
cd MLflow_Labs/Lab_1

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

### Step 2: Run the Script

```bash
python digits_mlflow.py
```

Expected output:

```
INFO - Loading Digits dataset...
INFO - Data split: Train=1437, Test=360
INFO - Training baseline models...
INFO - KNN: Accuracy=0.9861, F1=0.9860
INFO - SVM: Accuracy=0.9889, F1=0.9889
INFO - Starting hyperparameter tuning for KNN...
INFO - Best params: {'metric': 'euclidean', 'n_neighbors': 3, 'weights': 'distance'}
INFO - Best model: Accuracy=0.9889, F1=0.9889
INFO - PIPELINE COMPLETE
INFO - View results: mlflow ui --port 5001
```

### Step 3: View Results in MLflow UI

```bash
mlflow ui --port 5001
```

Open `http://127.0.0.1:5001` in your browser.

You will see:

- **digits_classification** experiment with all runs
- Baseline runs for KNN and SVM
- Model_Comparison run with bar chart artifact
- KNN_GridSearch run with 16 nested runs for each parameter combo
- Registered model "DigitsClassifier" in the Models tab

## Troubleshooting

**MLflow UI won't start** — Port 5000 may be in use (macOS AirPlay). Use `mlflow ui --port 5001`.

**"No module named mlflow"** — Run `pip install mlflow`.

**Model Registry not showing** — Use SQLite backend:
```bash
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
python digits_mlflow.py
```

**Plots not appearing in artifacts** — Ensure matplotlib and seaborn are installed: `pip install matplotlib seaborn`.

## Commands Reference

```bash
# Run the pipeline
python digits_mlflow.py

# Start MLflow UI
mlflow ui --port 5001

# List experiments
mlflow experiments list

# Search runs
mlflow runs list --experiment-id 0
```
# Digits Classification with ML Metadata Pipeline Tracking

## Overview

This lab demonstrates how to use [ML Metadata (MLMD)](https://www.tensorflow.org/tfx/guide/mlmd) independently of TFX to manually track a complete ML pipeline. Starting from the original course walkthrough (which only covered data validation on the Chicago Taxi dataset), this lab has been significantly extended with a different dataset, additional pipeline stages, and persistent storage.

### Key Modifications from Original Lab

| Aspect | Original Lab | Modified Lab |
|--------|-------------|--------------|
| **Dataset** | Chicago Taxi (10K+ rows, 18 features) | Digits (1,797 samples, 64 pixel features + target, 10 classes) |
| **Pipeline stages** | 1 (Data Validation only) | 4 (Data Validation, Anomaly Detection, Model Training, Model Evaluation) |
| **Artifact types** | 3 (DataSet, Schema, statistics) | 6 (DataSet, Schema, statistics, Anomalies, Model, ModelEvaluation) |
| **Execution types** | 1 (Data Validation) | 4 (Data Validation, Anomaly Detection, Model Training, Model Evaluation) |
| **Storage backend** | In-memory fake database | Persistent SQLite database (`metadata/mlmd.sqlite`) |
| **ML model** | None | KNN classifier (scikit-learn) with StandardScaler |
| **Evaluation metrics** | None | Accuracy, F1 score, Precision, Recall (weighted) |
| **Anomaly detection** | None | TFDV schema validation on eval data |

---

## Dataset: Scikit-learn Digits

- **Source**: [Scikit-learn Digits dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#optical-recognition-of-handwritten-digits-dataset)
- **Samples**: 1,797 total
- **Target**: Multiclass classification — digits 0 through 9
- **Features**: 64 pixel intensity values (8x8 images flattened), named `pixel_0` through `pixel_63`

### Data Splits

| Split | Rows | Purpose |
|-------|------|---------|
| Train (60%) | 1,078 | Schema inference, model training |
| Eval (20%) | 359 | Anomaly detection, model evaluation |
| Serving (20%) | 360 | Inference simulation |

All splits are generated with `random_state=42` for reproducibility. The notebook generates the data at runtime from scikit-learn.

---

## Pipeline Architecture

```
                          ┌──────────────────┐
                          │  Dataset (train)  │
                          └────────┬─────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    ▼              ▼               │
          ┌─────────────┐  ┌─────────────┐        │
          │    Data      │  │    Model    │        │
          │  Validation  │  │  Training   │        │
          └──────┬───────┘  └──────┬──────┘        │
                 │                 │                │
                 ▼                 ▼                │
          ┌──────────┐     ┌──────────┐            │
          │  Schema  │     │  Model   │            │
          └────┬─────┘     └────┬─────┘            │
               │                │                  │
               ▼                │       ┌──────────┘
        ┌─────────────┐        │       │
        │   Anomaly   │◄───────┼───────┘
        │  Detection  │        │   Dataset (eval)
        └──────┬──────┘        │
               │               ▼
               ▼        ┌─────────────┐
        ┌───────────┐   │    Model    │◄── Dataset (eval)
        │ Anomalies │   │ Evaluation  │
        └───────────┘   └──────┬──────┘
                               │
                               ▼
                      ┌─────────────────┐
                      │ ModelEvaluation  │
                      │ (metrics)        │
                      └─────────────────┘
```

All artifacts and executions are grouped under a single **Experiment Context** (`Digits Classification Pipeline`).

---

## Project Structure

```
Labs/MLMD_Labs/Lab1/
├── C2_W3_Lab_1_MLMetadata.ipynb   # Main notebook
├── schema.pbtxt                    # Generated — Digits schema
├── README.md                       # This file
├── data/
│   ├── train/
│   │   └── data.csv               # Training samples
│   ├── eval/
│   │   └── data.csv               # Evaluation samples
│   └── serving/
│       └── data.csv               # Serving samples
├── img/
│   └── mlmd_overview.png           # MLMD architecture diagram
├── model/                          # Generated at runtime
│   ├── model.pkl                   # Trained KNN model
│   ├── scaler.pkl                  # StandardScaler
│   └── eval_metrics.json           # Evaluation metrics (JSON)
├── metadata/                       # Generated at runtime
│   └── mlmd.sqlite                 # Persistent MLMD database
└── anomalies.pbtxt                 # Generated at runtime
```

---

## Requirements

- Python 3.9+
- TensorFlow 2.x
- TensorFlow Data Validation (TFDV)
- ML Metadata (`ml-metadata`)
- scikit-learn
- pandas

```bash
pip install tensorflow tensorflow-data-validation ml-metadata scikit-learn pandas
```

---

## How to Run

1. Open `C2_W3_Lab_1_MLMetadata.ipynb` in Jupyter Notebook or JupyterLab
2. Run all cells sequentially (Kernel -> Restart & Run All)
3. The notebook will:
   - Load and split the Digits dataset from scikit-learn
   - Create a SQLite metadata store at `metadata/mlmd.sqlite`
   - Run all 4 pipeline stages with full MLMD tracking
   - Demonstrate lineage queries tracing artifacts back through the pipeline
4. After running, the SQLite database persists at `metadata/mlmd.sqlite` for future queries

---

## Troubleshooting

**"No module named ml_metadata"** — Run `pip install ml-metadata`

**"No module named tensorflow_data_validation"** — Run `pip install tensorflow-data-validation`

**SQLite database locked** — Restart the kernel to release the connection, then re-run

**TFDV warnings about apache-beam** — These are safe to ignore; TFDV still works correctly

---

## References

- [ML Metadata Documentation](https://www.tensorflow.org/tfx/guide/mlmd)
- [MLMD API Reference](https://www.tensorflow.org/tfx/ml_metadata/api_docs/python/mlmd/MetadataStore)
- [TFDV Documentation](https://www.tensorflow.org/tfx/data_validation/get_started)
- [Scikit-learn Digits Dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#optical-recognition-of-handwritten-digits-dataset)

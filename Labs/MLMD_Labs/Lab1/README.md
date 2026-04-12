# Wine Classification with ML Metadata Pipeline Tracking

## Overview

This lab demonstrates the core concepts of [ML Metadata (MLMD)](https://www.tensorflow.org/tfx/guide/mlmd) by implementing the same data model — ArtifactTypes, Artifacts, ExecutionTypes, Executions, Events, Contexts, Attributions, and Associations — using a **lightweight SQLite-based metadata store**. This approach avoids TensorFlow/TFDV dependency issues while demonstrating the exact same tracking principles.

### Key Modifications from Original Lab

| Aspect | Original Lab | Modified Lab |
|--------|-------------|--------------|
| **Dataset** | Chicago Taxi (10K+ rows, 18 features) | Wine (178 samples, 13 chemical features, 3 classes) |
| **Pipeline stages** | 1 (Data Validation only) | 4 (Data Validation, Anomaly Detection, Model Training, Model Evaluation) |
| **Artifact types** | 3 (DataSet, Schema, statistics) | 6 (DataSet, Schema, Statistics, Anomalies, Model, ModelEvaluation) |
| **Execution types** | 1 (Data Validation) | 4 (Data Validation, Anomaly Detection, Model Training, Model Evaluation) |
| **Storage backend** | In-memory fake database | Persistent SQLite database (`metadata/mlmd.sqlite`) |
| **ML model** | None | SVM classifier (scikit-learn) with StandardScaler |
| **Schema inference** | TFDV | Custom pandas-based (dtype, min/max, nulls, unique counts) |
| **Anomaly detection** | None | Custom range/null validation against schema |
| **Evaluation metrics** | None | Accuracy, F1 score, Precision, Recall (weighted) |
| **Dependencies** | TensorFlow, TFDV, ml-metadata | pandas, scikit-learn, numpy (no TF required) |

---

## Dataset: Scikit-learn Wine

- **Source**: [Scikit-learn Wine dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#wine-recognition-dataset)
- **Samples**: 178 total
- **Target**: Multiclass classification — 3 wine cultivars
- **Features**: 13 chemical measurements (alcohol, malic acid, ash, etc.)

### Data Splits

| Split | Rows | Purpose |
|-------|------|---------|
| Train (60%) | 106 | Schema inference, model training |
| Eval (20%) | 36 | Anomaly detection, model evaluation |
| Serving (20%) | 36 | Inference simulation |

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

---

## Project Structure

```
Labs/MLMD_Labs/Lab1/
├── C2_W3_Lab_1_MLMetadata.ipynb   # Main notebook
├── README.md                       # This file
├── img/
│   └── mlmd_overview.png           # MLMD architecture diagram
├── data/                           # Generated at runtime
│   ├── train/data.csv
│   ├── eval/data.csv
│   └── serving/data.csv
├── model/                          # Generated at runtime
│   ├── model.pkl
│   ├── scaler.pkl
│   └── eval_metrics.json
├── metadata/                       # Generated at runtime
│   └── mlmd.sqlite
├── schema.json                     # Generated at runtime
└── anomalies.json                  # Generated at runtime
```

---

## Requirements

Only standard data science packages — no TensorFlow needed:

```bash
pip install scikit-learn pandas numpy notebook
```

---

## How to Run

1. Open `C2_W3_Lab_1_MLMetadata.ipynb` in Jupyter Notebook
2. Run all cells sequentially (Kernel → Restart & Run All)
3. The notebook will:
   - Load and split the Wine dataset from scikit-learn
   - Create a SQLite metadata store at `metadata/mlmd.sqlite`
   - Run all 4 pipeline stages with full metadata tracking
   - Demonstrate lineage queries tracing artifacts back through the pipeline

---

## Troubleshooting

**"No module named sklearn"** — Run `pip install scikit-learn`

**SQLite database locked** — Restart the kernel to release the connection

**Kernel not using correct environment** — Run `python -m ipykernel install --user --name=mlmd_env` and select that kernel

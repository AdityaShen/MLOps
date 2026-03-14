# Handwritten Digit Classification API with FastAPI

---

- **Video Explanation (Original Lab):** [FastAPI Lab](https://www.youtube.com/watch?v=KReburHqRIQ&list=PLcS4TrUUc53LeKBIyXAaERFKBJ3dvc9GZ&index=4)
- **Blog:** [FastAPI Lab-1](https://www.mlwithramin.com/blog/fastapi-lab1)

---

## Overview

This lab demonstrates how to deploy a machine learning model as a REST API using **FastAPI** and **Uvicorn**. A **K-Nearest Neighbors (KNN)** classifier is trained on the **Digits** dataset (1,797 samples of 8x8 handwritten digit images) and served through API endpoints for single and batch predictions.

### Technologies Used

- **FastAPI** — Modern Python web framework for building APIs
- **Uvicorn** — ASGI server to run the FastAPI application
- **Scikit-Learn** — ML library used to train the KNN classifier

---

## Modifications from the Original Lab

| Component | Original Lab | Modified Implementation |
|-----------|-------------|------------------------|
| Dataset | Iris (150 samples, 4 features) | Digits (1,797 samples, 64 features) |
| Model | Decision Tree | KNN (k=5, distance-weighted) |
| Preprocessing | None | StandardScaler |
| Prediction | Single predict endpoint | Single + Batch prediction endpoints |
| API | Basic endpoints | Additional model info + input validation |
| Schema | 4 named iris features | 64-element pixel array |

---

## Project Structure

```
fastapi_lab1/
├── model/
│   ├── digits_knn.pkl
│   └── digits_scaler.pkl
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── main.py
│   ├── predict.py
│   └── train.py
├── assets/
├── requirements.txt
└── README.md
```

| File | Description |
|------|-------------|
| `train.py` | Loads Digits dataset, trains KNN model, saves model + scaler |
| `predict.py` | Loads trained model and scaler, performs predictions |
| `data.py` | Pydantic request/response schemas |
| `main.py` | FastAPI application with all API endpoints |

---

## Setup

### 1. Create and Activate Virtual Environment

```bash
python3 -m venv fastapi_env
source fastapi_env/bin/activate  # Mac/Linux
# fastapi_env\Scripts\activate   # Windows
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Lab

### 1. Train the Model

```bash
cd src
python train.py
```

Expected output:
```
Model Accuracy: 0.9861
Model and scaler saved successfully.
```

### 2. Start the API Server

From the `src` directory:

```bash
uvicorn main:app --reload
```

The server starts at `http://127.0.0.1:8000`.

### 3. View API Documentation

Open `http://127.0.0.1:8000/docs` for the interactive Swagger UI.

---

## API Endpoints

### GET `/` — Health Check

```json
{ "status": "healthy", "message": "Digit Classification API" }
```

### POST `/predict` — Single Prediction

Send 64 pixel values (0.0–16.0) representing an 8x8 digit image.

**Request:**
```json
{
  "pixels": [0,0,5,13,9,1,0,0,0,0,13,15,10,15,5,0,0,3,15,2,0,11,8,0,0,4,12,0,0,8,8,0,0,5,8,0,0,9,8,0,0,4,11,0,1,12,7,0,0,2,14,5,10,12,0,0,0,0,6,13,10,0,0,0]
}
```

**Response:**
```json
{ "prediction": 0 }
```

### POST `/batch_predict` — Batch Prediction

**Request:**
```json
{
  "samples": [
    { "pixels": [0,0,5,13,9,1,0,0,...] },
    { "pixels": [0,0,0,12,13,5,0,0,...] }
  ]
}
```

**Response:**
```json
{ "predictions": [0, 1] }
```

### GET `/model_info` — Model Metadata

```json
{
  "model": "KNeighborsClassifier",
  "dataset": "Digits (Handwritten Digit Recognition)",
  "features": "64 pixel intensities from 8x8 images",
  "classes": "10 (digits 0-9)",
  "task": "Multiclass digit classification"
}
```

---

## Troubleshooting

**"Module not found" errors** — Make sure you're running `uvicorn` from inside the `src/` directory.

**"No such file" for model** — Run `python train.py` first to generate the `.pkl` files in `model/`.

**Port already in use** — Kill the existing process or use a different port: `uvicorn main:app --reload --port 8001`

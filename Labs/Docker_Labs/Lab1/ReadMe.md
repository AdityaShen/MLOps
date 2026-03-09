# Handwritten Digit Classification with KNN (Dockerized)

## Overview
this project trains a **KNN** classifier on the **Digits** dataset from scikit-learn. The dataset contains 1,797 samples of 8x8 pixel handwritten digit images (0–9), represented as 64 numerical features. The entire training and evaluation pipeline runs inside a Docker container

### What Changed from the Original

| | Original | Updated |
|---|---|---|
| **Dataset** | Iris (150 samples, 4 features, 3 classes) | Digits (1,797 samples, 64 features, 10 classes) |
| **Model** | Random Forest | KNN (k=5, distance-weighted) |
| **Preprocessing** | None | StandardScaler  |
| **Evaluation** | None | Accuracy, F1, classification report, confusion matrix |
| **Error Handling** | None | Try-catch with proper exit codes |
| **Output Files** | `iris_model.pkl` | `digits_knn.pkl`, `digits_scaler.pkl`, `model_metrics.json`, `sample_predictions.json` |

### Why KNN Needs Scaling
KNN classifies points based on the distance to their neighbors. If features have different scales, the larger-scale feature dominates the distance calculation. `StandardScaler` normalizes all features to zero mean and unit variance so each feature contributes equally

---

## Project Structure
```
Lab1/
├── Dockerfile
├── README.md
└── src/
    ├── main.py
    └── requirements.txt
```

---

## Prerequisites
- [Docker](https://docs.docker.com/get-docker/) installed and running on your machine.

---

## Step-by-Step Instructions

### 1. Build the Docker Image
```bash
cd Lab1
docker build -t lab1:v1 .
```

### 2. Run the Container
```bash
docker run --name digits-knn lab1:v1
```

### 3. Extract Results
```bash
docker cp digits-knn:/app/digits_knn.pkl ./
docker cp digits-knn:/app/digits_scaler.pkl ./
docker cp digits-knn:/app/model_metrics.json ./
docker cp digits-knn:/app/sample_predictions.json ./
```

### 4. View Results
```bash
cat model_metrics.json
cat sample_predictions.json
```

### 5. Save Image to Tar
```bash
docker save lab1:v1 > my_image.tar
```

### 6. Load Image from Tar (on another machine)
```bash
docker load < my_image.tar
```

---



---

## Results

<img width="774" height="677" alt="image" src="https://github.com/user-attachments/assets/cdd88c70-bb47-4406-96c0-bb7825139505" />

---

## Useful Docker Commands

| Command | Description |
|---|---|
| `docker images` | List all local images |
| `docker ps -a` | List all containers (running and stopped) |
| `docker logs digits-knn` | View container output |
| `docker rmi lab1:v1` | Remove the image |
| `docker system prune` | Clean up unused containers/images |


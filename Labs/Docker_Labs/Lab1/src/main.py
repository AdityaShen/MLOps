# Import necessary libraries
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import joblib
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    try:
        # Load the Digits dataset (8x8 pixel handwritten digits, 0-9)
        logger.info("Loading Digits dataset...")
        data = load_digits()
        X, y = data.data, data.target

        logger.info(f"Dataset shape: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"Classes: {list(data.target_names)}")

        # Validate data
        if X.shape[0] == 0:
            raise ValueError("Dataset is empty")

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        logger.info(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

        # Scale features (critical for KNN since it uses distance metrics)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Train a KNN classifier
        logger.info("Training KNN classifier (k=5)...")
        model = KNeighborsClassifier(n_neighbors=5, weights='distance', metric='minkowski')
        model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        logger.info(f"Test Accuracy: {acc:.4f}")
        logger.info(f"F1 Score:      {f1:.4f}")

        # Print classification report
        print("\n" + "=" * 50)
        print("Classification Report")
        print("=" * 50)
        print(classification_report(y_test, y_pred, target_names=[str(d) for d in data.target_names]))

        # Print confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:")
        print(cm)

        # Save metrics to JSON
        metrics = {
            "model": "KNeighborsClassifier",
            "dataset": "Digits (Handwritten Digit Recognition)",
            "n_neighbors": 5,
            "accuracy": round(acc, 4),
            "f1_score": round(f1, 4),
            "confusion_matrix": cm.tolist()
        }
        with open("model_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Metrics saved to model_metrics.json")

        # Save sample predictions
        sample_preds = []
        for i in range(min(10, len(y_test))):
            sample_preds.append({
                "actual": int(y_test[i]),
                "predicted": int(y_pred[i]),
                "correct": bool(y_test[i] == y_pred[i])
            })
        with open("sample_predictions.json", "w") as f:
            json.dump(sample_preds, f, indent=2)
        logger.info("Sample predictions saved to sample_predictions.json")

        # Save the model and scaler
        joblib.dump(model, 'digits_knn.pkl')
        joblib.dump(scaler, 'digits_scaler.pkl')

        print(f"\nTest Accuracy: {acc:.4f}")
        print("The model training was successful")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        exit(1)

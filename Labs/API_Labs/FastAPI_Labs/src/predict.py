import joblib
import numpy as np

# Load model and scaler once at module level
model = joblib.load("../model/digits_knn.pkl")
scaler = joblib.load("../model/digits_scaler.pkl")


def predict_digit(data):
    """
    Predict the digit class (0-9) for a single sample.
    Args:
        data: DigitData object with a 'pixels' list of 64 floats.
    Returns:
        int: Predicted digit (0-9).
    """
    features = np.array(data.pixels).reshape(1, -1)
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    return int(prediction[0])

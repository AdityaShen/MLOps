from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib
import os

# Load dataset
digits = load_digits()
X, y = digits.data, digits.target

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features (critical for KNN)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = KNeighborsClassifier(n_neighbors=5, weights='distance', metric='minkowski')
model.fit(X_train, y_train)

# Evaluate
pred = model.predict(X_test)
accuracy = accuracy_score(y_test, pred)
print(f"Model Accuracy: {accuracy:.4f}")

# Save model and scaler
os.makedirs("../model", exist_ok=True)
joblib.dump(model, "../model/digits_knn.pkl")
joblib.dump(scaler, "../model/digits_scaler.pkl")
print("Model and scaler saved successfully.")

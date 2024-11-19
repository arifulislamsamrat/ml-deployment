import requests
import numpy as np
from sklearn.datasets import make_classification

# Generate some test data with correct parameters
X, _ = make_classification(
    n_samples=2,           # Number of samples
    n_features=2,          # Total features
    n_classes=2,           # Number of classes
    n_informative=2,       # Number of informative features
    n_redundant=0,         # No redundant features
    n_repeated=0,          # No repeated features
    random_state=42        # For reproducibility
)

# Format data for API
data = {
    'features': [
        {'feature1': float(X[0][0]), 'feature2': float(X[0][1])},
        {'feature1': float(X[1][0]), 'feature2': float(X[1][1])}
    ]
}

print("Sending data:", data)

# Make prediction request
try:
    response = requests.post('http://localhost:8000/api/v1/predict', json=data)
    print("Status Code:", response.status_code)
    print("Response:", response.json())
except requests.exceptions.ConnectionError:
    print("Error: Could not connect to the API. Make sure the Flask server is running.")
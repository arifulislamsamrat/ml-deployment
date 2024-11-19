import requests
import numpy as np
from sklearn.datasets import make_classification

# Generate test data
X, _ = make_classification(
    n_samples=2,
    n_features=2,
    n_classes=2,
    n_informative=2,
    n_redundant=0,
    n_repeated=0,
    random_state=42
)

# Format data for API
data = {
    'features': [
        {
            'feature1': float(X[0][0]),
            'feature2': float(X[0][1])
        },
        {
            'feature1': float(X[1][0]),
            'feature2': float(X[1][1])
        }
    ]
}

print("Sending data:", data)

try:
    # First test health endpoint
    health_response = requests.get('http://localhost:8000/api/v1/health')
    print("\nHealth check:")
    print("Status:", health_response.status_code)
    print("Response:", health_response.json())

    # Then test prediction endpoint
    pred_response = requests.post('http://localhost:8000/api/v1/predict', json=data)
    print("\nPrediction request:")
    print("Status:", pred_response.status_code)
    print("Response:", pred_response.json())
except requests.exceptions.ConnectionError:
    print("Error: Could not connect to the API. Make sure the Flask server is running.")
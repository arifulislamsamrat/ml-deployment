import requests

# Test data
data = {
    'features': [
        {'feature1': 0.5, 'feature2': -0.5},
        {'feature1': -0.2, 'feature2': 0.3}
    ]
}

# Make prediction request
response = requests.post('http://localhost:8000/api/v1/predict', json=data)
print("Status:", response.status_code)
print("Predictions:", response.json())
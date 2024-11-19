import mlflow
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Set tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# Generate sample data
X, y = make_classification(
    n_samples=100,
    n_features=2,
    n_classes=2,
    n_informative=2,
    n_redundant=0,
    n_repeated=0,
    random_state=42
)

# Start MLflow run
with mlflow.start_run() as run:
    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
    
    # Log metrics
    mlflow.log_metric("accuracy", model.score(X, y))
    
    # Register model
    mlflow.register_model(f"runs:/{run.info.run_id}/model", "sklearn_model")
    
    print(f"Run ID: {run.info.run_id}")
    print("Model registered successfully")
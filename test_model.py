import mlflow
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Generate sample data with corrected parameters
X, y = make_classification(
    n_samples=100,          # Number of samples
    n_features=2,           # Number of features
    n_classes=2,            # Number of classes
    n_informative=2,        # Number of informative features (must be <= n_features)
    n_redundant=0,          # No redundant features
    n_repeated=0,           # No repeated features
    random_state=42         # For reproducibility
)

X_train = X[:80]
X_test = X[80:]
y_train = y[:80]
y_test = y[80:]

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# Start MLflow run
with mlflow.start_run():
    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Log metrics
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    mlflow.log_metrics({
        "train_score": train_score,
        "test_score": test_score
    })
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
    
    # Register model
    model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
    mv = mlflow.register_model(model_uri, "sklearn_model")
    print(f"Model registered with version: {mv.version}")
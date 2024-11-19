from mlflow.tracking import MlflowClient
import mlflow

# Initialize client
client = MlflowClient("http://localhost:5000")

# List all registered models to verify
models = client.search_registered_models()  # Changed from list_registered_models()
for m in models:
    print(f"Model: {m.name}")

try:
    # Transition the model to production
    client.transition_model_version_stage(
        name="sklearn_model",
        version=1,  # Use the version number from the registration step
        stage="Production"
    )
    print("Model transitioned to production successfully")
except Exception as e:
    print(f"Error transitioning model: {e}")

# Verify the model status
try:
    model_version = client.get_model_version("sklearn_model", 1)
    print(f"Model version: {model_version.version}")
    print(f"Current stage: {model_version.current_stage}")
except Exception as e:
    print(f"Error getting model version: {e}")
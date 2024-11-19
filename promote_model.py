from mlflow.tracking import MlflowClient

# Initialize client
client = MlflowClient("http://localhost:5000")

# List all registered models to verify
models = client.list_registered_models()
for m in models:
    print(f"Model: {m.name}")

# Transition the model to production
client.transition_model_version_stage(
    name="sklearn_model",
    version=1,  # Use the version number from the registration step
    stage="Production"
)
print("Model transitioned to production")
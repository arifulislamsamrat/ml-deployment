from mlflow.tracking import MlflowClient

client = MlflowClient("http://localhost:5000")
client.transition_model_version_stage(
    name="sklearn_model",
    version=1,
    stage="Production"
)
import mlflow
from mlflow.tracking import MlflowClient

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")
client = MlflowClient()

def promote_to_production(model_name="sklearn_model"):
    try:
        # Get latest version
        versions = client.search_model_versions(f"name='{model_name}'")
        if not versions:
            raise Exception(f"No versions found for model {model_name}")
            
        latest_version = sorted(versions, key=lambda x: x.version, reverse=True)[0]
        
        # Transition to production
        client.transition_model_version_stage(
            name=model_name,
            version=latest_version.version,
            stage="Production"
        )
        print(f"Model {model_name} version {latest_version.version} transitioned to Production")
    except Exception as e:
        print(f"Error promoting model: {e}")
        raise

if __name__ == "__main__":
    promote_to_production()
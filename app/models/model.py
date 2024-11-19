import mlflow
from mlflow.tracking import MlflowClient

class MLModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.client = MlflowClient()
        
    def get_latest_version(self):
        latest_version = self.client.get_latest_versions(self.model_name, stages=["Production"])
        if not latest_version:
            raise ValueError(f"No production version found for model {self.model_name}")
        return latest_version[0]
    
    def predict(self, data):
        model_version = self.get_latest_version()
        model = mlflow.pyfunc.load_model(model_version.source)
        return model.predict(data)
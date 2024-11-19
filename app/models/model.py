import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd

class MLModel:
    def __init__(self, model_name):
        self.model_name = model_name
        mlflow.set_tracking_uri("http://mlflow-server:5000")
        self.client = MlflowClient()

    def get_latest_version(self):
        try:
            versions = self.client.search_model_versions(f"name='{self.model_name}'")
            production_versions = [mv for mv in versions if mv.current_stage == 'Production']
            if not production_versions:
                raise ValueError(f"No production version found for model {self.model_name}")
            return production_versions[0]
        except Exception as e:
            print(f"Error getting model version: {e}")
            raise

    def predict(self, features_df):
        try:
            version = self.get_latest_version()
            print(f"Loading model version: {version.version}")
            
            model = mlflow.sklearn.load_model(
                model_uri=f"models:/{self.model_name}/Production"
            )
            
            return model.predict(features_df)
        except Exception as e:
            print(f"Prediction error: {e}")
            raise
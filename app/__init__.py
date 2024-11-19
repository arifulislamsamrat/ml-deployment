from flask import Flask
from app.api.endpoints import api
import mlflow

def create_app():
    app = Flask(__name__)
    app.register_blueprint(api, url_prefix='/api/v1')
    return app
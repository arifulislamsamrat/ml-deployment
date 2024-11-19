from flask import Blueprint, request, jsonify
from app.models.model import MLModel
from app.config import Config
import pandas as pd

api = Blueprint('api', __name__)
model = MLModel(Config.MODEL_NAME)

@api.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame(data['features'])
        predictions = model.predict(df)
        return jsonify({'predictions': predictions.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 400
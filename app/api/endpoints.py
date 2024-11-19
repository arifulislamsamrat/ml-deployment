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
        if not data or 'features' not in data:
            return jsonify({'error': 'No features provided'}), 400

        # Convert features to DataFrame format
        features_df = pd.DataFrame([
            {'feature1': feature['feature1'], 'feature2': feature['feature2']}
            for feature in data['features']
        ])

        predictions = model.predict(features_df)
        return jsonify({'predictions': predictions})
    except Exception as e:
        return jsonify({'error': str(e)}), 400
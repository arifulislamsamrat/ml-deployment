from flask import Blueprint, request, jsonify
from app.models.model import MLModel
import pandas as pd

api_bp = Blueprint('api', __name__)
model = MLModel("sklearn_model")

@api_bp.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"}), 200

@api_bp.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Validate input
        if not data or 'features' not in data:
            return jsonify({"error": "Missing features in request"}), 400
            
        # Convert features to DataFrame
        features_df = pd.DataFrame(data['features'])
        
        # Make prediction
        predictions = model.predict(features_df)
        
        return jsonify({"predictions": predictions.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 400
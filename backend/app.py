from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from joblib import load
from get_planets import get_planet_features
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'rag', 'app'))
from chatbot import ask

app = Flask(__name__)
CORS(app)  # Enable CORS for React app

# Load the model
model_path = os.path.join(os.path.dirname(__file__), 'weather_planet_model.pkl')
model = load(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    date_str = data.get('date')
    if not date_str:
        return jsonify({'error': 'Date is required'}), 400

    try:
        # Fetch planet features
        features = get_planet_features(date_str)

        # Prediction
        prediction = model.predict(features)[0]

        # Convert features to dict for JSON response
        features_dict = features.iloc[0].to_dict()

        return jsonify({
            'temperature': round(prediction, 2),
            'date': date_str,
            'features': features_dict
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    query = data.get('query')
    if not query:
        return jsonify({'error': 'Query is required'}), 400

    try:
        result = ask(query)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)

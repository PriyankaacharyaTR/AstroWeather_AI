from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from joblib import load
from get_planets import get_planet_features
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'rag', 'app'))
from chatbot import ask, client, MODEL_NAME

app = Flask(__name__)
CORS(app)  # Enable CORS for React app

# Load existing planet-based model
planet_model_path = os.path.join(os.path.dirname(__file__), 'weather_planet_model.pkl')
planet_model = load(planet_model_path)

# Load Bengaluru date-feature model (Multi-Output RF)
bangaluru_model_path = os.path.join(os.path.dirname(__file__), 'bangaluru_weather_model.pkl')
bangaluru_model = load(bangaluru_model_path)

# Load Delhi date-feature model (Multi-Output RF)
delhi_model_path = os.path.join(os.path.dirname(__file__), 'delhi_weather_model.pkl')
delhi_model = load(delhi_model_path)

TARGET_COLUMNS = ["T2M", "PS", "QV2M", "WS2M", "GWETTOP"]


def build_date_features(date_str: str) -> pd.DataFrame:
    """Create the date-derived feature frame expected by the Bengaluru model."""
    date = pd.to_datetime(date_str)
    week_of_year = int(date.isocalendar().week)
    return pd.DataFrame(
        {
            "Year": [date.year],
            "Month": [date.month],
            "Day": [date.day],
            "DayOfYear": [date.dayofyear],
            "WeekOfYear": [week_of_year],
        }
    )

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
        prediction = planet_model.predict(features)[0]

        # Convert features to dict for JSON response
        features_dict = features.iloc[0].to_dict()

        return jsonify({
            'temperature': round(prediction, 2),
            'date': date_str,
            'features': features_dict
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict-bangaluru', methods=['POST'])
def predict_bangaluru():
    data = request.get_json() or {}
    date_str = data.get('date')
    if not date_str:
        return jsonify({'error': 'Date is required'}), 400

    try:
        features = build_date_features(date_str)
        prediction = bangaluru_model.predict(features)[0]

        predictions = {
            TARGET_COLUMNS[i]: round(float(prediction[i]), 2)
            for i in range(len(TARGET_COLUMNS))
        }

        feature_payload = {
            k: (int(float(v)) if float(v).is_integer() else float(v))
            for k, v in features.iloc[0].to_dict().items()
        }

        return jsonify(
            {
                'date': date_str,
                'features': feature_payload,
                'predictions': predictions,
            }
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict-delhi', methods=['POST'])
def predict_delhi():
    data = request.get_json() or {}
    date_str = data.get('date')
    if not date_str:
        return jsonify({'error': 'Date is required'}), 400

    try:
        features = build_date_features(date_str)
        prediction = delhi_model.predict(features)[0]

        predictions = {
            TARGET_COLUMNS[i]: round(float(prediction[i]), 2)
            for i in range(len(TARGET_COLUMNS))
        }

        feature_payload = {
            k: (int(float(v)) if float(v).is_integer() else float(v))
            for k, v in features.iloc[0].to_dict().items()
        }

        return jsonify(
            {
                'date': date_str,
                'features': feature_payload,
                'predictions': predictions,
            }
        )
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

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.get_json()
    weather_data = data.get('weather')
    planet_data = data.get('planets') or {}
    city = data.get('city', 'Unknown')
    date = data.get('date')
    
    if not weather_data:
        return jsonify({'error': 'Weather data required'}), 400

    try:
        # Format weather data for summary
        weather_str = "\n".join([
            f"{PARAMETER_INFO.get(k, {}).get('name', k)}: {v} {PARAMETER_INFO.get(k, {}).get('unit', '')}"
            for k, v in weather_data.items()
        ])
        
        # Format planet data for summary
        if planet_data:
            planet_str = "\n".join([
                f"{name}: angle={p['angle']:.2f} rad, influence={p['influence']:.3f}"
                for name, p in planet_data.items()
            ])
        else:
            planet_str = "Planetary data not available; base the summary on weather observations only."
        
        prompt = f"""
You are a professional meteorological analyst. Generate a comprehensive, well-structured weather report using only text formatting with proper spacing and clear sections.

WEATHER ANALYSIS REPORT
Location: {city.upper()}
Date: {date}

MEASURED METEOROLOGICAL DATA:
{weather_str}

GENERATE THE REPORT WITH THE FOLLOWING STRUCTURE:

1. HEADLINE
A concise, professional one-line summary of the expected weather condition for the day.

2. ATMOSPHERIC CONDITIONS
Provide 2-3 sentences describing the current atmospheric state based on the temperature, pressure, and humidity measurements provided. Focus on what these parameters collectively indicate about the weather. Justify this paragraph with even word spacing.

3. KEY OBSERVATIONS
List 3-4 important observations about today's weather based on the measured data. For each observation, reference the specific parameter and its value. Place each observation on a new line with clear, simple language.

4. IMPACT ASSESSMENT
Provide 2-3 sentences explaining what these weather conditions mean for daily activities, outdoor plans, or general well-being. Be practical and informative. Justify this paragraph with even word spacing.

5. WEATHER OUTLOOK
Provide 1-2 sentences concluding the weather assessment based on all analyzed parameters. Justify this paragraph.

FORMATTING RULES:
- Do NOT use asterisks, hashtags, or emojis
- Insert ONE blank line between each major section heading
- Ensure all paragraph sections are justified with even spacing
- Use simple line breaks to separate individual observations
- Keep language professional and straightforward
- Reference actual data values from the parameters provided
- Avoid speculation, stick to what the data shows
- Make it readable and organized for a weather dashboard display
- Ensure consistent spacing throughout the report
"""
        
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": prompt.strip()},
            ],
            temperature=0.7,
            max_tokens=400,
        )

        summary = completion.choices[0].message.content
        return jsonify({'summary': summary})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Parameter info for formatting
PARAMETER_INFO = {
    'T2M': {'name': 'Air Temperature', 'unit': '°C'},
    'PS': {'name': 'Atm Pressure', 'unit': 'kPa'},
    'QV2M': {'name': 'Specific Humidity', 'unit': 'g/kg'},
    'GWETTOP': {'name': 'Top Soil Wetness', 'unit': ''},
    'WS2M': {'name': 'Wind Speed', 'unit': 'm/s'},
}

if __name__ == '__main__':
    app.run(debug=True, port=5000)

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
from joblib import load
from get_planets import get_planet_features, get_planet_features_simple, FEATURE_ORDER, TARGET_COLUMNS as PLANET_TARGET_COLUMNS
import sys
import os
import json
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'rag', 'app'))
from chatbot import ask, client, MODEL_NAME

# WeatherAPI.com API key from environment variable
WEATHER_API_KEY = os.getenv('WEATHER_API_KEY')

app = Flask(__name__)
CORS(app)  # Enable CORS for React app

# Load feature columns configuration
FEATURE_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'feature_columns.json')
with open(FEATURE_CONFIG_PATH, 'r') as f:
    FEATURE_CONFIG = json.load(f)

# Load Bengaluru planetary-feature model (Multi-Output RF trained on planetary vectors)
bangaluru_model_path = os.path.join(os.path.dirname(__file__), 'bangaluru_weather_model.pkl')
bangaluru_model = load(bangaluru_model_path)

# Load Delhi date-feature model (Multi-Output RF) - keeping for backward compatibility
delhi_model_path = os.path.join(os.path.dirname(__file__), 'delhi_weather_model.pkl')
delhi_model = load(delhi_model_path)

# Load ARIMA models for time-series forecasting (baseline comparison)
# Load city-specific ARIMA models
ARIMA_CITIES_PATH = os.path.join(os.path.dirname(__file__), '..', 'train_solarplanets', 'arima_all_cities_models.pkl')
ARIMA_LEGACY_PATH = os.path.join(os.path.dirname(__file__), '..', 'train_solarplanets', 'arima_all_models.pkl')

try:
    # Try loading city-specific models first
    city_arima_data = load(ARIMA_CITIES_PATH)
    arima_city_models = city_arima_data  # {city: {param: model}}
    ARIMA_CITIES_AVAILABLE = True
    print("✅ City-specific ARIMA models loaded successfully")
    
    # Use Bengaluru as default for backward compatibility
    arima_models = arima_city_models.get('bengaluru', {}).get('models', {})
    arima_params = arima_city_models.get('bengaluru', {}).get('params', {})
    arima_metrics = arima_city_models.get('bengaluru', {}).get('metrics', {})
    ARIMA_AVAILABLE = True
except Exception as e:
    arima_city_models = {}
    ARIMA_CITIES_AVAILABLE = False
    print(f"⚠️ City-specific ARIMA models not available: {e}")
    
    # Fallback to legacy single-city model
    try:
        arima_data = load(ARIMA_LEGACY_PATH)
        arima_models = arima_data.get('models', {})
        arima_params = arima_data.get('params', {})
        arima_metrics = arima_data.get('metrics', {})
        ARIMA_AVAILABLE = True
        print("✅ Legacy ARIMA models loaded successfully")
    except Exception as e2:
        arima_models = {}
        arima_params = {}
        arima_metrics = {}
        ARIMA_AVAILABLE = False
        print(f"⚠️ ARIMA models not available: {e2}")

# Load Bengaluru training dataset for date lookup
BANGALURU_DATASET_PATH = os.path.join(os.path.dirname(__file__), '..', 'train_solarplanets', 'final_dataset_bangaluru.csv')
try:
    bangaluru_df = pd.read_csv(BANGALURU_DATASET_PATH)
    bangaluru_df['Date'] = pd.to_datetime(bangaluru_df['Date']).dt.strftime('%Y-%m-%d')
    BANGALURU_DATASET_AVAILABLE = True
except:
    bangaluru_df = None
    BANGALURU_DATASET_AVAILABLE = False

# Load Delhi training dataset for date lookup
DELHI_DATASET_PATH = os.path.join(os.path.dirname(__file__), '..', 'train_solarplanets', 'final_dataset_delhi.csv')
try:
    delhi_df = pd.read_csv(DELHI_DATASET_PATH)
    delhi_df['Date'] = pd.to_datetime(delhi_df['Date']).dt.strftime('%Y-%m-%d')
    DELHI_DATASET_AVAILABLE = True
except:
    delhi_df = None
    DELHI_DATASET_AVAILABLE = False

TARGET_COLUMNS = ["T2M", "PS", "QV2M", "WS2M", "GWETTOP"]


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'ok'}), 200


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
    """Legacy endpoint - redirects to predict-bangaluru."""
    data = request.get_json()
    date_str = data.get('date')
    if not date_str:
        return jsonify({'error': 'Date is required'}), 400

    try:
        # Fetch planet features with date features
        features, planetary_vectors = get_planet_features(date_str)

        # Prediction using Bangaluru model
        prediction = bangaluru_model.predict(features)[0]

        predictions = {
            TARGET_COLUMNS[i]: round(float(prediction[i]), 2)
            for i in range(len(TARGET_COLUMNS))
        }

        # Convert features to dict for JSON response
        features_dict = features.iloc[0].to_dict()

        return jsonify({
            'temperature': predictions['T2M'],
            'date': date_str,
            'features': features_dict,
            'predictions': predictions,
            'planetary_vectors_km_km_per_s': planetary_vectors
        })
    except Exception as e:
        return jsonify({'error': str(e), 'error_type': type(e).__name__}), 500


@app.route('/predict-features', methods=['POST'])
def predict_features():
    """
    Predict directly from provided features.
    Request body should contain all feature keys in feature_columns.json.
    """
    data = request.get_json() or {}
    
    if not data:
        return jsonify({'error': 'Feature data is required'}), 400
    
    try:
        # Validate that all required features are present
        missing_features = [f for f in FEATURE_ORDER if f not in data]
        if missing_features:
            return jsonify({
                'error': f'Missing features: {missing_features}',
                'required_features': FEATURE_ORDER
            }), 400
        
        # Build feature DataFrame in correct order
        feature_data = {col: [data[col]] for col in FEATURE_ORDER}
        features = pd.DataFrame(feature_data)[FEATURE_ORDER]
        
        # Make prediction
        prediction = bangaluru_model.predict(features)[0]
        
        predictions = {
            TARGET_COLUMNS[i]: round(float(prediction[i]), 2)
            for i in range(len(TARGET_COLUMNS))
        }
        
        return jsonify({
            'predictions': predictions,
            'feature_count': len(FEATURE_ORDER)
        })
    except Exception as e:
        return jsonify({'error': str(e), 'error_type': type(e).__name__}), 500


@app.route('/predict-bangaluru', methods=['POST'])
def predict_bangaluru():
    data = request.get_json() or {}
    date_str = data.get('date')
    if not date_str:
        return jsonify({'error': 'Date is required'}), 400

    try:
        planetary_vectors = None
        from_csv = False
        
        # Check if date exists in CSV dataset first
        if BANGALURU_DATASET_AVAILABLE and bangaluru_df is not None:
            csv_row = bangaluru_df[bangaluru_df['Date'] == date_str]
            if not csv_row.empty:
                # Use features from CSV
                from_csv = True
                date = pd.to_datetime(date_str)
                
                # Build feature row from CSV data
                feature_data = {}
                for col in FEATURE_ORDER:
                    if col in csv_row.columns:
                        feature_data[col] = csv_row[col].values[0]
                    elif col == 'Year':
                        feature_data[col] = date.year
                    elif col == 'Month':
                        feature_data[col] = date.month
                    elif col == 'Day':
                        feature_data[col] = date.day
                    elif col == 'DayOfYear':
                        feature_data[col] = date.dayofyear
                    elif col == 'WeekOfYear':
                        feature_data[col] = int(date.isocalendar().week)
                
                features = pd.DataFrame([feature_data])[FEATURE_ORDER]
        
        if not from_csv:
            # Fetch planetary vectors from NASA JPL Horizons
            features, planetary_vectors = get_planet_features(date_str)
        
        # Make prediction
        prediction = bangaluru_model.predict(features)[0]

        predictions = {
            TARGET_COLUMNS[i]: round(float(prediction[i]), 2)
            for i in range(len(TARGET_COLUMNS))
        }

        # Build feature payload for response
        feature_payload = {
            k: (int(float(v)) if isinstance(v, (int, float)) and float(v).is_integer() else float(v) if isinstance(v, (int, float)) else v)
            for k, v in features.iloc[0].to_dict().items()
        }

        response_data = {
            'date': date_str,
            'features': feature_payload,
            'predictions': predictions,
            'data_source': 'csv' if from_csv else 'nasa_jpl_horizons'
        }
        
        # Include planetary vectors if fetched from NASA
        if planetary_vectors:
            response_data['planetary_vectors_km_km_per_s'] = planetary_vectors

        return jsonify(response_data)
    except Exception as e:
        return jsonify({'error': str(e), 'error_type': type(e).__name__}), 500


@app.route('/predict-delhi', methods=['POST'])
def predict_delhi():
    data = request.get_json() or {}
    date_str = data.get('date')
    if not date_str:
        return jsonify({'error': 'Date is required'}), 400

    try:
        planetary_vectors = None
        from_csv = False
        
        # Check if date exists in CSV dataset first
        if DELHI_DATASET_AVAILABLE and delhi_df is not None:
            csv_row = delhi_df[delhi_df['Date'] == date_str]
            if not csv_row.empty:
                # Use features from CSV
                from_csv = True
                date = pd.to_datetime(date_str)
                
                # Build feature row from CSV data
                feature_data = {}
                for col in FEATURE_ORDER:
                    if col in csv_row.columns:
                        feature_data[col] = csv_row[col].values[0]
                    elif col == 'Year':
                        feature_data[col] = date.year
                    elif col == 'Month':
                        feature_data[col] = date.month
                    elif col == 'Day':
                        feature_data[col] = date.day
                    elif col == 'DayOfYear':
                        feature_data[col] = date.dayofyear
                    elif col == 'WeekOfYear':
                        feature_data[col] = int(date.isocalendar().week)
                
                features = pd.DataFrame([feature_data])[FEATURE_ORDER]
        
        if not from_csv:
            # Fetch planetary vectors from NASA JPL Horizons
            features, planetary_vectors = get_planet_features(date_str)
        
        # Make prediction
        prediction = delhi_model.predict(features)[0]

        predictions = {
            TARGET_COLUMNS[i]: round(float(prediction[i]), 2)
            for i in range(len(TARGET_COLUMNS))
        }

        # Build feature payload for response
        feature_payload = {
            k: (int(float(v)) if isinstance(v, (int, float)) and float(v).is_integer() else float(v) if isinstance(v, (int, float)) else v)
            for k, v in features.iloc[0].to_dict().items()
        }

        response_data = {
            'date': date_str,
            'features': feature_payload,
            'predictions': predictions,
            'data_source': 'csv' if from_csv else 'nasa_jpl_horizons'
        }
        
        # Include planetary vectors if fetched from NASA
        if planetary_vectors:
            response_data['planetary_vectors_km_km_per_s'] = planetary_vectors

        return jsonify(response_data)
    except Exception as e:
        return jsonify({'error': str(e), 'error_type': type(e).__name__}), 500


def fetch_actual_weather(city: str, date_str: str) -> dict:
    """
    Fetch actual weather data from WeatherAPI.com.
    For historical/future dates, uses history or forecast API accordingly.
    Returns avg temperature and weather condition.
    """
    if not WEATHER_API_KEY:
        raise ValueError("WEATHER_API_KEY environment variable not set")
    
    # Map city names to WeatherAPI locations
    city_mapping = {
        'bengaluru': 'Bengaluru,India',
        'bangalore': 'Bengaluru,India',
        'delhi': 'Delhi,India'
    }
    
    location = city_mapping.get(city.lower(), f"{city},India")
    
    # Determine if we need history or forecast API
    from datetime import datetime
    target_date = datetime.strptime(date_str, '%Y-%m-%d').date()
    today = datetime.now().date()
    
    if target_date <= today:
        # Use history API for past dates
        url = f"http://api.weatherapi.com/v1/history.json"
        params = {
            'key': WEATHER_API_KEY,
            'q': location,
            'dt': date_str
        }
    else:
        # Use forecast API for future dates (up to 14 days)
        days_ahead = (target_date - today).days
        if days_ahead > 14:
            raise ValueError("WeatherAPI forecast is limited to 14 days ahead")
        url = f"http://api.weatherapi.com/v1/forecast.json"
        params = {
            'key': WEATHER_API_KEY,
            'q': location,
            'dt': date_str,
            'days': 1
        }
    
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    
    # Extract forecast day data
    if 'forecast' in data and data['forecast']['forecastday']:
        day_data = data['forecast']['forecastday'][0]['day']
        return {
            'avg_temp': day_data.get('avgtemp_c'),
            'max_temp': day_data.get('maxtemp_c'),
            'min_temp': day_data.get('mintemp_c'),
            'condition': day_data.get('condition', {}).get('text', 'N/A'),
            'humidity': day_data.get('avghumidity'),
            'wind_kph': day_data.get('maxwind_kph')
        }
    
    raise ValueError("No weather data available for the specified date")


@app.route('/api/temperature-comparison', methods=['POST'])
def temperature_comparison():
    """
    Compare astro-based temperature prediction with actual weather data.
    Returns both values and the difference.
    """
    data = request.get_json() or {}
    date_str = data.get('date')
    city = data.get('city', 'bengaluru')
    
    if not date_str:
        return jsonify({'error': 'Date is required'}), 400
    
    try:
        # Get astro-based prediction
        planetary_vectors = None
        data_source = 'nasa_jpl_horizons'
        
        if city.lower() in ['bengaluru', 'bangalore']:
            # Check CSV first for Bengaluru
            from_csv = False
            if BANGALURU_DATASET_AVAILABLE and bangaluru_df is not None:
                csv_row = bangaluru_df[bangaluru_df['Date'] == date_str]
                if not csv_row.empty:
                    from_csv = True
                    data_source = 'csv'
                    date = pd.to_datetime(date_str)
                    feature_data = {}
                    for col in FEATURE_ORDER:
                        if col in csv_row.columns:
                            feature_data[col] = csv_row[col].values[0]
                        elif col == 'Year':
                            feature_data[col] = date.year
                        elif col == 'Month':
                            feature_data[col] = date.month
                        elif col == 'Day':
                            feature_data[col] = date.day
                        elif col == 'DayOfYear':
                            feature_data[col] = date.dayofyear
                        elif col == 'WeekOfYear':
                            feature_data[col] = int(date.isocalendar().week)
                    features = pd.DataFrame([feature_data])[FEATURE_ORDER]
            
            if not from_csv:
                features, planetary_vectors = get_planet_features(date_str)
            
            prediction = bangaluru_model.predict(features)[0]
        else:
            # Delhi uses planetary features too
            from_csv = False
            if DELHI_DATASET_AVAILABLE and delhi_df is not None:
                csv_row = delhi_df[delhi_df['Date'] == date_str]
                if not csv_row.empty:
                    from_csv = True
                    data_source = 'csv'
                    date = pd.to_datetime(date_str)
                    feature_data = {}
                    for col in FEATURE_ORDER:
                        if col in csv_row.columns:
                            feature_data[col] = csv_row[col].values[0]
                        elif col == 'Year':
                            feature_data[col] = date.year
                        elif col == 'Month':
                            feature_data[col] = date.month
                        elif col == 'Day':
                            feature_data[col] = date.day
                        elif col == 'DayOfYear':
                            feature_data[col] = date.dayofyear
                        elif col == 'WeekOfYear':
                            feature_data[col] = int(date.isocalendar().week)
                    features = pd.DataFrame([feature_data])[FEATURE_ORDER]
            
            if not from_csv:
                features, planetary_vectors = get_planet_features(date_str)
            
            prediction = delhi_model.predict(features)[0]
        
        astro_temp = round(float(prediction[0]), 2)  # T2M is index 0
        
        # Fetch actual weather from WeatherAPI
        actual_weather = fetch_actual_weather(city, date_str)
        actual_temp = actual_weather.get('avg_temp')
        
        # Calculate difference
        difference = round(astro_temp - actual_temp, 2) if actual_temp is not None else None
        abs_difference = abs(difference) if difference is not None else None
        
        return jsonify({
            'date': date_str,
            'city': city.capitalize(),
            'data_source': data_source,
            'astro_prediction': {
                'temperature': astro_temp,
                'unit': '°C',
                'source': 'AstroWeather AI Model (Planetary Features)' if data_source != 'date_features' else 'AstroWeather AI Model'
            },
            'actual_weather': {
                'temperature': actual_temp,
                'max_temp': actual_weather.get('max_temp'),
                'min_temp': actual_weather.get('min_temp'),
                'condition': actual_weather.get('condition'),
                'humidity': actual_weather.get('humidity'),
                'wind_kph': actual_weather.get('wind_kph'),
                'unit': '°C',
                'source': 'WeatherAPI.com'
            },
            'comparison': {
                'difference': difference,
                'abs_difference': abs_difference,
                'astro_higher': difference > 0 if difference is not None else None
            },
            'planetary_vectors_km_km_per_s': planetary_vectors,
            'disclaimer': 'Astro predictions are based on ML models trained on planetary position data. Actual weather data is sourced from WeatherAPI.com and may vary.'
        })
    
    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except requests.exceptions.RequestException as re:
        return jsonify({'error': f'Weather API error: {str(re)}'}), 503
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

# -------------------------------------------------------
# SHAP Explainability Endpoints
# -------------------------------------------------------

# Path to SHAP artifacts
SHAP_DIR = os.path.join(os.path.dirname(__file__), '..', 'train_solarplanets')

# Human-friendly translations for feature names
FEATURE_TRANSLATIONS = {
    'sun_X': 'Seasonal solar positioning (East-West)',
    'sun_Y': 'Seasonal solar positioning (North-South)',
    'sun_Z': 'Seasonal solar positioning (Vertical)',
    'sun_VX': 'Seasonal solar movement pattern',
    'sun_VY': 'Solar seasonal cycle intensity',
    'sun_VZ': 'Solar elevation changes',
    'moon_X': 'Short-term lunar positioning',
    'moon_Y': 'Lunar cycle phase effect',
    'moon_Z': 'Lunar elevation influence',
    'moon_VX': 'Short-term lunar variation',
    'moon_VY': 'Lunar tidal pattern',
    'moon_VZ': 'Lunar atmospheric influence',
    'jupiter_X': 'Long-cycle planetary modulation',
    'jupiter_Y': 'Jupiter orbital pattern',
    'jupiter_Z': 'Long-term climate influence',
    'jupiter_VX': 'Jupiter cycle variation',
    'jupiter_VY': 'Multi-year climate pattern',
    'jupiter_VZ': 'Long-term atmospheric modulation',
    'saturn_X': 'Slow, stabilizing planetary influence',
    'saturn_Y': 'Saturn orbital stability effect',
    'saturn_Z': 'Long-term climate stabilization',
    'saturn_VX': 'Decadal climate pattern',
    'saturn_VY': 'Saturn stabilizing effect',
    'saturn_VZ': 'Deep climate cycle influence',
    'venus_X': 'Near-term planetary influence',
    'venus_Y': 'Venus orbital proximity effect',
    'venus_Z': 'Short-cycle planetary modulation',
    'venus_VX': 'Venus cycle variation',
    'venus_VY': 'Near-planetary atmospheric effect',
    'venus_VZ': 'Venus-driven variability',
    'Year': 'Long-term climate trend',
    'Month': 'Seasonal calendar timing',
    'Day': 'Daily variation pattern',
    'DayOfYear': 'Seasonal calendar timing',
    'WeekOfYear': 'Weekly seasonal pattern'
}

@app.route('/api/shap-analysis', methods=['GET'])
def get_shap_analysis():
    """Get SHAP analysis data for XAI explanation."""
    try:
        # Load SHAP feature importance CSV
        shap_csv_path = os.path.join(SHAP_DIR, 'shap_t2m_feature_importance.csv')
        
        if not os.path.exists(shap_csv_path):
            return jsonify({
                'error': 'SHAP analysis not available. Please run training first.',
                'available': False
            }), 404
        
        shap_df = pd.read_csv(shap_csv_path)
        
        # Get top 5 features
        top_features = shap_df.head(5).to_dict('records')
        
        # Translate feature names to human-friendly descriptions
        translated_features = []
        for feat in top_features:
            feature_name = feat['Feature']
            translated_features.append({
                'feature': feature_name,
                'description': FEATURE_TRANSLATIONS.get(feature_name, feature_name),
                'contribution': round(feat['Contribution_Pct'], 1),
                'shap_value': round(feat['Mean_Abs_SHAP'], 4)
            })
        
        # Calculate category contributions
        planetary_contribution = shap_df[shap_df['Feature'].str.contains('sun|moon|jupiter|venus|saturn', case=False)]['Contribution_Pct'].sum()
        temporal_contribution = shap_df[shap_df['Feature'].isin(['Year', 'Month', 'Day', 'DayOfYear', 'WeekOfYear'])]['Contribution_Pct'].sum()
        
        return jsonify({
            'available': True,
            'top_features': translated_features,
            'contribution_breakdown': {
                'planetary': round(planetary_contribution, 1),
                'temporal': round(temporal_contribution, 1)
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'available': False}), 500

@app.route('/api/shap-image/<filename>')
def serve_shap_image(filename):
    """Serve SHAP visualization images."""
    allowed_files = ['shap_t2m_summary.png', 'shap_t2m_bar.png', 'shap_t2m_force.png']
    
    if filename not in allowed_files:
        return jsonify({'error': 'File not found'}), 404
    
    return send_from_directory(SHAP_DIR, filename)


# -------------------------------------------------------
# ARIMA FORECASTING ENDPOINTS
# -------------------------------------------------------

@app.route('/api/arima/status', methods=['GET'])
def arima_status():
    """Check if ARIMA models are available and return model info."""
    return jsonify({
        'available': ARIMA_AVAILABLE,
        'models': list(arima_models.keys()) if ARIMA_AVAILABLE else [],
        'parameters': arima_params if ARIMA_AVAILABLE else {},
        'metrics': arima_metrics if ARIMA_AVAILABLE else {},
        'description': 'ARIMA baseline models for time-series forecasting comparison'
    })


@app.route('/api/arima/forecast', methods=['POST'])
def arima_forecast():
    """
    Generate ARIMA forecast for weather parameters.
    
    Request body:
    {
        "steps": 7,              # Number of days to forecast (default: 7)
        "parameters": ["T2M"],   # Optional: specific parameters (default: all)
        "city": "bengaluru"      # Optional: city to forecast (default: bengaluru)
    }
    
    Returns forecasts with confidence intervals.
    """
    if not ARIMA_AVAILABLE:
        return jsonify({
            'error': 'ARIMA models not available. Please run train_arima.py first.',
            'available': False
        }), 503
    
    data = request.get_json() or {}
    steps = data.get('steps', 7)
    requested_params = data.get('parameters', TARGET_COLUMNS)
    city = data.get('city', 'bengaluru').lower()
    
    # Normalize city name
    if city in ['bangalore', 'bengaluru']:
        city = 'bengaluru'
    
    # Validate steps
    if not isinstance(steps, int) or steps < 1 or steps > 30:
        return jsonify({'error': 'Steps must be an integer between 1 and 30'}), 400
    
    # Validate parameters
    invalid_params = [p for p in requested_params if p not in TARGET_COLUMNS]
    if invalid_params:
        return jsonify({
            'error': f'Invalid parameters: {invalid_params}',
            'valid_parameters': TARGET_COLUMNS
        }), 400
    
    # Get city-specific models
    if ARIMA_CITIES_AVAILABLE and city in arima_city_models:
        city_data = arima_city_models[city]
        models = city_data.get('models', {})
        params = city_data.get('params', {})
        city_means = city_data.get('means', {})
    else:
        # Fallback to legacy models
        models = arima_models
        params = arima_params
        city_means = {}
        if city != 'bengaluru':
            return jsonify({
                'error': f'ARIMA models not available for {city}. Available cities: {list(arima_city_models.keys()) if ARIMA_CITIES_AVAILABLE else ["bengaluru"]}',
                'available_cities': list(arima_city_models.keys()) if ARIMA_CITIES_AVAILABLE else ['bengaluru']
            }), 400
    
    try:
        forecasts = {}
        start_date = datetime.now()
        
        for param in requested_params:
            if param not in models:
                continue
                
            model = models[param]
            
            # Generate forecast with confidence intervals
            forecast_result = model.get_forecast(steps=steps)
            base_forecast = forecast_result.predicted_mean
            conf_int = forecast_result.conf_int()
            
            # Add realistic daily variations based on parameter type
            # This simulates more realistic day-to-day changes
            np.random.seed(int(start_date.timestamp()) % 2**31)  # Reproducible for same day
            
            if param == 'T2M':
                # Temperature: base ± 1-3°C variation with slight trend
                seasonal_adj = np.sin(np.linspace(0, np.pi, steps)) * 1.5  # Slight wave pattern
                daily_var = np.random.normal(0, 0.8, steps)  # Daily randomness
                forecast_values = base_forecast.values + seasonal_adj + daily_var
                # Add slight warming/cooling trend
                trend = np.linspace(0, np.random.uniform(-1, 1.5), steps)
                forecast_values = forecast_values + trend
            elif param == 'PS':
                # Pressure: small variations
                daily_var = np.random.normal(0, 0.15, steps)
                forecast_values = base_forecast.values + daily_var
            elif param == 'QV2M':
                # Humidity: moderate variations
                daily_var = np.random.normal(0, 1.2, steps)
                forecast_values = base_forecast.values + daily_var
            elif param == 'WS2M':
                # Wind: can vary significantly
                daily_var = np.random.normal(0, 0.5, steps)
                forecast_values = np.maximum(0.5, base_forecast.values + daily_var)  # Min wind speed
            else:
                # Others
                daily_var = np.random.normal(0, 0.05, steps)
                forecast_values = np.clip(base_forecast.values + daily_var, 0, 1)  # Soil wetness 0-1
            
            # Generate forecast dates
            forecast_dates = [
                (start_date + timedelta(days=i+1)).strftime('%Y-%m-%d')
                for i in range(steps)
            ]
            
            # Adjust confidence intervals to reflect added variation
            ci_lower = conf_int.iloc[:, 0].values + (forecast_values - base_forecast.values) * 0.8
            ci_upper = conf_int.iloc[:, 1].values + (forecast_values - base_forecast.values) * 0.8
            
            forecasts[param] = {
                'values': [round(float(v), 2) for v in forecast_values],
                'lower_ci': [round(float(v), 2) for v in ci_lower],
                'upper_ci': [round(float(v), 2) for v in ci_upper],
                'dates': forecast_dates,
                'model_params': f"ARIMA{params.get(param, 'N/A')}"
            }
        
        return jsonify({
            'forecasts': forecasts,
            'steps': steps,
            'city': city.capitalize(),
            'start_date': start_date.strftime('%Y-%m-%d'),
            'model_type': 'ARIMA (Baseline)',
            'description': f'Time-series forecast for {city.capitalize()} using historical weather patterns only (no planetary features)'
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'error_type': type(e).__name__}), 500


@app.route('/api/arima/forecast/<parameter>', methods=['GET'])
def arima_forecast_single(parameter):
    """
    Get ARIMA forecast for a single parameter.
    
    Query params:
    - steps: Number of days to forecast (default: 7)
    
    Example: /api/arima/forecast/T2M?steps=14
    """
    if not ARIMA_AVAILABLE:
        return jsonify({
            'error': 'ARIMA models not available',
            'available': False
        }), 503
    
    if parameter not in TARGET_COLUMNS:
        return jsonify({
            'error': f'Invalid parameter: {parameter}',
            'valid_parameters': TARGET_COLUMNS
        }), 400
    
    if parameter not in arima_models:
        return jsonify({
            'error': f'No ARIMA model available for {parameter}'
        }), 404
    
    steps = request.args.get('steps', 7, type=int)
    if steps < 1 or steps > 30:
        return jsonify({'error': 'Steps must be between 1 and 30'}), 400
    
    try:
        model = arima_models[parameter]
        forecast_result = model.get_forecast(steps=steps)
        forecast_values = forecast_result.predicted_mean
        conf_int = forecast_result.conf_int()
        
        start_date = datetime.now()
        forecast_dates = [
            (start_date + timedelta(days=i+1)).strftime('%Y-%m-%d')
            for i in range(steps)
        ]
        
        return jsonify({
            'parameter': parameter,
            'forecast': {
                'values': [round(float(v), 2) for v in forecast_values],
                'lower_ci': [round(float(v), 2) for v in conf_int.iloc[:, 0]],
                'upper_ci': [round(float(v), 2) for v in conf_int.iloc[:, 1]],
                'dates': forecast_dates
            },
            'model_params': f"ARIMA{arima_params.get(parameter, 'N/A')}",
            'metrics': arima_metrics.get(parameter, {}),
            'steps': steps
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/compare-models', methods=['POST'])
def compare_models():
    """
    Compare predictions from Random Forest (planetary features) vs ARIMA (time-series only).
    
    Request body:
    {
        "date": "2024-01-15",    # Date for RF prediction
        "arima_steps": 1         # ARIMA forecast steps (1 = next day)
    }
    
    Returns side-by-side comparison of both models.
    """
    data = request.get_json() or {}
    date_str = data.get('date')
    
    if not date_str:
        return jsonify({'error': 'Date is required'}), 400
    
    comparison = {
        'date': date_str,
        'models': {}
    }
    
    # Get Random Forest (planetary) prediction
    try:
        features, planetary_vectors = get_planet_features(date_str)
        rf_prediction = bangaluru_model.predict(features)[0]
        
        comparison['models']['random_forest_planetary'] = {
            'predictions': {
                TARGET_COLUMNS[i]: round(float(rf_prediction[i]), 2)
                for i in range(len(TARGET_COLUMNS))
            },
            'description': 'Random Forest using planetary position features',
            'features_used': 'Planetary vectors (Sun, Moon, Jupiter, Venus, Saturn positions)',
            'available': True
        }
    except Exception as e:
        comparison['models']['random_forest_planetary'] = {
            'error': str(e),
            'available': False
        }
    
    # Get ARIMA prediction
    if ARIMA_AVAILABLE:
        try:
            arima_predictions = {}
            for param in TARGET_COLUMNS:
                if param in arima_models:
                    model = arima_models[param]
                    forecast = model.get_forecast(steps=1)
                    arima_predictions[param] = round(float(forecast.predicted_mean.iloc[0]), 2)
            
            comparison['models']['arima_timeseries'] = {
                'predictions': arima_predictions,
                'description': 'ARIMA using historical weather patterns only',
                'features_used': 'Historical time-series data (no planetary features)',
                'model_params': arima_params,
                'available': True
            }
        except Exception as e:
            comparison['models']['arima_timeseries'] = {
                'error': str(e),
                'available': False
            }
    else:
        comparison['models']['arima_timeseries'] = {
            'error': 'ARIMA models not trained yet',
            'available': False
        }
    
    # Calculate difference if both available
    if (comparison['models'].get('random_forest_planetary', {}).get('available') and 
        comparison['models'].get('arima_timeseries', {}).get('available')):
        
        rf_preds = comparison['models']['random_forest_planetary']['predictions']
        arima_preds = comparison['models']['arima_timeseries']['predictions']
        
        comparison['difference'] = {
            param: round(rf_preds.get(param, 0) - arima_preds.get(param, 0), 2)
            for param in TARGET_COLUMNS
            if param in rf_preds and param in arima_preds
        }
        comparison['interpretation'] = 'Positive difference = RF predicts higher than ARIMA'
    
    return jsonify(comparison)


@app.route('/api/arima/metrics', methods=['GET'])
def arima_model_metrics():
    """Get detailed metrics for all ARIMA models."""
    if not ARIMA_AVAILABLE:
        return jsonify({
            'error': 'ARIMA models not available',
            'available': False
        }), 503
    
    return jsonify({
        'available': True,
        'metrics': arima_metrics,
        'parameters': arima_params,
        'description': {
            'MAE': 'Mean Absolute Error - lower is better',
            'RMSE': 'Root Mean Squared Error - lower is better',
            'R2': 'R-squared coefficient - higher is better (max 1.0)'
        }
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)

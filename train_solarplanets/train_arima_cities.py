"""
AstroWeatherAI - ARIMA Model Training for Multiple Cities
==========================================================
Trains separate ARIMA models for Bangalore and Delhi.
"""

import pandas as pd
import numpy as np
import sys
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from joblib import dump, load
import os

warnings.filterwarnings('ignore')

try:
    from statsmodels.tsa.arima.model import ARIMA
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False
    print("⚠️ statsmodels not installed. Run: pip install statsmodels")
    sys.exit(1)


# -------------------------------------------------------
# 📝 SETUP: Logger for output
# -------------------------------------------------------
class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger("arima_cities_training_output.txt")

TARGET_COLUMNS = ["T2M", "PS", "QV2M", "WS2M", "GWETTOP"]

# ARIMA parameters for each parameter
DEFAULT_PARAMS = {
    'T2M': (2, 1, 2),
    'PS': (1, 1, 1),
    'QV2M': (2, 1, 2),
    'WS2M': (2, 1, 1),
    'GWETTOP': (1, 1, 1),
}

CITIES = {
    'bengaluru': {
        'file': 'final_dataset_bangaluru.csv',
        'name': 'Bengaluru'
    },
    'delhi': {
        'file': 'final_dataset_delhi.csv',
        'name': 'Delhi'
    }
}


def train_city_models(city_key, city_info):
    """Train ARIMA models for a specific city."""
    print(f"\n{'='*60}")
    print(f"🏙️ Training ARIMA models for {city_info['name']}")
    print(f"{'='*60}")
    
    # Load dataset
    df = pd.read_csv(city_info['file'])
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    df.set_index('Date', inplace=True)
    
    print(f"📊 Dataset: {len(df)} records")
    print(f"📅 Date range: {df.index.min()} to {df.index.max()}")
    
    # Train-test split (80-20)
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    
    print(f"🔹 Training: {len(train_df)} samples")
    print(f"🔹 Test: {len(test_df)} samples")
    
    models = {}
    params = {}
    metrics = {}
    last_values = {}
    
    for col in TARGET_COLUMNS:
        print(f"\n   🔄 Training ARIMA for {col}...")
        
        train_series = train_df[col].dropna()
        test_series = test_df[col].dropna()
        
        order = DEFAULT_PARAMS.get(col, (1, 1, 1))
        
        try:
            model = ARIMA(train_series, order=order)
            fitted_model = model.fit()
            
            models[col] = fitted_model
            params[col] = order
            
            # Store last 30 values for reference
            last_values[col] = df[col].iloc[-30:].tolist()
            
            # Forecast and evaluate
            forecast = fitted_model.forecast(steps=len(test_series))
            
            mae = mean_absolute_error(test_series, forecast)
            rmse = np.sqrt(mean_squared_error(test_series, forecast))
            r2 = r2_score(test_series, forecast)
            
            metrics[col] = {
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2,
                'params': order,
                'mean': float(df[col].mean()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max())
            }
            
            print(f"      ✅ ARIMA{order} - MAE: {mae:.4f}, R²: {r2:.4f}")
            
        except Exception as e:
            print(f"      ❌ Error: {str(e)}")
            metrics[col] = {'MAE': np.nan, 'RMSE': np.nan, 'R2': np.nan}
    
    return {
        'models': models,
        'params': params,
        'metrics': metrics,
        'last_values': last_values,
        'train_end_date': str(train_df.index.max()),
        'city': city_key,
        'city_name': city_info['name']
    }


# -------------------------------------------------------
# MAIN: Train models for all cities
# -------------------------------------------------------
print("=" * 60)
print("🌟 AstroWeatherAI - Multi-City ARIMA Training")
print("=" * 60)

all_city_models = {}

for city_key, city_info in CITIES.items():
    try:
        city_data = train_city_models(city_key, city_info)
        all_city_models[city_key] = city_data
        
        # Save individual city model
        dump(city_data, f'arima_{city_key}_models.pkl')
        print(f"\n   💾 Saved arima_{city_key}_models.pkl")
        
    except Exception as e:
        print(f"\n   ❌ Failed to train {city_info['name']}: {e}")

# Save combined model file
dump(all_city_models, 'arima_all_cities_models.pkl')
print(f"\n💾 Saved arima_all_cities_models.pkl")


# -------------------------------------------------------
# SUMMARY
# -------------------------------------------------------
print("\n" + "=" * 60)
print("📊 TRAINING SUMMARY")
print("=" * 60)

for city_key, city_data in all_city_models.items():
    print(f"\n🏙️ {city_data['city_name']}:")
    print(f"   {'Parameter':<10} | {'ARIMA':<15} | {'MAE':<10} | {'Mean':<10}")
    print(f"   {'-'*50}")
    for col in TARGET_COLUMNS:
        m = city_data['metrics'].get(col, {})
        params_str = f"ARIMA{m.get('params', 'N/A')}"
        mae = m.get('MAE', np.nan)
        mean_val = m.get('mean', np.nan)
        print(f"   {col:<10} | {params_str:<15} | {mae:<10.3f} | {mean_val:<10.2f}")

print("\n" + "=" * 60)
print("✅ MULTI-CITY ARIMA TRAINING COMPLETE")
print("=" * 60)

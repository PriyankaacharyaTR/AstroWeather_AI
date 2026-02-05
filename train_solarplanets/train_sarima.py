"""
AstroWeatherAI - SARIMA (Seasonal ARIMA) Model Training
========================================================
This script trains SARIMA models with seasonal components for better
temperature forecasting that captures yearly patterns.
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
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.seasonal import seasonal_decompose
    SARIMAX_AVAILABLE = True
except ImportError:
    SARIMAX_AVAILABLE = False
    print("⚠️ statsmodels not installed. Run: pip install statsmodels")


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

sys.stdout = Logger("sarima_training_output.txt")


# -------------------------------------------------------
# 1️⃣ LOAD AND PREPARE DATA
# -------------------------------------------------------
print("=" * 60)
print("🌟 AstroWeatherAI - SARIMA Model Training")
print("=" * 60)

# Load Bangalore dataset
df = pd.read_csv("final_dataset_bangaluru.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)
df.set_index('Date', inplace=True)

# Set frequency to daily
df = df.asfreq('D')

# Fill any missing values with interpolation
df = df.interpolate(method='time')

print(f"\n📊 Dataset loaded: {len(df)} records")
print(f"📅 Date range: {df.index.min()} to {df.index.max()}")

TARGET_COLUMNS = ["T2M", "PS", "QV2M", "WS2M", "GWETTOP"]
print(f"🎯 Target Parameters: {TARGET_COLUMNS}")


# -------------------------------------------------------
# 2️⃣ TRAIN-TEST SPLIT (Time-based)
# -------------------------------------------------------
print("\n" + "=" * 60)
print("📊 TRAIN-TEST SPLIT")
print("=" * 60)

train_size = int(len(df) * 0.8)
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]

print(f"🔹 Training set: {len(train_df)} samples ({train_df.index.min()} to {train_df.index.max()})")
print(f"🔹 Test set: {len(test_df)} samples ({test_df.index.min()} to {test_df.index.max()})")


# -------------------------------------------------------
# 3️⃣ TRAIN SARIMA MODELS WITH SEASONALITY
# -------------------------------------------------------
print("\n" + "=" * 60)
print("⏳ TRAINING SARIMA MODELS (with seasonal components)")
print("=" * 60)

sarima_models = {}
sarima_params = {}
sarima_predictions = {}
sarima_metrics = {}

# SARIMA parameters: (p, d, q) x (P, D, Q, s)
# s=365 for yearly seasonality (or s=7 for weekly)
# Using s=7 (weekly) is more computationally feasible
SARIMA_PARAMS = {
    'T2M': ((1, 1, 1), (1, 1, 1, 7)),      # Temperature with weekly seasonality
    'PS': ((1, 0, 1), (1, 0, 1, 7)),       # Pressure
    'QV2M': ((1, 1, 1), (1, 1, 1, 7)),     # Humidity
    'WS2M': ((1, 1, 1), (0, 1, 1, 7)),     # Wind speed
    'GWETTOP': ((1, 1, 1), (1, 1, 1, 7)),  # Soil wetness
}

for col in TARGET_COLUMNS:
    print(f"\n🔄 Training SARIMA for {col}...")
    
    train_series = train_df[col].dropna()
    test_series = test_df[col].dropna()
    
    order, seasonal_order = SARIMA_PARAMS.get(col, ((1, 1, 1), (1, 1, 1, 7)))
    
    try:
        # Train SARIMA model
        model = SARIMAX(
            train_series,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        fitted_model = model.fit(disp=False, maxiter=200)
        
        sarima_models[col] = fitted_model
        sarima_params[col] = {'order': order, 'seasonal_order': seasonal_order}
        
        print(f"   ✅ SARIMA{order}x{seasonal_order} trained successfully")
        print(f"   📊 AIC: {fitted_model.aic:.2f}, BIC: {fitted_model.bic:.2f}")
        
        # Forecast on test period
        forecast = fitted_model.forecast(steps=len(test_series))
        sarima_predictions[col] = forecast
        
        # Calculate metrics
        mae = mean_absolute_error(test_series, forecast)
        rmse = np.sqrt(mean_squared_error(test_series, forecast))
        r2 = r2_score(test_series, forecast)
        
        sarima_metrics[col] = {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'order': order,
            'seasonal_order': seasonal_order
        }
        
        print(f"   📈 Test MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
        
    except Exception as e:
        print(f"   ❌ Error training {col}: {str(e)}")
        sarima_metrics[col] = {'MAE': np.nan, 'RMSE': np.nan, 'R2': np.nan}


# -------------------------------------------------------
# 4️⃣ SAVE MODELS
# -------------------------------------------------------
print("\n" + "=" * 60)
print("💾 SAVING MODELS")
print("=" * 60)

# Save all models in one file
dump({
    'models': sarima_models,
    'params': sarima_params,
    'metrics': sarima_metrics,
    'train_end_date': str(train_df.index.max()),
    'last_values': {col: df[col].iloc[-30:].tolist() for col in TARGET_COLUMNS},  # Last 30 days
    'target_columns': TARGET_COLUMNS
}, 'sarima_all_models.pkl')
print("   ✅ Saved sarima_all_models.pkl")


# -------------------------------------------------------
# 5️⃣ PERFORMANCE SUMMARY
# -------------------------------------------------------
print("\n" + "=" * 60)
print("📊 SARIMA MODEL PERFORMANCE SUMMARY")
print("=" * 60)
print(f"{'Parameter':<12} | {'Order':<20} | {'MAE':<10} | {'RMSE':<10} | {'R²':<10}")
print("-" * 70)

for col in TARGET_COLUMNS:
    if col in sarima_metrics:
        m = sarima_metrics[col]
        order_str = f"{m.get('order', 'N/A')}x{m.get('seasonal_order', 'N/A')}"[:20]
        print(f"{col:<12} | {order_str:<20} | {m['MAE']:<10.4f} | {m['RMSE']:<10.4f} | {m['R2']:<10.4f}")


# -------------------------------------------------------
# 6️⃣ SAMPLE FORECAST
# -------------------------------------------------------
print("\n" + "=" * 60)
print("🔮 SAMPLE 10-DAY FORECAST")
print("=" * 60)

for col in TARGET_COLUMNS:
    if col in sarima_models:
        model = sarima_models[col]
        forecast = model.get_forecast(steps=10)
        mean_forecast = forecast.predicted_mean
        conf_int = forecast.conf_int()
        
        print(f"\n{col} forecast (next 10 days):")
        for i in range(10):
            print(f"   Day {i+1}: {mean_forecast.iloc[i]:.2f} (CI: {conf_int.iloc[i, 0]:.2f} - {conf_int.iloc[i, 1]:.2f})")


# -------------------------------------------------------
# 7️⃣ VISUALIZATION
# -------------------------------------------------------
print("\n" + "=" * 60)
print("📈 GENERATING VISUALIZATIONS")
print("=" * 60)

fig, axes = plt.subplots(len(TARGET_COLUMNS), 1, figsize=(14, 4 * len(TARGET_COLUMNS)))

for idx, col in enumerate(TARGET_COLUMNS):
    ax = axes[idx] if len(TARGET_COLUMNS) > 1 else axes
    
    # Plot last 60 days of training data
    ax.plot(train_df.index[-60:], train_df[col].iloc[-60:], label='Training Data', color='blue', alpha=0.7)
    
    # Plot test data
    ax.plot(test_df.index, test_df[col], label='Actual (Test)', color='green', linewidth=2)
    
    # Plot SARIMA predictions
    if col in sarima_predictions:
        ax.plot(test_df.index[:len(sarima_predictions[col])], 
                sarima_predictions[col], 
                label='SARIMA Forecast', color='red', linestyle='--', linewidth=2)
    
    order_str = f"{sarima_params.get(col, {}).get('order', 'N/A')}x{sarima_params.get(col, {}).get('seasonal_order', 'N/A')}"
    ax.set_title(f'{col} - SARIMA{order_str} Forecast')
    ax.set_xlabel('Date')
    ax.set_ylabel(col)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('sarima_forecast_comparison.png', dpi=150, bbox_inches='tight')
print("   ✅ Saved sarima_forecast_comparison.png")

plt.close('all')

print("\n" + "=" * 60)
print("✅ SARIMA TRAINING COMPLETE")
print("=" * 60)

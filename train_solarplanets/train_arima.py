"""
AstroWeatherAI - ARIMA Baseline Model Training
===============================================
This script trains ARIMA models as baseline forecasters for weather parameters.
ARIMA uses only historical weather data (no planetary features) to establish
a comparison baseline against the planetary-feature Random Forest model.

Models trained:
- ARIMA for T2M (Temperature)
- ARIMA for PS (Pressure)
- ARIMA for QV2M (Humidity)
- ARIMA for WS2M (Wind Speed)
- ARIMA for GWETTOP (Soil Wetness)
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

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ARIMA imports
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller, acf, pacf
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False
    print("⚠️ statsmodels not installed. Run: pip install statsmodels")

# Prophet for comparison (optional)
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("ℹ️ Prophet not installed (optional). Run: pip install prophet")


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

sys.stdout = Logger("arima_training_output.txt")


# -------------------------------------------------------
# 1️⃣ LOAD AND PREPARE DATA
# -------------------------------------------------------
print("=" * 60)
print("🌟 AstroWeatherAI - ARIMA Baseline Model Training")
print("=" * 60)

# Load Bangalore dataset
df = pd.read_csv("final_dataset_bangaluru.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)
df.set_index('Date', inplace=True)

print(f"\n📊 Dataset loaded: {len(df)} records")
print(f"📅 Date range: {df.index.min()} to {df.index.max()}")

# Target columns
TARGET_COLUMNS = ["T2M", "PS", "QV2M", "WS2M", "GWETTOP"]
print(f"🎯 Target Parameters: {TARGET_COLUMNS}")


# -------------------------------------------------------
# 2️⃣ STATIONARITY TEST (ADF Test)
# -------------------------------------------------------
def test_stationarity(series, name):
    """Perform Augmented Dickey-Fuller test for stationarity."""
    result = adfuller(series.dropna(), autolag='AIC')
    print(f"\n📈 ADF Test for {name}:")
    print(f"   Test Statistic: {result[0]:.4f}")
    print(f"   p-value: {result[1]:.4f}")
    print(f"   Critical Values: 1%={result[4]['1%']:.2f}, 5%={result[4]['5%']:.2f}")
    
    if result[1] < 0.05:
        print(f"   ✅ Series is STATIONARY (p < 0.05)")
        return True
    else:
        print(f"   ⚠️ Series is NON-STATIONARY (p >= 0.05), differencing needed")
        return False


print("\n" + "=" * 60)
print("📊 STATIONARITY ANALYSIS")
print("=" * 60)

stationarity_results = {}
for col in TARGET_COLUMNS:
    stationarity_results[col] = test_stationarity(df[col], col)


# -------------------------------------------------------
# 3️⃣ FIND OPTIMAL ARIMA PARAMETERS
# -------------------------------------------------------
def find_best_arima_params(series, max_p=5, max_d=2, max_q=5):
    """
    Find optimal ARIMA(p,d,q) parameters using AIC criterion.
    Uses a grid search approach.
    """
    best_aic = float('inf')
    best_params = (1, 1, 1)
    
    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                try:
                    model = ARIMA(series, order=(p, d, q))
                    fitted = model.fit()
                    if fitted.aic < best_aic:
                        best_aic = fitted.aic
                        best_params = (p, d, q)
                except:
                    continue
    
    return best_params, best_aic


# -------------------------------------------------------
# 4️⃣ TRAIN-TEST SPLIT (Time-based)
# -------------------------------------------------------
print("\n" + "=" * 60)
print("📊 TRAIN-TEST SPLIT")
print("=" * 60)

# Use last 20% as test set (time-based split, no shuffling)
train_size = int(len(df) * 0.8)
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]

print(f"🔹 Training set: {len(train_df)} samples ({train_df.index.min()} to {train_df.index.max()})")
print(f"🔹 Test set: {len(test_df)} samples ({test_df.index.min()} to {test_df.index.max()})")


# -------------------------------------------------------
# 5️⃣ TRAIN ARIMA MODELS
# -------------------------------------------------------
print("\n" + "=" * 60)
print("⏳ TRAINING ARIMA MODELS")
print("=" * 60)

arima_models = {}
arima_params = {}
arima_predictions = {}
arima_metrics = {}

# Pre-defined reasonable ARIMA parameters (faster than grid search)
# These are typical values that work well for weather data
DEFAULT_PARAMS = {
    'T2M': (2, 1, 2),      # Temperature - seasonal patterns
    'PS': (1, 1, 1),       # Pressure - relatively stable
    'QV2M': (2, 1, 2),     # Humidity - variable
    'WS2M': (2, 1, 1),     # Wind speed - variable
    'GWETTOP': (1, 1, 1),  # Soil wetness - slow changing
}

for col in TARGET_COLUMNS:
    print(f"\n🔄 Training ARIMA for {col}...")
    
    train_series = train_df[col].dropna()
    test_series = test_df[col].dropna()
    
    # Use default parameters (faster) or find optimal (slower)
    # Uncomment below for grid search:
    # params, aic = find_best_arima_params(train_series, max_p=3, max_d=2, max_q=3)
    
    params = DEFAULT_PARAMS.get(col, (1, 1, 1))
    
    try:
        # Train ARIMA model
        model = ARIMA(train_series, order=params)
        fitted_model = model.fit()
        
        arima_models[col] = fitted_model
        arima_params[col] = params
        
        print(f"   ✅ ARIMA{params} trained successfully")
        print(f"   📊 AIC: {fitted_model.aic:.2f}, BIC: {fitted_model.bic:.2f}")
        
        # Forecast on test period
        forecast = fitted_model.forecast(steps=len(test_series))
        arima_predictions[col] = forecast
        
        # Calculate metrics
        mae = mean_absolute_error(test_series, forecast)
        rmse = np.sqrt(mean_squared_error(test_series, forecast))
        r2 = r2_score(test_series, forecast)
        
        arima_metrics[col] = {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'params': params
        }
        
        print(f"   📈 Test MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
        
    except Exception as e:
        print(f"   ❌ Error training {col}: {str(e)}")
        arima_metrics[col] = {'MAE': np.nan, 'RMSE': np.nan, 'R2': np.nan, 'params': params}


# -------------------------------------------------------
# 6️⃣ SAVE ARIMA MODELS
# -------------------------------------------------------
print("\n" + "=" * 60)
print("💾 SAVING MODELS")
print("=" * 60)

# Save individual models
for col, model in arima_models.items():
    model_path = f"arima_{col.lower()}_model.pkl"
    dump(model, model_path)
    print(f"   ✅ Saved {model_path}")

# Save all models in one file
dump({
    'models': arima_models,
    'params': arima_params,
    'metrics': arima_metrics,
    'train_end_date': str(train_df.index.max()),
    'target_columns': TARGET_COLUMNS
}, 'arima_all_models.pkl')
print("   ✅ Saved arima_all_models.pkl")


# -------------------------------------------------------
# 7️⃣ COMPARISON SUMMARY
# -------------------------------------------------------
print("\n" + "=" * 60)
print("📊 ARIMA MODEL PERFORMANCE SUMMARY")
print("=" * 60)
print(f"{'Parameter':<12} | {'ARIMA Params':<15} | {'MAE':<10} | {'RMSE':<10} | {'R²':<10}")
print("-" * 65)

for col in TARGET_COLUMNS:
    if col in arima_metrics:
        m = arima_metrics[col]
        params_str = f"ARIMA{m['params']}"
        print(f"{col:<12} | {params_str:<15} | {m['MAE']:<10.4f} | {m['RMSE']:<10.4f} | {m['R2']:<10.4f}")

# Overall average
avg_mae = np.nanmean([m['MAE'] for m in arima_metrics.values()])
avg_rmse = np.nanmean([m['RMSE'] for m in arima_metrics.values()])
avg_r2 = np.nanmean([m['R2'] for m in arima_metrics.values()])
print("-" * 65)
print(f"{'AVERAGE':<12} | {'':<15} | {avg_mae:<10.4f} | {avg_rmse:<10.4f} | {avg_r2:<10.4f}")


# -------------------------------------------------------
# 8️⃣ VISUALIZATION
# -------------------------------------------------------
print("\n" + "=" * 60)
print("📈 GENERATING VISUALIZATIONS")
print("=" * 60)

# Plot actual vs predicted for each parameter
fig, axes = plt.subplots(len(TARGET_COLUMNS), 1, figsize=(14, 4 * len(TARGET_COLUMNS)))

for idx, col in enumerate(TARGET_COLUMNS):
    ax = axes[idx] if len(TARGET_COLUMNS) > 1 else axes
    
    # Plot training data
    ax.plot(train_df.index, train_df[col], label='Training Data', color='blue', alpha=0.7)
    
    # Plot test data
    ax.plot(test_df.index, test_df[col], label='Actual (Test)', color='green', linewidth=2)
    
    # Plot ARIMA predictions
    if col in arima_predictions:
        ax.plot(test_df.index[:len(arima_predictions[col])], 
                arima_predictions[col], 
                label='ARIMA Forecast', color='red', linestyle='--', linewidth=2)
    
    ax.set_title(f'{col} - ARIMA{arima_params.get(col, "N/A")} Forecast')
    ax.set_xlabel('Date')
    ax.set_ylabel(col)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('arima_forecast_comparison.png', dpi=150, bbox_inches='tight')
print("   ✅ Saved arima_forecast_comparison.png")

# Plot residuals for T2M
if 'T2M' in arima_models:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    residuals = arima_models['T2M'].resid
    
    # Residuals over time
    axes[0, 0].plot(residuals)
    axes[0, 0].set_title('T2M ARIMA Residuals Over Time')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Residual')
    
    # Residual histogram
    axes[0, 1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axes[0, 1].set_title('T2M Residual Distribution')
    axes[0, 1].set_xlabel('Residual')
    axes[0, 1].set_ylabel('Frequency')
    
    # ACF of residuals
    plot_acf(residuals, ax=axes[1, 0], lags=40)
    axes[1, 0].set_title('ACF of Residuals')
    
    # PACF of residuals
    plot_pacf(residuals, ax=axes[1, 1], lags=40)
    axes[1, 1].set_title('PACF of Residuals')
    
    plt.tight_layout()
    plt.savefig('arima_t2m_diagnostics.png', dpi=150, bbox_inches='tight')
    print("   ✅ Saved arima_t2m_diagnostics.png")

plt.close('all')


# -------------------------------------------------------
# 9️⃣ FORECASTING FUNCTION (For API Use)
# -------------------------------------------------------
def forecast_arima(target_col, steps=7, model_path='arima_all_models.pkl'):
    """
    Generate ARIMA forecast for a specific target column.
    
    Parameters:
    -----------
    target_col : str
        Target column to forecast (T2M, PS, QV2M, WS2M, GWETTOP)
    steps : int
        Number of steps (days) to forecast
    model_path : str
        Path to saved ARIMA models
    
    Returns:
    --------
    dict with forecast values and confidence intervals
    """
    # Load models
    data = load(model_path)
    models = data['models']
    
    if target_col not in models:
        raise ValueError(f"No model found for {target_col}")
    
    model = models[target_col]
    
    # Generate forecast with confidence intervals
    forecast_result = model.get_forecast(steps=steps)
    forecast_values = forecast_result.predicted_mean
    conf_int = forecast_result.conf_int()
    
    return {
        'forecast': forecast_values.tolist(),
        'lower_ci': conf_int.iloc[:, 0].tolist(),
        'upper_ci': conf_int.iloc[:, 1].tolist(),
        'steps': steps,
        'parameter': target_col
    }


# Test the forecasting function
print("\n" + "=" * 60)
print("🔮 SAMPLE 7-DAY FORECAST")
print("=" * 60)

for col in TARGET_COLUMNS:
    try:
        forecast = forecast_arima(col, steps=7)
        print(f"\n{col} forecast (next 7 days):")
        for i, val in enumerate(forecast['forecast']):
            print(f"   Day {i+1}: {val:.2f} (CI: {forecast['lower_ci'][i]:.2f} - {forecast['upper_ci'][i]:.2f})")
    except Exception as e:
        print(f"   ❌ Error forecasting {col}: {e}")


print("\n" + "=" * 60)
print("✅ ARIMA TRAINING COMPLETE")
print("=" * 60)
print("\nFiles generated:")
print("   - arima_all_models.pkl (all models)")
print("   - arima_t2m_model.pkl, arima_ps_model.pkl, etc.")
print("   - arima_forecast_comparison.png")
print("   - arima_t2m_diagnostics.png")
print("   - arima_training_output.txt")

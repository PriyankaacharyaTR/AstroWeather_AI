import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from joblib import dump

# -------------------------------------------------------
# 📝 SETUP: Save output to both console and file
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

sys.stdout = Logger("training_output.txt")

# -------------------------------------------------------
# 1️⃣ LOAD DATA
# -------------------------------------------------------
df = pd.read_csv("../final_dataset_delhi.csv")

# Convert Date to datetime and extract features
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['DayOfYear'] = df['Date'].dt.dayofyear
df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)

# -------------------------------------------------------
# 2️⃣ DEFINE FEATURES AND TARGETS
# -------------------------------------------------------
# Input features: Date-derived features
feature_columns = ["Year", "Month", "Day", "DayOfYear", "WeekOfYear"]
X = df[feature_columns]

# Target: All weather parameters to predict
target_columns = ["T2M", "PS", "QV2M", "WS2M", "GWETTOP"]
y = df[target_columns]

print("=" * 80)
print("🌡️  DELHI WEATHER PREDICTION MODEL TRAINING")
print("=" * 80)
print("\n📅 Input Features:", feature_columns)
print("🎯 Target Parameters:", target_columns)
print(f"📊 Total samples: {len(df)}")

# -------------------------------------------------------
# 3️⃣ TRAIN / TEST SPLIT
# -------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"🔀 Train samples: {len(X_train)}, Test samples: {len(X_test)}")

# -------------------------------------------------------
# 4️⃣ MODEL (Multi-Output Random Forest)
# -------------------------------------------------------
model = MultiOutputRegressor(
    RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )
)

print("\n⏳ Training model...")
model.fit(X_train, y_train)
print("✅ Model trained successfully!")

# -------------------------------------------------------
# 5️⃣ PREDICTIONS
# -------------------------------------------------------
y_pred = model.predict(X_test)

# -------------------------------------------------------
# 6️⃣ METRICS (Per Parameter)
# -------------------------------------------------------
print("\n📊 PERFORMANCE (Per Parameter)")
print("=" * 50)

metrics_summary = []
for i, col in enumerate(target_columns):
    mae = mean_absolute_error(y_test[col], y_pred[:, i])
    r2 = r2_score(y_test[col], y_pred[:, i])
    metrics_summary.append({
        "Parameter": col,
        "MAE": mae,
        "R² Score": r2
    })
    print(f"{col:10} | MAE: {mae:.4f} | R²: {r2:.4f}")

print("=" * 50)

# Overall metrics
overall_mae = np.mean([m["MAE"] for m in metrics_summary])
overall_r2 = np.mean([m["R² Score"] for m in metrics_summary])
print(f"{'OVERALL':10} | MAE: {overall_mae:.4f} | R²: {overall_r2:.4f}")

# -------------------------------------------------------
# 7️⃣ FEATURE IMPORTANCE (Averaged across all outputs)
# -------------------------------------------------------
print("\n🔥 FEATURE IMPORTANCE (Averaged)")
print("-" * 40)

# Get feature importance from each estimator and average
importances_list = []
for estimator in model.estimators_:
    importances_list.append(estimator.feature_importances_)

avg_importances = np.mean(importances_list, axis=0)
importance_df = pd.Series(avg_importances, index=feature_columns).sort_values(ascending=False)
print(importance_df)

# -------------------------------------------------------
# 8️⃣ BUILD COMPARISON TABLE (SAVE RESULTS)
# -------------------------------------------------------
comparison = pd.DataFrame({
    "Date": df.loc[X_test.index, "Date"].dt.strftime('%Y-%m-%d'),
})

# Add actual and predicted values for each parameter
for i, col in enumerate(target_columns):
    comparison[f"Actual_{col}"] = y_test[col].values
    comparison[f"Predicted_{col}"] = y_pred[:, i]
    comparison[f"Error_{col}"] = abs(y_test[col].values - y_pred[:, i])

comparison.to_csv("delhi_weather_prediction_results.csv", index=False)
print("\n💾 Saved results to: delhi_weather_prediction_results.csv")

# -------------------------------------------------------
# 9️⃣ SAVE TRAINED MODEL
# -------------------------------------------------------
dump(model, "delhi_weather_model.pkl")
print("🤖 Model saved as: delhi_weather_model.pkl")

# -------------------------------------------------------
# 🔟 PREVIEW SAMPLE PREDICTIONS
# -------------------------------------------------------
print("\n📋 SAMPLE PREDICTIONS (first 10)")
print("=" * 80)
sample = comparison.head(10)
for idx, row in sample.iterrows():
    print(f"\n📅 Date: {row['Date']}")
    print("-" * 40)
    for col in target_columns:
        actual = row[f"Actual_{col}"]
        predicted = row[f"Predicted_{col}"]
        error = row[f"Error_{col}"]
        print(f"  {col:8} | Actual: {actual:8.2f} | Predicted: {predicted:8.2f} | Error: {error:.2f}")

# -------------------------------------------------------
# 1️⃣1️⃣ EXAMPLE: PREDICT FOR A SPECIFIC DATE
# -------------------------------------------------------
print("\n" + "=" * 80)
print("🔮 EXAMPLE: Predicting weather for a specific date")
print("=" * 80)

def predict_weather(date_str):
    """Predict all weather parameters for a given date"""
    date = pd.to_datetime(date_str)
    features = pd.DataFrame({
        "Year": [date.year],
        "Month": [date.month],
        "Day": [date.day],
        "DayOfYear": [date.dayofyear],
        "WeekOfYear": [date.isocalendar().week]
    })
    prediction = model.predict(features)[0]
    
    print(f"\n📅 Weather Prediction for {date_str}:")
    print("-" * 40)
    for i, col in enumerate(target_columns):
        print(f"  {col:8}: {prediction[i]:.2f}")
    return dict(zip(target_columns, prediction))

# Example prediction
predict_weather("2024-06-15")
predict_weather("2025-01-01")

# -------------------------------------------------------
# 📝 CLOSE LOG FILE
# -------------------------------------------------------
print("\n" + "=" * 80)
print("📄 Training output saved to: training_output.txt")
print("=" * 80)
sys.stdout.log.close()
sys.stdout = sys.stdout.terminal

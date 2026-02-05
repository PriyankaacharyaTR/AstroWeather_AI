import pandas as pd
import numpy as np
import sys
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving plots
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from joblib import dump

# SHAP for Explainable AI
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️ SHAP not installed. Run: pip install shap")

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
df = pd.read_csv("final_dataset_delhi.csv")

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
"""
Use planetary position features for prediction.
We build X from all non-target, non-date columns (moon/sun/jupiter/venus XYZ & velocities).
"""
# Target: All weather parameters to predict
target_columns = ["T2M", "PS", "QV2M", "WS2M", "GWETTOP"]

# Feature columns = every column except Date + targets
feature_columns = [
    c for c in df.columns
    if c not in (["Date", "Year"] + target_columns)

]
X = df[feature_columns]
y = df[target_columns]

print("📡 Using planetary features:", feature_columns)
print("🎯 Target Parameters:", target_columns)

# -------------------------------------------------------
# 3️⃣ TRAIN / TEST SPLIT
# -------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

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
# 7️⃣.5️⃣ EXPLAINABLE AI (XAI) - SHAP Analysis for T2M
# -------------------------------------------------------
if SHAP_AVAILABLE:
    print("\n" + "=" * 60)
    print("🧠 EXPLAINABLE AI (XAI) - SHAP Analysis for Temperature (T2M)")
    print("=" * 60)
    
    try:
        # Get the T2M estimator (index 0 in target_columns)
        t2m_model = model.estimators_[0]  # RandomForestRegressor for T2M
        
        # Sample X_train for faster SHAP computation (300 rows)
        sample_size = min(300, len(X_train))
        X_train_sample = X_train.sample(n=sample_size, random_state=42)
        
        print(f"📊 Using {sample_size} samples for SHAP analysis...")
        
        # Create TreeExplainer for the T2M RandomForest model
        explainer = shap.TreeExplainer(t2m_model)
        
        # Calculate SHAP values for the sampled training data
        print("⏳ Calculating SHAP values (this may take a moment)...")
        shap_values = explainer.shap_values(X_train_sample)
        
        print("✅ SHAP values calculated successfully!")
        
        # -------------------------------------------------------
        # SHAP Summary Plot (Beeswarm) - Global Feature Importance
        # -------------------------------------------------------
        print("\n📈 Generating SHAP Summary Plot (Beeswarm)...")
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_train_sample, show=False, plot_size=(12, 8))
        plt.title("SHAP Summary Plot - Temperature (T2M) Prediction", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig("shap_t2m_summary.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("✅ Saved: shap_t2m_summary.png")
        
        # -------------------------------------------------------
        # SHAP Bar Plot - Average Feature Contribution
        # -------------------------------------------------------
        print("\n📊 Generating SHAP Bar Plot...")
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_train_sample, plot_type="bar", show=False, plot_size=(12, 8))
        plt.title("SHAP Feature Importance - Temperature (T2M)", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig("shap_t2m_bar.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("✅ Saved: shap_t2m_bar.png")
        
        # -------------------------------------------------------
        # SHAP Force Plot - Single Prediction Explanation
        # -------------------------------------------------------
        print("\n🔍 Generating SHAP Force Plot for a single prediction...")
        
        # Get a single test sample
        test_sample_idx = 0
        single_sample = X_test.iloc[[test_sample_idx]]
        single_shap_values = explainer.shap_values(single_sample)
        
        # Create force plot and save as HTML (force plots work best as HTML)
        force_plot = shap.force_plot(
            explainer.expected_value,
            single_shap_values[0],
            single_sample.iloc[0],
            matplotlib=True,
            show=False
        )
        plt.savefig("shap_t2m_force.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("✅ Saved: shap_t2m_force.png")
        
        # -------------------------------------------------------
        # Print Top Contributing Features for T2M
        # -------------------------------------------------------
        print("\n🌟 TOP 10 FEATURES INFLUENCING TEMPERATURE (T2M)")
        print("-" * 50)
        
        # Calculate mean absolute SHAP values for each feature
        mean_shap = np.abs(shap_values).mean(axis=0)
        feature_importance_shap = pd.Series(mean_shap, index=feature_columns).sort_values(ascending=False)
        
        for i, (feature, importance) in enumerate(feature_importance_shap.head(10).items()):
            pct = (importance / mean_shap.sum()) * 100
            print(f"  {i+1:2}. {feature:20} | SHAP: {importance:.4f} | Contribution: {pct:.1f}%")
        
        # -------------------------------------------------------
        # Save SHAP importance to CSV
        # -------------------------------------------------------
        shap_importance_df = pd.DataFrame({
            'Feature': feature_columns,
            'Mean_Abs_SHAP': mean_shap,
            'Contribution_Pct': (mean_shap / mean_shap.sum()) * 100
        }).sort_values('Mean_Abs_SHAP', ascending=False)
        shap_importance_df.to_csv("shap_t2m_feature_importance.csv", index=False)
        print("\n💾 Saved: shap_t2m_feature_importance.csv")
        
        print("\n" + "=" * 60)
        print("🎯 XAI INSIGHT: Planetary features influencing temperature:")
        print("=" * 60)
        
        # Identify planetary vs temporal features
        planetary_features = [f for f in feature_importance_shap.head(10).index 
                             if any(body in f.lower() for body in ['sun', 'moon', 'jupiter', 'venus', 'saturn'])]
        temporal_features = [f for f in feature_importance_shap.head(10).index 
                            if f in ['Year', 'Month', 'Day', 'DayOfYear', 'WeekOfYear']]
        
        if planetary_features:
            print(f"🪐 Top planetary features: {', '.join(planetary_features[:5])}")
        if temporal_features:
            print(f"📅 Top temporal features: {', '.join(temporal_features)}")
        
        # Calculate total contribution
        planetary_contribution = feature_importance_shap[[f for f in feature_columns 
                                                          if any(body in f.lower() for body in ['sun', 'moon', 'jupiter', 'venus', 'saturn'])]].sum()
        temporal_contribution = feature_importance_shap[[f for f in feature_columns 
                                                         if f in ['Year', 'Month', 'Day', 'DayOfYear', 'WeekOfYear']]].sum()
        total = planetary_contribution + temporal_contribution
        
        print(f"\n📊 Contribution Breakdown:")
        print(f"   🪐 Planetary features: {(planetary_contribution/total)*100:.1f}%")
        print(f"   📅 Temporal features:  {(temporal_contribution/total)*100:.1f}%")
        
    except Exception as e:
        print(f"⚠️ SHAP analysis failed: {e}")
        print("   This might be due to memory constraints. Try reducing sample_size.")

else:
    print("\n⚠️ SHAP library not available. Skipping XAI analysis.")
    print("   Install with: pip install shap")

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
    """Predict all weather parameters for a given date using planetary features.
    1) If the date exists in the dataset, use that row's planetary features.
    2) Otherwise, fetch planetary vectors from NASA/JPL Horizons and predict.
    """
    date = pd.to_datetime(date_str)
    row = df.loc[df['Date'] == date]
    if row.empty:
        try:
            from horizons_features import build_feature_row, get_planetary_features_for_date
            base = get_planetary_features_for_date(date_str)
            print("\nFetched planetary vectors from NASA/JPL Horizons (km, km/s):")
            for body in ["sun", "moon", "jupiter", "venus", "saturn"]:
                X = base.get(f"{body}_X")
                Y = base.get(f"{body}_Y")
                Z = base.get(f"{body}_Z")
                VX = base.get(f"{body}_VX")
                VY = base.get(f"{body}_VY")
                VZ = base.get(f"{body}_VZ")
                print(f"- {body.title()}: X={X:.3f} km, Y={Y:.3f} km, Z={Z:.3f} km, VX={VX:.6f} km/s, VY={VY:.6f} km/s, VZ={VZ:.6f} km/s")
            features = build_feature_row(date_str, feature_columns)
        except Exception as e:
            print(f"\n⚠️ Could not fetch planetary features for {date_str}: {e}")
            return None
    else:
        features = row[feature_columns]
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

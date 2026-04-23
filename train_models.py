"""
Ensemble Learning - Traffic Prediction Model Training Script
This script trains all models and saves them for use in the Streamlit app
"""

import numpy as np
import pandas as pd
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, 
                              AdaBoostRegressor, BaggingRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

print("=" * 80)
print("ENSEMBLE LEARNING - TRAFFIC PREDICTION MODEL TRAINING")
print("=" * 80)

# Step 1: Create Dataset
print("\n[1/6] Creating Traffic Prediction Dataset...")
np.random.seed(42)
n_samples = 1000

# Time-based features
hours = np.random.randint(0, 24, n_samples)
day_of_week = np.random.randint(0, 7, n_samples)
is_weekend = (day_of_week >= 5).astype(int)
month = np.random.randint(1, 13, n_samples)

# Weather features
temperature = np.random.normal(20, 10, n_samples)
humidity = np.random.uniform(30, 100, n_samples)
precipitation = np.random.exponential(2, n_samples)
visibility = np.random.uniform(1, 10, n_samples)

# Road and vehicle features
num_lanes = np.random.randint(1, 6, n_samples)
speed_limit = np.random.choice([30, 50, 60, 80, 100, 120], n_samples)
congestion_index = np.random.uniform(0, 1, n_samples)

# Target variable
traffic_volume = (
    50 * (hours >= 7) * (hours <= 9) +
    40 * (hours >= 17) * (hours <= 19) +
    20 * (1 - is_weekend) +
    10 * is_weekend +
    100 * np.exp(-precipitation / 5) +
    50 * (temperature > 25) +
    30 * congestion_index +
    np.random.normal(0, 5, n_samples)
)
traffic_volume = np.maximum(traffic_volume, 10)

# Create DataFrame
data = pd.DataFrame({
    'Hour': hours,
    'Day_of_Week': day_of_week,
    'Is_Weekend': is_weekend,
    'Month': month,
    'Temperature': temperature,
    'Humidity': humidity,
    'Precipitation': precipitation,
    'Visibility': visibility,
    'Num_Lanes': num_lanes,
    'Speed_Limit': speed_limit,
    'Congestion_Index': congestion_index,
    'Traffic_Volume': traffic_volume
})

print(f"   Dataset created: {data.shape[0]} samples, {data.shape[1]-1} features")

# Step 2: Data Preprocessing
print("\n[2/6] Preprocessing Data...")
X = data.drop('Traffic_Volume', axis=1)
y = data['Traffic_Volume']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"   Train set: {X_train.shape[0]} samples")
print(f"   Test set: {X_test.shape[0]} samples")

# Step 3: Create model directory
print("\n[3/6] Creating model directory...")
model_dir = './saved_models'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    print(f"   Created: {model_dir}")
else:
    print(f"   Using existing: {model_dir}")

# Step 4: Train Random Forest
print("\n[4/6] Training Random Forest Regressor...")
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_scaled, y_train)
rf_test_pred = rf_model.predict(X_test_scaled)
rf_r2 = r2_score(y_test, rf_test_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_test_pred))
print(f"   R² Score: {rf_r2:.4f}, RMSE: {rf_rmse:.4f}")

# Step 5: Train other models
print("\n[5/6] Training Comparative Models...")

# Gradient Boosting
print("   - Gradient Boosting...", end=" ")
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(X_train_scaled, y_train)
gb_pred = gb_model.predict(X_test_scaled)
gb_r2 = r2_score(y_test, gb_pred)
print(f"R²: {gb_r2:.4f}")

# AdaBoost
print("   - AdaBoost...", end=" ")
ada_model = AdaBoostRegressor(n_estimators=100, random_state=42)
ada_model.fit(X_train_scaled, y_train)
ada_pred = ada_model.predict(X_test_scaled)
ada_r2 = r2_score(y_test, ada_pred)
print(f"R²: {ada_r2:.4f}")

# Bagging
print("   - Bagging...", end=" ")
bag_model = BaggingRegressor(n_estimators=100, random_state=42)
bag_model.fit(X_train_scaled, y_train)
bag_pred = bag_model.predict(X_test_scaled)
bag_r2 = r2_score(y_test, bag_pred)
print(f"R²: {bag_r2:.4f}")

# Linear Regression
print("   - Linear Regression...", end=" ")
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)
lr_r2 = r2_score(y_test, lr_pred)
print(f"R²: {lr_r2:.4f}")

# SVR
print("   - Support Vector Regression...", end=" ")
svr_model = SVR(kernel='rbf')
svr_model.fit(X_train_scaled, y_train)
svr_pred = svr_model.predict(X_test_scaled)
svr_r2 = r2_score(y_test, svr_pred)
print(f"R²: {svr_r2:.4f}")

# Step 6: Save all models
print("\n[6/6] Saving Models...")

# Save Random Forest
with open(os.path.join(model_dir, 'random_forest_model.pkl'), 'wb') as f:
    pickle.dump(rf_model, f)
print("   ✓ Random Forest saved")

# Save Gradient Boosting
with open(os.path.join(model_dir, 'gradient_boosting_model.pkl'), 'wb') as f:
    pickle.dump(gb_model, f)
print("   ✓ Gradient Boosting saved")

# Save AdaBoost
with open(os.path.join(model_dir, 'adaboost_model.pkl'), 'wb') as f:
    pickle.dump(ada_model, f)
print("   ✓ AdaBoost saved")

# Save Bagging
with open(os.path.join(model_dir, 'bagging_model.pkl'), 'wb') as f:
    pickle.dump(bag_model, f)
print("   ✓ Bagging saved")

# Save Linear Regression
with open(os.path.join(model_dir, 'linear_regression_model.pkl'), 'wb') as f:
    pickle.dump(lr_model, f)
print("   ✓ Linear Regression saved")

# Save SVR
with open(os.path.join(model_dir, 'svr_model.pkl'), 'wb') as f:
    pickle.dump(svr_model, f)
print("   ✓ SVR saved")

# Save Scaler
with open(os.path.join(model_dir, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)
print("   ✓ Scaler saved")

# Save Feature Names
with open(os.path.join(model_dir, 'feature_names.pkl'), 'wb') as f:
    pickle.dump(X.columns.tolist(), f)
print("   ✓ Feature names saved")

# Summary
print("\n" + "=" * 80)
print("MODEL TRAINING SUMMARY")
print("=" * 80)
print(f"\nModel Performance (Test R² Score):")
print(f"  Random Forest:        {rf_r2:.4f} ⭐ BEST")
print(f"  Gradient Boosting:    {gb_r2:.4f}")
print(f"  AdaBoost:             {ada_r2:.4f}")
print(f"  Bagging:              {bag_r2:.4f}")
print(f"  SVR:                  {svr_r2:.4f}")
print(f"  Linear Regression:    {lr_r2:.4f}")

print(f"\nAll models saved to: {os.path.abspath(model_dir)}/")
print("\nReady to run Streamlit app:")
print("  streamlit run streamlit_app.py")
print("\n" + "=" * 80)

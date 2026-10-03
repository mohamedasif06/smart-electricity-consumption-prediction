
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor
import joblib


# --------------------------------------------------
# 1. Load processed dataset
# --------------------------------------------------

file_path = "../data/processed/electricity_features.csv"

df = pd.read_csv(file_path)

df["Datetime"] = pd.to_datetime(df["Datetime"])

df = df.sort_values("Datetime")

print("Dataset shape:", df.shape)


# --------------------------------------------------
# 2. Create target variable
# --------------------------------------------------

# Predict the next hour's electricity consumption

df["target"] = df["Global_active_power"].shift(-1)

df = df.dropna()


# --------------------------------------------------
# 3. Select features
# --------------------------------------------------

features = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
    "hour",
    "dayofweek",
    "month",
    "day",
    "year",
    "is_weekend",
    "lag_1h"
]

X = df[features]

y = df["target"]


# --------------------------------------------------
# 4. Time-based train-test split
# --------------------------------------------------

split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# --------------------------------------------------
# 5. Train Random Forest model
# --------------------------------------------------

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

print("\nTraining model...")

model.fit(X_train, y_train)


# --------------------------------------------------
# 6. Make predictions
# --------------------------------------------------

y_pred = model.predict(X_test)


# --------------------------------------------------
# 7. Evaluate model
# --------------------------------------------------

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

r2 = r2_score(y_test, y_pred)

print("\nModel Performance")
print("------------------------")
print("MAE :", mae)
print("RMSE:", rmse)
print("R²  :", r2)


# --------------------------------------------------
# 8. Save model
# --------------------------------------------------

joblib.dump(model, "../models/electricity_model.pkl")

print("\nModel saved successfully!")

# --------------------------------------------------
# 9. Train XGBoost model
# --------------------------------------------------

xgb_model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    n_jobs=-1
)

print("\nTraining XGBoost...")

xgb_model.fit(X_train, y_train)


# --------------------------------------------------
# 10. Make XGBoost predictions
# --------------------------------------------------

y_pred_xgb = xgb_model.predict(X_test)


# --------------------------------------------------
# 11. Evaluate XGBoost
# --------------------------------------------------

mae_xgb = mean_absolute_error(y_test, y_pred_xgb)

rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))

r2_xgb = r2_score(y_test, y_pred_xgb)

print("\nXGBoost Performance")
print("------------------------")
print("MAE :", mae_xgb)
print("RMSE:", rmse_xgb)
print("R²  :", r2_xgb)


# --------------------------------------------------
# 12. Save XGBoost model
# --------------------------------------------------

joblib.dump(xgb_model, "../models/electricity_xgboost_model.pkl")

print("\nXGBoost model saved successfully!")

import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor


# --------------------------------------------------
# 1. Load processed dataset
# --------------------------------------------------

file_path = "../data/processed/electricity_features.csv"

df = pd.read_csv(file_path)

df["Datetime"] = pd.to_datetime(df["Datetime"])

df = df.sort_values("Datetime")


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

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# --------------------------------------------------
# 5. Define models
# --------------------------------------------------

models = {
    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )
}


# --------------------------------------------------
# 6. Train and evaluate models
# --------------------------------------------------

results = []

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    r2 = r2_score(y_test, y_pred)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })


# --------------------------------------------------
# 7. Display comparison
# --------------------------------------------------

results_df = pd.DataFrame(results)

print("\nModel Comparison")
print("------------------------")

print(results_df.to_string(index=False))

import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


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


# --------------------------------------------------
# 5. Load trained model
# --------------------------------------------------

model = joblib.load("../models/electricity_model.pkl")

print("Model loaded successfully!")


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
# 8. Create actual vs predicted DataFrame
# --------------------------------------------------

results = pd.DataFrame({
    "Datetime": df.iloc[split_index:]["Datetime"].values,
    "Actual": y_test.values,
    "Predicted": y_pred
})

print("\nActual vs Predicted:")
print(results.head(10))


# --------------------------------------------------
# 9. Plot Actual vs Predicted
# --------------------------------------------------

plt.figure(figsize=(14, 5))

plt.plot(
    results["Datetime"].iloc[:300],
    results["Actual"].iloc[:300],
    label="Actual"
)

plt.plot(
    results["Datetime"].iloc[:300],
    results["Predicted"].iloc[:300],
    label="Predicted"
)

plt.title("Actual vs Predicted Electricity Consumption")

plt.xlabel("Datetime")

plt.ylabel("Global Active Power (kW)")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()


# --------------------------------------------------
# 10. Feature importance
# --------------------------------------------------

importance = pd.Series(
    model.feature_importances_,
    index=features
).sort_values(ascending=False)

print("\nFeature Importance:")
print(importance)

plt.figure(figsize=(10, 6))

importance.plot(kind="bar")

plt.title("Feature Importance - Random Forest")

plt.xlabel("Features")

plt.ylabel("Importance")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()
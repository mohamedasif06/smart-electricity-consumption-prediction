import pandas as pd
from pathlib import Path

# --------------------------------------------------
# 1. Define project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

input_path = BASE_DIR / "data" / "raw" / "household_power_consumption.txt"
output_path = BASE_DIR / "data" / "processed" / "electricity_features.csv"


# --------------------------------------------------
# 2. Load dataset
# --------------------------------------------------

df = pd.read_csv(
    input_path,
    sep=";",
    na_values="?",
    low_memory=False
)

print("Original shape:", df.shape)


# --------------------------------------------------
# 3. Create Datetime column
# --------------------------------------------------

df["Datetime"] = pd.to_datetime(
    df["Date"] + " " + df["Time"],
    dayfirst=True
)

df = df.sort_values("Datetime")


# --------------------------------------------------
# 4. Convert numerical columns
# --------------------------------------------------

numeric_columns = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")


# --------------------------------------------------
# 5. Handle missing values
# --------------------------------------------------

df[numeric_columns] = df[numeric_columns].interpolate(
    method="linear",
    limit=60
)

# Remove remaining missing values
df = df.dropna(subset=numeric_columns)

print("Shape after missing-value handling:", df.shape)


# --------------------------------------------------
# 6. Set Datetime as index
# --------------------------------------------------

df = df.set_index("Datetime")


# --------------------------------------------------
# 7. Resample to hourly data
# --------------------------------------------------

hourly_df = df.resample("h").agg({
    "Global_active_power": "mean",
    "Global_reactive_power": "mean",
    "Voltage": "mean",
    "Global_intensity": "mean",
    "Sub_metering_1": "mean",
    "Sub_metering_2": "mean",
    "Sub_metering_3": "mean"
})

print("\nHourly dataset shape:", hourly_df.shape)


# --------------------------------------------------
# 8. Create time-based features
# --------------------------------------------------

hourly_df["hour"] = hourly_df.index.hour

hourly_df["dayofweek"] = hourly_df.index.dayofweek

hourly_df["month"] = hourly_df.index.month

hourly_df["day"] = hourly_df.index.day

hourly_df["year"] = hourly_df.index.year

hourly_df["is_weekend"] = (
    hourly_df["dayofweek"] >= 5
).astype(int)


# --------------------------------------------------
# 9. Create lag features
# --------------------------------------------------

# Previous hour's consumption
hourly_df["lag_1h"] = (
    hourly_df["Global_active_power"].shift(1)
)

# Consumption 24 hours ago
hourly_df["lag_24h"] = (
    hourly_df["Global_active_power"].shift(24)
)

# Consumption 7 days ago
hourly_df["lag_7d"] = (
    hourly_df["Global_active_power"].shift(24 * 7)
)


# --------------------------------------------------
# 10. Create rolling features
# --------------------------------------------------

hourly_df["rolling_mean_24h"] = (
    hourly_df["Global_active_power"]
    .shift(1)
    .rolling(window=24)
    .mean()
)


# --------------------------------------------------
# 11. Create target variable
# --------------------------------------------------

# Predict the NEXT hour's consumption
hourly_df["target_next_hour"] = (
    hourly_df["Global_active_power"].shift(-1)
)


# --------------------------------------------------
# 12. Remove rows created by lag/target operations
# --------------------------------------------------

hourly_df = hourly_df.dropna()


# --------------------------------------------------
# 13. Save processed dataset
# --------------------------------------------------

output_path.parent.mkdir(parents=True, exist_ok=True)

hourly_df.to_csv(output_path)

print("\nProcessed dataset saved successfully!")
print("Output path:", output_path)

print("\nFinal shape:", hourly_df.shape)

print("\nFirst five rows:")
print(hourly_df.head())


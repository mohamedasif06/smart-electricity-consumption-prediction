import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

file_path = "../data/raw/household_power_consumption.txt"

df = pd.read_csv(
    file_path,
    sep=";",
    na_values="?",
    low_memory=False
)

print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

# Dataset shape: (number of rows, number of columns)
print("\nDataset shape:")
print(df.shape)

print("\nDataset information:")
df.info()

print("\nMissing values:")
print(df.isnull().sum())

print("\nDescriptive statistics:")
print(df.describe())

# Convert Date and Time into a single timestamp
df["Datetime"] = pd.to_datetime(
    df["Date"] + " " + df["Time"],
    dayfirst=True
)

print("\nDatetime column:")
print(df[["Date", "Time", "Datetime"]].head())

# Numerical columns
numeric_columns = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3"
]

# Convert columns to numeric
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

print("\nData types after conversion:")
print(df.dtypes)

# Sort data chronologically
df = df.sort_values("Datetime")

print("\nFirst five rows after sorting:")
print(df[["Datetime", "Global_active_power"]].head())

# Check missing values after conversion
print("\nMissing values after conversion:")
print(df.isnull().sum())

# Display rows containing missing values
print("\nRows with missing values:")
print(df[df["Global_active_power"].isnull()].head())

# Count rows where Global_active_power is missing
missing_power = df["Global_active_power"].isnull()

print("\nTotal missing Global_active_power readings:")
print(missing_power.sum())

# Check consecutive missing readings
missing_mask = df["Global_active_power"].isnull()

# Create groups of consecutive missing/non-missing values
groups = missing_mask.ne(missing_mask.shift()).cumsum()

# Count consecutive missing values
missing_streaks = (
    df[missing_mask]
    .groupby(groups)
    .size()
)

print("\nLongest missing streak:")
print(missing_streaks.max())

print("\nMissing streak distribution:")
print(missing_streaks.describe())

# Identify consecutive missing periods
missing_mask = df["Global_active_power"].isnull()

groups = missing_mask.ne(missing_mask.shift()).cumsum()

missing_periods = (
    df[missing_mask]
    .groupby(groups)
    .agg(
        start_time=("Datetime", "min"),
        end_time=("Datetime", "max"),
        missing_count=("Datetime", "size")
    )
    .sort_values("missing_count", ascending=False)
)

print("\nTop 10 longest missing periods:")
print(missing_periods.head(10))

# Interpolate only short missing gaps (up to 60 minutes)

# Create a copy of the original dataframe
df_clean = df.copy()

# Interpolate numerical columns
for col in numeric_columns:
    df_clean[col] = df_clean[col].interpolate(
        method="linear",
        limit=60,
        limit_area="inside"
    )

print("\nMissing values after short-gap interpolation:")
print(df_clean[numeric_columns].isnull().sum())

# Create a daily average electricity consumption dataset
daily_consumption = (
    df_clean
    .set_index("Datetime")["Global_active_power"]
    .resample("D")
    .mean()
)

print("\nDaily consumption:")
print(daily_consumption.head())

# Plot daily average electricity consumption
plt.figure(figsize=(14, 5))

plt.plot(
    daily_consumption.index,
    daily_consumption.values
)

plt.title("Daily Average Electricity Consumption")
plt.xlabel("Date")
plt.ylabel("Average Active Power (kW)")
plt.grid(True)
plt.tight_layout()

plt.show()

# Create time-based features
df_clean["Hour"] = df_clean["Datetime"].dt.hour
df_clean["DayOfWeek"] = df_clean["Datetime"].dt.dayofweek
df_clean["Month"] = df_clean["Datetime"].dt.month

# Calculate average consumption for each hour
hourly_consumption = (
    df_clean
    .groupby("Hour")["Global_active_power"]
    .mean()
)

print("\nAverage consumption by hour:")
print(hourly_consumption)

# Plot average electricity consumption by hour
plt.figure(figsize=(12, 5))

plt.plot(
    hourly_consumption.index,
    hourly_consumption.values,
    marker="o"
)

plt.title("Average Electricity Consumption by Hour")
plt.xlabel("Hour of the Day")
plt.ylabel("Average Active Power (kW)")
plt.xticks(range(24))
plt.grid(True)
plt.tight_layout()

plt.show()

# Create weekday/weekend classification
df_clean["DayType"] = df_clean["DayOfWeek"].apply(
    lambda x: "Weekend" if x >= 5 else "Weekday"
)

# Calculate average consumption by day type
daytype_consumption = (
    df_clean
    .groupby("DayType")["Global_active_power"]
    .mean()
)

print("\nAverage consumption by day type:")
print(daytype_consumption)

# Plot weekday vs weekend consumption
plt.figure(figsize=(7, 5))

plt.bar(
    daytype_consumption.index,
    daytype_consumption.values
)

plt.title("Average Electricity Consumption: Weekday vs Weekend")
plt.xlabel("Day Type")
plt.ylabel("Average Active Power (kW)")
plt.grid(axis="y")
plt.tight_layout()

plt.show()

# Calculate average electricity consumption by month
monthly_consumption = (
    df_clean
    .groupby("Month")["Global_active_power"]
    .mean()
)

print("\nAverage consumption by month:")
print(monthly_consumption)

# Plot average electricity consumption by month
plt.figure(figsize=(10, 5))

plt.plot(
    monthly_consumption.index,
    monthly_consumption.values,
    marker="o"
)

plt.title("Average Electricity Consumption by Month")
plt.xlabel("Month")
plt.ylabel("Average Active Power (kW)")
plt.xticks(range(1, 13))
plt.grid(True)
plt.tight_layout()

plt.show()

# Select valid Global_active_power readings
power_values = df_clean["Global_active_power"].dropna()

# Plot distribution
plt.figure(figsize=(10, 5))

plt.hist(
    power_values,
    bins=50
)

plt.title("Distribution of Global Active Power")
plt.xlabel("Global Active Power (kW)")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()

plt.show()

# Plot boxplot
plt.figure(figsize=(8, 4))

plt.boxplot(power_values)

plt.title("Boxplot of Global Active Power")
plt.ylabel("Global Active Power (kW)")
plt.grid(True)
plt.tight_layout()

plt.show()

# Calculate skewness
print("\nSkewness of Global_active_power:")
print(power_values.skew())

# Calculate correlation matrix
correlation_matrix = df_clean[numeric_columns].corr()

print("\nCorrelation matrix:")
print(correlation_matrix)

# Plot correlation heatmap
plt.figure(figsize=(10, 7))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap of Electricity Variables")
plt.tight_layout()

plt.show()
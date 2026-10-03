"""Flask dashboard for next-hour household electricity-demand predictions."""

import logging
import math
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "electricity_xgboost_model.pkl"
DATA_PATH = BASE_DIR / "data" / "processed" / "electricity_features.csv"
FEATURES = [
    "Global_active_power", "Global_reactive_power", "Voltage", "Global_intensity",
    "Sub_metering_1", "Sub_metering_2", "Sub_metering_3", "hour", "dayofweek",
    "month", "day", "year", "is_weekend", "lag_1h",
]
INTEGER_FEATURES = {"hour", "dayofweek", "month", "day", "year", "is_weekend"}
MEASUREMENT_FEATURES = set(FEATURES) - INTEGER_FEATURES
ELECTRICITY_RATE_PER_KWH = 6.00  # Estimated local tariff; change for your location.
TEST_RMSE = 0.464619
# Configurable energy-awareness thresholds for predicted next-hour power in kW.
LOW_CONSUMPTION_THRESHOLD = 1.0
HIGH_CONSUMPTION_THRESHOLD = 2.0

app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024  # CSV uploads only; 2 MB is ample for sample rows.
app.logger.setLevel(logging.INFO)
model, defaults, feature_importance, resource_error = None, {}, [], None


def load_resources():
    """Load the model/data once at startup and derive display values from them."""
    global model, defaults, feature_importance, resource_error
    try:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
        if not DATA_PATH.exists():
            raise FileNotFoundError(f"Processed dataset not found: {DATA_PATH}")
        dataset = pd.read_csv(DATA_PATH)
        missing = set(FEATURES) - set(dataset.columns)
        if missing:
            raise ValueError(f"Processed data lacks required columns: {sorted(missing)}")
        if dataset.empty:
            raise ValueError("Processed dataset has no rows.")
        latest_row = dataset.iloc[-1]
        # Convert NumPy scalar values so Jinja can safely serialize defaults for JavaScript.
        defaults = {
            name: latest_row[name].item() if hasattr(latest_row[name], "item") else latest_row[name]
            for name in FEATURES
        }
        model = joblib.load(MODEL_PATH)
        importances = getattr(model, "feature_importances_", None)
        if importances is None or len(importances) != len(FEATURES):
            raise ValueError("Saved model has no importance values for the 14 expected features.")
        total = float(sum(importances))
        feature_importance = [
            {"name": name, "percentage": float(value) / total * 100 if total else 0}
            for name, value in sorted(zip(FEATURES, importances), key=lambda item: item[1], reverse=True)
        ]
        app.logger.info("Model and processed data loaded successfully.")
    except Exception:
        app.logger.exception("Unable to initialize prediction resources")
        resource_error = "The prediction service could not be initialized. Check terminal logs and model/data files."


def validate_form(form):
    """Convert form data to numeric model values and collect readable errors."""
    values, errors = {}, []
    for name in FEATURES:
        raw = form.get(name, "").strip()
        if not raw:
            errors.append(f"{name.replace('_', ' ')} is required.")
            continue
        try:
            numeric_value = float(raw)
            if name in INTEGER_FEATURES and not numeric_value.is_integer():
                raise ValueError
            value = int(numeric_value) if name in INTEGER_FEATURES else numeric_value
        except ValueError:
            errors.append(f"{name.replace('_', ' ')} must be numeric.")
            continue
        if not math.isfinite(value):
            errors.append(f"{name.replace('_', ' ')} must be a finite number.")
        elif name in MEASUREMENT_FEATURES and value < 0:
            errors.append(f"{name.replace('_', ' ')} cannot be negative.")
        values[name] = value
    for name, low, high in [("hour", 0, 23), ("dayofweek", 0, 6), ("month", 1, 12), ("day", 1, 31), ("is_weekend", 0, 1)]:
        if name in values and not low <= values[name] <= high:
            errors.append(f"{name.replace('_', ' ')} must be between {low} and {high}.")
    return values, errors


def classify_consumption(prediction):
    """Describe an estimated demand level using configurable, neutral thresholds."""
    if prediction < LOW_CONSUMPTION_THRESHOLD:
        return {
            "category": "Low consumption",
            "remark": "Predicted electricity demand is relatively low for the selected conditions.",
            "suggestion": "Continue monitoring unnecessary standby devices and switch off appliances that are not being used.",
            "level": "low",
        }
    if prediction < HIGH_CONSUMPTION_THRESHOLD:
        return {
            "category": "Moderate consumption",
            "remark": "Predicted electricity demand is moderate. Review the active appliances if you want to reduce unnecessary usage.",
            "suggestion": "Consider shifting non-essential appliance use away from other active loads where practical.",
            "level": "moderate",
        }
    return {
        "category": "High consumption",
        "remark": "Predicted electricity demand is relatively high based on the submitted inputs.",
        "suggestion": "Check whether high-power appliances such as air conditioners, heaters, washing machines, or other heavy loads are operating.",
        "level": "high",
    }


def json_values(row):
    """Return a validated, JSON-safe feature dictionary or None for an invalid row."""
    raw_values = {name: "" if pd.isna(row.get(name)) else str(row.get(name)) for name in FEATURES}
    values, errors = validate_form(raw_values)
    if errors:
        return None
    return values


def sample_response(frame, source):
    """Extract up to 20 valid model-ready rows for the form's row picker."""
    samples = []
    for index, row in frame.iterrows():
        values = json_values(row)
        if values is not None:
            samples.append({"row_number": int(index) + 1, "values": values})
        if len(samples) == 20:
            break
    if not samples:
        raise ValueError("No valid row with all 14 required model features was found.")
    return {"success": True, "source": source, "rows_detected": int(len(frame)), "samples": samples}


def prepare_uploaded_frame(uploaded):
    """Map known column aliases and safely derive time features from original data."""
    frame = pd.read_csv(uploaded.stream)
    if frame.empty:
        raise ValueError("The uploaded CSV is empty.")

    aliases = {
        "global_active_power": "Global_active_power", "global_reactive_power": "Global_reactive_power",
        "voltage": "Voltage", "global_intensity": "Global_intensity",
        "sub_metering_1": "Sub_metering_1", "sub_metering_2": "Sub_metering_2",
        "sub_metering_3": "Sub_metering_3", "day_of_week": "dayofweek",
        "weekend_flag": "is_weekend", "lag_1": "lag_1h",
    }
    rename_map = {column: aliases[column.lower()] for column in frame.columns if column.lower() in aliases}
    frame = frame.rename(columns=rename_map)

    # Raw UCI-style data can derive calendar fields from Date + Time or Datetime.
    timestamp = None
    if "Datetime" in frame.columns:
        timestamp = pd.to_datetime(frame["Datetime"], errors="coerce")
    elif {"Date", "Time"}.issubset(frame.columns):
        timestamp = pd.to_datetime(frame["Date"].astype(str) + " " + frame["Time"].astype(str), dayfirst=True, errors="coerce")
    if timestamp is not None:
        derived = {"hour": timestamp.dt.hour, "dayofweek": timestamp.dt.dayofweek,
                   "month": timestamp.dt.month, "day": timestamp.dt.day,
                   "year": timestamp.dt.year}
        for name, values in derived.items():
            if name not in frame.columns:
                frame[name] = values
        if "is_weekend" not in frame.columns:
            frame["is_weekend"] = (frame["dayofweek"] >= 5).astype("Int64")
    if "lag_1h" not in frame.columns and "Global_active_power" in frame.columns:
        frame["lag_1h"] = pd.to_numeric(frame["Global_active_power"], errors="coerce").shift(1)

    missing = [name for name in FEATURES if name not in frame.columns]
    if missing:
        raise ValueError("CSV is missing required features: " + ", ".join(missing) + ".")
    return frame


def page_context(**overrides):
    context = {"features": FEATURES, "defaults": defaults, "form_values": defaults,
               "feature_importance": feature_importance, "prediction": None, "error": None,
               "success_message": None, "approximate_lower": None, "approximate_upper": None,
               "demand_status": None, "status_level": None, "remark": None,
               "suggestion": None, "estimated_cost": None,
               "electricity_rate": ELECTRICITY_RATE_PER_KWH, "resource_error": resource_error}
    context.update(overrides)
    return context


@app.get("/")
def index():
    return render_template("index.html", **page_context())


@app.post("/load-sample")
def load_sample():
    """Return the latest valid processed-data row for the Load sample data button."""
    try:
        if not DATA_PATH.exists():
            raise FileNotFoundError("Processed sample CSV is unavailable.")
        frame = pd.read_csv(DATA_PATH)
        missing = [name for name in FEATURES if name not in frame.columns]
        if missing:
            raise ValueError("Processed CSV is missing required features: " + ", ".join(missing))
        # Reverse first: latest valid row is selected before returning to chronological order.
        response = sample_response(frame.iloc[::-1].reset_index(drop=True), "Latest processed dataset row")
        response["samples"] = response["samples"][:1]
        response["samples"][0]["row_number"] = len(frame)
        return jsonify(response)
    except Exception as error:
        app.logger.warning("Could not load sample data: %s", error)
        return jsonify({"success": False, "error": str(error)}), 400


@app.post("/upload-sample")
def upload_sample():
    """Read a CSV in memory and return validated candidate rows; never save the file."""
    uploaded = request.files.get("sample_file")
    if uploaded is None or not uploaded.filename:
        return jsonify({"success": False, "error": "Choose a CSV file to upload."}), 400
    filename = secure_filename(uploaded.filename)
    if not filename.lower().endswith(".csv"):
        return jsonify({"success": False, "error": "Only .csv files are allowed."}), 400
    try:
        response = sample_response(prepare_uploaded_frame(uploaded), filename)
        return jsonify(response)
    except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as error:
        app.logger.warning("Invalid uploaded CSV %s: %s", filename, error)
        return jsonify({"success": False, "error": f"Could not use this CSV: {error}"}), 400
    except Exception:
        app.logger.exception("Unexpected CSV upload error")
        return jsonify({"success": False, "error": "Could not read the uploaded CSV."}), 400


@app.errorhandler(413)
def file_too_large(_error):
    return jsonify({"success": False, "error": "CSV file is too large. Maximum size is 2 MB."}), 413


@app.post("/predict")
def predict():
    form_values = {name: request.form.get(name, "") for name in FEATURES}
    if resource_error or model is None:
        return render_template("index.html", **page_context(form_values=form_values, error=resource_error)), 503
    values, errors = validate_form(request.form)
    if errors:
        return render_template("index.html", **page_context(form_values=form_values, error=" ".join(errors))), 400
    try:
        input_frame = pd.DataFrame([[values[name] for name in FEATURES]], columns=FEATURES)
        prediction = float(model.predict(input_frame)[0])
        if not math.isfinite(prediction) or prediction < 0:
            app.logger.error("Model returned invalid prediction: %r", prediction)
            return render_template("index.html", **page_context(
                form_values=values,
                error="The model returned an invalid negative or non-finite estimate. No prediction was displayed.")), 500
        lower, upper = max(0.0, prediction - TEST_RMSE), prediction + TEST_RMSE
        classification = classify_consumption(prediction)
        return render_template("index.html", **page_context(
            form_values=values, prediction=prediction, approximate_lower=lower,
            approximate_upper=upper, demand_status=classification["category"],
            status_level=classification["level"], remark=classification["remark"],
            suggestion=classification["suggestion"],
            estimated_cost=prediction * ELECTRICITY_RATE_PER_KWH,
            success_message="Prediction generated with the saved XGBoost model."))
    except Exception:
        app.logger.exception("Prediction failed")
        return render_template("index.html", **page_context(form_values=form_values, error="Prediction failed. Please check your values and try again.")), 500


load_resources()

if __name__ == "__main__":
    app.run(debug=True)

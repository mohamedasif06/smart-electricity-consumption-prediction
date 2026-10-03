import joblib

# Load the saved XGBoost model
model = joblib.load("../models/electricity_xgboost_model.pkl")

print("XGBoost model loaded successfully!")

print("\nModel type:")
print(type(model))

print("\nNumber of features:")
print(model.n_features_in_)
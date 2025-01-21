import os
import pandas as pd
import pickle

DATA_PATH = "./data/uploaded_data.csv"
MODEL_PATH = "./saved_models/model.pkl"

def save_csv(file):
    """Save uploaded CSV file."""
    try:
        data = pd.read_csv(file)
        data.to_csv(DATA_PATH, index=False)
        return True
    except Exception as e:
        print(f"Error saving file: {e}")
        return False

def validate_dataset():
    """Validate if dataset is correct."""
    if not os.path.exists(DATA_PATH):
        return False, "No dataset found. Upload the data first."
    data = pd.read_csv(DATA_PATH)
    required_columns = {"Temperature", "Run_Time", "Downtime_Flag"}
    if not required_columns.issubset(data.columns):
        return False, "Dataset must contain 'Temperature', 'Run_Time', and 'Downtime_Flag'."
    return True, "Dataset is valid."

def load_model():
    """Load the trained model."""
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

def predict_input(model, input_data):
    """Predict input data using the model."""
    try:
        features = [[input_data["Temperature"], input_data["Run_Time"]]]
        prediction = model.predict(features)
        confidence = model.predict_proba(features).max()
        return {"Downtime": "Yes" if prediction[0] == 1 else "No", "Confidence": confidence}
    except Exception as e:
        return {"error": str(e)}

import os
import pandas as pd
from flask import request, jsonify
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import joblib

DATA_PATH = "data/sample_data.csv"
MODEL_PATH = "saved_models/downtime_model.pkl"

def preprocess_data():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    os.makedirs("data", exist_ok=True)
    file.save(DATA_PATH)
    return jsonify({"message": "File uploaded successfully"}), 200


def train_model():
    if not os.path.exists(DATA_PATH):
        return jsonify({"error": "No dataset found. Please upload data first."}), 400

    data = pd.read_csv(DATA_PATH)
    if 'Downtime_Flag' not in data.columns:
        return jsonify({"error": "Dataset must include 'Downtime_Flag' column."}), 400

    data['Downtime_Flag'] = LabelEncoder().fit_transform(data['Downtime_Flag'])
    X = data[['Temperature', 'Run_Time']]
    y = data['Downtime_Flag']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    os.makedirs("saved_models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    return jsonify({"accuracy": accuracy, "f1_score": f1}), 200


def make_prediction():
    if not os.path.exists(MODEL_PATH):
        return jsonify({"error": "Model not trained yet. Please train the model first."}), 400

    model = joblib.load(MODEL_PATH)
    data = request.get_json()

    if not data or 'Temperature' not in data or 'Run_Time' not in data:
        return jsonify({"error": "Invalid input. Provide 'Temperature' and 'Run_Time'."}), 400

    X_new = pd.DataFrame([data])
    prediction = model.predict(X_new)
    confidence = model.predict_proba(X_new).max()

    return jsonify({
        "Downtime": "Yes" if prediction[0] == 1 else "No",
        "Confidence": round(confidence, 2)
    }), 200

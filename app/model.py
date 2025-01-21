import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import pickle

DATA_PATH = "./data/uploaded_data.csv"
MODEL_PATH = "./saved_models/model.pkl"

def train_model():
    """Train the machine learning model."""
    data = pd.read_csv(DATA_PATH)
    X = data[["Temperature", "Run_Time"]]
    y = data["Downtime_Flag"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Save model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    return {"accuracy": accuracy, "f1_score": f1}

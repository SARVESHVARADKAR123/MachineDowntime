from fastapi import FastAPI, UploadFile, File
from app.utils import save_csv, validate_dataset, load_model, predict_input

from app.model import train_model

app = FastAPI()

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload CSV file endpoint."""
    if file.content_type != "text/csv":
        return {"error": "Only CSV files are supported."}
    success = save_csv(file.file)
    if not success:
        return {"error": "Failed to save file."}
    return {"message": "File uploaded successfully."}

@app.post("/train")
def train():
    """Train model endpoint."""
    valid, message = validate_dataset()
    if not valid:
        return {"error": message}
    metrics = train_model()
    return metrics

@app.post("/predict")
def predict(input_data: dict):
    """Predict endpoint."""
    model = load_model()
    if not model:
        return {"error": "No trained model found. Please train the model first."}
    result = predict_input(model, input_data)
    return result

from fastapi import FastAPI, UploadFile, File
from inference import predict

app = FastAPI(title="Deepfake Image Detector")

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    result = predict(file)
    return result

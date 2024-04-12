from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import numpy as np
from PIL import Image
import joblib
import pickle
from tensorflow.keras.models import load_model
import os
import io
import pathlib

BASE_DIR = pathlib.Path(__file__).parent

app = FastAPI()


@app.post('/predict')
async def predict(img_file:UploadFile = File(...)):
    bytes_str = io.BytesIO(await img_file.read())
    try:
        img = Image.open(bytes_str)
    except:
        raise HTTPException(detail="Invalid image", status_code=400)

    # load and preprocces the image
    img = np.array([img.resize((30,30))])

    # load the model
    try:
        model = load_model('traffic_classifier.h5a')
    except:
        raise HTTPException(detail="Invalid model", status_code=400)

    # prediction
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)
    
    return {"class": str(predicted_class)}

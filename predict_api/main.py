from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import io
import pathlib
from .functions import *

BASE_DIR = pathlib.Path(__file__).parent

app = FastAPI()

@app.get('/')
def home():
    return {'BASE_DIR': BASE_DIR}


@app.post('/predict')
async def predict(img_file:UploadFile = File(...)):
    bytes_str = io.BytesIO(await img_file.read())
    try:
        img = Image.open(bytes_str)
    except:
        raise HTTPException(detail="Invalid image", status_code=400)

    # load and preprocess the image
    img = np.array([img.resize((30,30))])

    # load the model
    try:
        model = load_model(BASE_DIR.parent / 'traffic_classifier.h5')
    except:
        raise HTTPException(detail="Invalid model", status_code=400)

    # prediction
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)

    return {"sign": sign_label(predicted_class)}
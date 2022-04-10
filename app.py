from statistics import mode
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile
from fastapi.params import File, Form
from starlette.responses import JSONResponse
import base64
import io
from PIL import Image
import torch
import pandas as pd
import numpy as np
import asyncio
import requests

app = FastAPI()

origins = ["*"]

# MIDDLERWARE
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load the model
model_path = "model/modelv1.pt"
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

# Convert image to numpy array
def transform_image(image):
    image_data = np.asarray(image)
    return image_data

def predict(image):
    results = mode(image)
    data_frame = pd.DataFrame()
    temp_data_frame = results.pandas().xyxy[0]

    for index, result in enumerate(temp_data_frame['confidence']):
        if result < 0.6:
            data_frame = temp_data_frame.drop(index)
        else:
            data_frame = temp_data_frame
    return data_frame



@app.post("/")
async def main(image_file: UploadFile = File(...)):
    try:
        image_bytes = await image_file.read()
        # base64_image = base64.b64encode(image_bytes).decode("utf8")
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        transformed_image = transform_image(image)
        prediction = prediction(transformed_image)
    except:
        pass
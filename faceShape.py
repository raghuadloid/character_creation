from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow
import cv2
from tensorflow.keras.applications.inception_v3 import preprocess_input

app = FastAPI()

model = tensorflow.keras.models.load_model("face_shape_model.h5")
@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def preprocess_image(data) -> np.ndarray:
    img = np.array(Image.open(BytesIO(data)))
    img = preprocess_input((cv2.resize(img, (224,224))))
    return img

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    #0-woman 1-man
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = preprocess_image(await file.read())
    # predictions = preprocess_image(await file.read())
    image_batch = np.expand_dims(image,0)
    classes = ['heart', 'oblong', 'oval', 'round', 'square']
    predictions = model.predict(image_batch)
    name = classes[predictions[0].argmax()]

    return {
        "pred ": name
    }

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8080)
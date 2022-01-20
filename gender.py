from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow
import cv2
from tensorflow.keras.applications.inception_v3 import preprocess_input

app = FastAPI()

model = tensorflow.keras.models.load_model("gender_model.h5")
@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def preprocess_image(data) -> np.ndarray:
    img = np.array(Image.open(BytesIO(data)))
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = cascade.detectMultiScale(gray,1.1,7)
    buffer = 30
    ymin = max(0, faces[0][1]-buffer)
    ymax = min(img.shape[1], faces[0][1]+faces[0][3]+buffer)
    xmin = max(0, faces[0][0]-buffer)
    xmax = min(img.shape[0], faces[0][0]+faces[0][2]+buffer)
    face = img[ymin:ymax,xmin:xmax]
    face = cv2.resize(face,(224,224))
    img_scaled = preprocess_input(face)
    reshape = np.reshape(img_scaled,(1,224,224,3))
    image = np.vstack([reshape])
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img = preprocess_image(await file.read())
    predictions = model.predict(img)
    classes = ['female', 'male']
    result = classes[predictions[0].argmax()]
    
    return {
        "pred ": result
    }

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8080)
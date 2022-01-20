from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import cv2
import cvlib as cv

app = FastAPI()

# model = tensorflow.keras.models.load_model("face_shape_model.h5")
@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def preprocess_image(data) -> str:
    img = np.array(Image.open(BytesIO(data)))
    img = cv2.resize(img, (256,256))
    face, conf = cv.detect_face(img)
    padding = 20
    for f in face:
        (startX,startY) = max(0, f[0]-padding), max(0, f[1]-padding)
        (endX,endY) = min(img.shape[1]-1, f[2]+padding), min(img.shape[0]-1, f[3]+padding)
        face_crop = np.copy(img[startY:endY, startX:endX])
        (label, confidence) = cv.detect_gender(face_crop)
        classes = ['male', 'female']
        pred = classes[np.argmax(confidence)]
    return pred

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    #0-woman 1-man
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    predicted = preprocess_image(await file.read())

    return {
        "pred ": predicted
    }

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8080)
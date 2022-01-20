from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import cv2
from sklearn.cluster import KMeans
from collections import Counter

app = FastAPI()

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def palette_dict(k_cluster):
    n_pixels = len(k_cluster.labels_)
    counter = Counter(k_cluster.labels_) # count how many pixels per cluster
    perc = {}
    for i in counter:
        perc[i] = np.round(counter[i]/n_pixels, 2)
    perc = dict(sorted(perc.items(), key=lambda p: p[1], reverse=True))
    return [perc, (k_cluster.cluster_centers_)]

def weighted_mean_colors(c1, w1, c2, w2):
  amount = [w1, w2]
  d1 = np.average([c1[0], c2[0]], weights=amount)
  d2 = np.average([c1[1], c2[1]], weights=amount)
  d3 = np.average([c1[2], c2[2]], weights=amount)
  return [d1, d2, d3]


def preprocess_image(data):
    img = np.array(Image.open(BytesIO(data)))
    n_clusters = 4
    cut = 30
    img = cv2.resize(img, (512,512))
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1,6)
    crop_img = img[faces[0][1]:faces[0][1]+faces[0][3], faces[0][0]:faces[0][0]+faces[0][2]]
    crop_img = cv2.resize(crop_img, (256,256))
    crop_img = crop_img[cut:crop_img.shape[1]-cut, cut:crop_img.shape[0]-cut]
    clt = KMeans(n_clusters) 
    clt_1 = clt.fit(crop_img.reshape(-1,3)) 
    lis = palette_dict(clt_1)
    col1 = lis[1][list(lis[0])[0]]
    if col1[0] < 50 and col1[1] < 50 and col1[2] < 50:
        return str([140,95,65])
    col2 = lis[1][list(lis[0])[1]]
    if col2[0] < 50 and col2[1] < 50 and col2[2] < 50:
        return str([140,95,65])
    w1 = list(lis[0].values())[0]
    w2 = list(lis[0].values())[1]
    values = weighted_mean_colors(col1, w1, col2, w2)
    return str(values)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    colors = preprocess_image(await file.read())

    return {
        "red": colors
    }

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8080)
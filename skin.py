from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow
import cv2
import mediapipe as mp
from sklearn.cluster import KMeans
from statistics import median, mean
from collections import Counter
import colorsys

app = FastAPI()

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

# def palette_dict(k_cluster):
#     n_pixels = len(k_cluster.labels_)
#     counter = Counter(k_cluster.labels_) # count how many pixels per cluster
#     perc = {}
#     for i in counter:
#         perc[i] = np.round(counter[i]/n_pixels, 2)
#     perc = dict(sorted(perc.items()))
#     return [perc, (k_cluster.cluster_centers_)]

def test():
    mp_face_mesh = mp.solutions.face_mesh
    return mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

def preprocess_image(data):
    img = np.array(Image.open(BytesIO(data)))
    n_clusters = 3
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = test()

    results = face_mesh.process(img)
    annotated_image = img.copy()
    return annotated_image


    #     #Right Cheek
    #     x_points = []
    #     y_points = []
    #     for r in [348, 411, 358, 410]: 
    #         x = results.multi_face_landmarks[0].landmark[r].x
    #         y = results.multi_face_landmarks[0].landmark[r].y
    #         shape = img.shape 
    #         relative_x = int(x * shape[1])
    #         x_points.append(relative_x)
    #         relative_y = int(y * shape[0])
    #         y_points.append(relative_y)
    #     x_min = min(x_points)
    #     y_min = min(y_points)
    #     x_max = max(x_points)
    #     y_max = max(y_points)
    #     crop_img = annotated_image[y_min:y_max, x_min:x_max]
    #     clt = KMeans(n_clusters)
    #     clt_1 = clt.fit(crop_img.reshape(-1,3))
    #     n_pixels = len(clt_1.labels_)
    #     counter = Counter(clt_1.labels_)
    #     perc = {}
    #     for i in counter:
    #         perc[i] = np.round(counter[i]/n_pixels, 2)
    #     perc = dict(sorted(perc.items()))
    #     Keymax = max(zip(perc.values(), perc.keys()))[1]
    #     dom_color1 = (clt_1.cluster_centers_[Keymax]).tolist()
    #     dom_color1_weight = perc[Keymax]


    #     #Left Cheek
    #     x_points = []
    #     y_points = []
    #     for r in [119, 187, 129, 186]: 
    #         x = results.multi_face_landmarks[0].landmark[r].x
    #         y = results.multi_face_landmarks[0].landmark[r].y
    #         shape = img.shape 
    #         relative_x = int(x * shape[1])
    #         x_points.append(relative_x)
    #         relative_y = int(y * shape[0])
    #         y_points.append(relative_y)
    #     x_min = min(x_points)
    #     y_min = min(y_points)
    #     x_max = max(x_points)
    #     y_max = max(y_points)
    #     crop_img = annotated_image[y_min:y_max, x_min:x_max]
    #     clt = KMeans(n_clusters)
    #     clt_1 = clt.fit(crop_img.reshape(-1,3))
    #     n_pixels = len(clt_1.labels_)
    #     counter = Counter(clt_1.labels_)
    #     perc = {}
    #     for i in counter:
    #         perc[i] = np.round(counter[i]/n_pixels, 2)
    #     perc = dict(sorted(perc.items()))
    #     Keymax = max(zip(perc.values(), perc.keys()))[1]
    #     dom_color2 = (clt_1.cluster_centers_[Keymax]).tolist()
    #     dom_color2_weight = perc[Keymax]


    #     #Forehead
    #     x_points = []
    #     y_points = []
    #     for r in [107, 109, 336, 338]: 
    #         x = results.multi_face_landmarks[0].landmark[r].x
    #         y = results.multi_face_landmarks[0].landmark[r].y
    #         shape = img.shape 
    #         relative_x = int(x * shape[1])
    #         x_points.append(relative_x)
    #         relative_y = int(y * shape[0])
    #         y_points.append(relative_y)
    #     x_min = min(x_points)
    #     y_min = min(y_points)
    #     x_max = max(x_points)
    #     y_max = max(y_points)
    #     crop_img = annotated_image[y_min:y_max, x_min:x_max]
    #     clt = KMeans(n_clusters)
    #     clt_1 = clt.fit(crop_img.reshape(-1,3))
    #     n_pixels = len(clt_1.labels_)
    #     counter = Counter(clt_1.labels_)
    #     perc = {}
    #     for i in counter:
    #         perc[i] = np.round(counter[i]/n_pixels, 2)
    #     perc = dict(sorted(perc.items()))
    #     Keymax = max(zip(perc.values(), perc.keys()))[1]
    #     dom_color3 = (clt_1.cluster_centers_[Keymax]).tolist()
    #     dom_color3_weight = perc[Keymax]

    #     amounts = [dom_color1_weight, dom_color2_weight, 2*dom_color3_weight]
    #     d1 = np.average([dom_color1[0], dom_color2[0], dom_color3[0]], weights=amounts)
    #     d2 = np.average([dom_color1[1], dom_color2[1], dom_color3[1]], weights=amounts)
    #     d3 = np.average([dom_color1[2], dom_color2[2], dom_color3[2]], weights=amounts)
    # return np.array([d1, d2, d3])
    # return img

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    colors = preprocess_image(await file.read())
    # rt_str = str(colors)

    return {
        "red": colors.shape
    }

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8080)
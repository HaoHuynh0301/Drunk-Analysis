import pandas as pd
import cv2
import numpy as np

#define_functions
image_size=(48,48)
def load_data(dataset_path):
    data = pd.read_csv(dataset_path)
    pixels = data['pixels'].tolist()
    width, height = 48, 48
    faces = []
    for pixel_sequence in pixels:
        face = [int(pixel) for pixel in pixel_sequence.split(' ')]
        face = np.asarray(face).reshape(width, height)
        face = cv2.resize(face.astype('uint8'),image_size)
        faces.append(face.astype('float32'))
    faces = np.asarray(faces)
    faces = np.expand_dims(faces, -1)
    emotions = pd.get_dummies(data['emotion']).to_numpy()
    return faces, emotions

def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x

def read_data(drunkfile, normalfile):
    drunk = pd.read_csv(drunkfile)
    normal = pd.read_csv(normalfile)
    data = pd.concat([drunk, normal])
    ans_list = [1] * len(drunk) + [0] * len(normal)
    return np.array(data[['smile','anger','contempt','disgust','fear','happiness','neutral','sadness','surprise','eyeOccluded']].values), np.array(ans_list)

#!/usr/bin/python3

import cv2
import sys
import face_recognition
from pprint import pprint
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import keras
from keras.models import load_model

model = load_model("./emotions/model_v6_23.hdf5")
face_image = face_recognition.load_image_file("kevin.jpg")
face_image = cv2.resize(face_image, (48,48))
face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
face_image = np.reshape(face_image, [1, face_image.shape[0], face_image.shape[1], 1])
emotion_dict= {'Angry': 0, 'Sad': 5, 'Neutral': 4, 'Disgust': 1, 'Surprise': 6, 'Fear': 2, 'Happy': 3}

print(face_image.shape)
predicted_class = np.argmax(model.predict(face_image))
label_map = dict((v,k) for k,v in emotion_dict.items())
predicted_label = label_map[predicted_class]
print(predicted_label)
#face_image  = cv2.imread("./kevin.jpg")
#small_frame = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
#predicted_class = np.argmax(model.predict(image(1,28,28,3)))


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

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
model = load_model("./emotions/model_v6_23.hdf5")
emotion_dict= {'Angry': 0, 'Sad': 5, 'Neutral': 4, 'Disgust': 1, 'Surprise': 6, 'Fear': 2, 'Happy': 3}

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

#face_image = face_recognition.load_image_file("kevin.jpg")
#face_image = cv2.resize(face_image, (48,48))
#face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
#face_image = np.reshape(face_image, [1, face_image.shape[0], face_image.shape[1], 1])

#print(face_image.shape)
#predicted_class = np.argmax(model.predict(face_image))
#label_map = dict((v,k) for k,v in emotion_dict.items())
#predicted_label = label_map[predicted_class]
#print(predicted_label)

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    face_image = cv2.resize(frame, (48,48))
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    face_image = np.reshape(face_image, [1, face_image.shape[0], face_image.shape[1], 1])
    predicted_class = np.argmax(model.predict(face_image))
    label_map = dict((v,k) for k,v in emotion_dict.items())
    predicted_label = label_map[predicted_class]
    print(predicted_label)

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()

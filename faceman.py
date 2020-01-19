#!/usr/bin/env python3
import cv2
import sys
import face_recognition
from pprint import pprint
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import keras
from keras.models import load_model
import glob
import os

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
model = load_model("./emotions/model_v6_23.hdf5")
emotion_dict= {'Angry': 0, 'Sad': 5, 'Neutral': 4, 'Disgust': 1, 'Surprise': 6, 'Fear': 2, 'Happy': 3}
cascPath = "haarcascade_frontalface_alt.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)

def facecrop(faces, image):
    for f in faces:
        x, y, w, h = [ v for v in f ]
        cv2.rectangle(image, (x,y), (x+w,y+h), (255,255,255))

        sub_face = image[y:y+h, x:x+w]
        sub_face = cv2.resize(sub_face, (48,48))
        return sub_face

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    face_crop = facecrop(faces,gray)
    if face_crop is not None:
        face_image = np.reshape(face_crop, [1, face_crop.shape[0], face_crop.shape[1], 1])
        predicted_class = np.argmax(model.predict(face_image))
        label_map = dict((v,k) for k,v in emotion_dict.items())
        predicted_label = label_map[predicted_class]
        print(predicted_label)

    # Draw a rectangle around the faces
    #for (x, y, w, h) in faces:
    #    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    #cv2.imshow('Video', frame)

    #if cv2.waitKey(1) & 0xFF == ord('q'):
    #    break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()

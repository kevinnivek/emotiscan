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

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

model = load_model("./emotions/model_v6_23.hdf5")
cascPath = "./haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

# Read the image
#image = cv2.imread('kevin.jpg')
#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
#faces = faceCascade.detectMultiScale(
#    gray,
#    scaleFactor=1.1,
#    minNeighbors=5,
#    minSize=(30, 30),
#    flags = cv2.CASCADE_SCALE_IMAGE
#)

#print("Found {0} faces!".format(len(faces)))

video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()


    #faces = faceCascade.detectMultiScale(
    #    frame,
    #    scaleFactor=1.1,
    #    minNeighbors=5,
    #    minSize=(30, 30),
    #    flags=cv2.CASCADE_SCALE_IMAGE
    #)
    predicted_class = np.argmax(model.predict(frame))
    pprint(predicted_class)

    # Draw a rectangle around the faces
#    for (x, y, w, h) in faces:
#        #cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
#        print("face found at x: " + str(x) + " y: " + str(y))


    # Display the resulting frame
    #cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()


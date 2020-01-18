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
import glob

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

# functions
def facecrop(img):
    facedata = "haarcascade_frontalface_alt.xml"
    cascade = cv2.CascadeClassifier(facedata)

#    minisize = (img.shape[1],img.shape[0])
#    miniframe = cv2.resize(img, minisize)

#    faces = cascade.detectMultiScale(miniframe)
    faces = cascade.detectMultiScale(
        img,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )

    pprint(faces)
    for f in faces:
        x, y, w, h = [ v for v in f ]
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,255))

        sub_face = img[y:y+h, x:x+w]
        print('saving file ..')
        cv2.imwrite("./cropped.jpg", sub_face)



    return

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()
    
#    face_crop = facecrop(frame)
    face_image = cv2.resize(frame, (48,48))
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    face_image = np.reshape(face_image, [1, face_image.shape[0], face_image.shape[1], 1])
    predicted_class = np.argmax(model.predict(face_image))
    label_map = dict((v,k) for k,v in emotion_dict.items())
    predicted_label = label_map[predicted_class]
    print(predicted_label)

#ret, frame = video_capture.read()
#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#cv2.imwrite('test.jpg', gray)
#facecrop(gray)
#print('done')

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()

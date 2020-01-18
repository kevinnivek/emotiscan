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
def DetectFace(image):
    # Detect the faces
    faces = face_recognition.face_locations(image)
    pprint(faces)

    return faces

def faceCrop(image,boxScale=1):
    # Select one of the haarcascade files:
    #   haarcascade_frontalface_alt.xml  <-- Best one?
    #   haarcascade_frontalface_alt2.xml
    #   haarcascade_frontalface_alt_tree.xml
    #   haarcascade_frontalface_default.xml
    #   haarcascade_profileface.xml
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

    faces=DetectFace(image)
    if faces:
        #n=1
        for face in faces:
            croppedImage=imgCrop(pil_im, face[0],boxScale=boxScale)
            croppedImage.save('./test_crop.jpg')
            return croppedImage
            #fname,ext=os.path.splitext(img)
            #croppedImage.save(fname+'_crop'+str(n)+ext)
            #n+=1
    else:
        #print('No faces found:', img)
        return False

def pil2cvGrey(pil_im):
    # Convert a PIL image to a greyscale cv image
    # from: http://pythonpath.wordpress.com/2012/05/08/pil-to-opencv-image/
    pil_im = pil_im.convert('L')
    cv_im = cv2.CreateImageHeader(pil_im.size, cv.IPL_DEPTH_8U, 1)
    cv.SetData(cv_im, pil_im.tostring(), pil_im.size[0]  )
    return cv_im

def cv2pil(cv_im):
    # Convert the cv image to a PIL image
    return Image.fromstring("L", cv.GetSize(cv_im), cv_im.tostring())

def imgCrop(image, cropBox, boxScale=1):
    # Crop a PIL image with the provided box [x(left), y(upper), w(width), h(height)]

    # Calculate scale factors
    xDelta=max(cropBox[2]*(boxScale-1),0)
    yDelta=max(cropBox[3]*(boxScale-1),0)

    # Convert cv box to PIL box [left, upper, right, lower]
    PIL_box=[cropBox[0]-xDelta, cropBox[1]-yDelta, cropBox[0]+cropBox[2]+xDelta, cropBox[1]+cropBox[3]+yDelta]

    return image.crop(PIL_box)


while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()
    
    face_crop = faceCrop(frame)
    face_image = cv2.resize(face_crop, (48,48))
    #face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    face_image = np.reshape(face_image, [1, face_image.shape[0], face_image.shape[1], 1])
    predicted_class = np.argmax(model.predict(face_image))
    label_map = dict((v,k) for k,v in emotion_dict.items())
    predicted_label = label_map[predicted_class]
    print(predicted_label)

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()

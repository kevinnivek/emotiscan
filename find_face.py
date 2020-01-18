#!/usr/bin/python3

import face_recognition
from pprint import pprint

image = face_recognition.load_image_file("kevin.jpg")
face_landmarks_list = face_recognition.face_landmarks(image)

pprint(face_landmarks_list)

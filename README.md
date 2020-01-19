# Faceman : An interactive art project

## Introduction

"Faceman" was concieved as an art installation project to combine AI based facial and facial expression (or emotion) detection with an immersive experience using multiple chained matrix LED output devices. This repository is an implementation of emotion recognition using (Deep Convolutional Neural Networks)[https://github.com/atulapra/Emotion-detection/blob/master/ResearchPaper.pdf].

The objective is to interact with the participant by recognizing their face, potentially matching with a pre-built database of matched faces, then scanning their face to determine what expressive emotion they may be conveying.

A chained set of matrix LED devices will then be used to respond to the participant, based on their detected face and detected emotion. Ultimately the experience by the participant will be immersive in nature by bridging the gap between people and the machines they use in every day life. If a machine can not only recognize their face, but interpret their emotions then the connection between machines and humans deepens to an almost philsophical level.

## Hardware Dependencies

The system is designed to utilize the following hardware components :

- Raspberry Pi 4 (2GB or 4GB ram)
- 8 Megapixel NoIR Camera v2
- Four or more 32x32 5mm pitch matrix LED panels

## Dependencies

- Python 3, OpenCV 3 or 4, Tensorflow 1 or 2



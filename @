#!/usr/bin/env python3
import cv2
import sys
from pprint import pprint
from samplebase import SampleBase
from rgbmatrix import graphics
from rgbmatrix import RGBMatrix, RGBMatrixOptions
import time

# Configuration for the matrix
options = RGBMatrixOptions()
options.rows = 32
options.cols = 32
options.disable_hardware_pulsing = 1
options.chain_length = 1
options.parallel = 1

options.hardware_mapping = 'adafruit-hat'  # If you have an Adafruit HAT: 'adafruit-hat'
#options.pwm_bits = 1
#options.pwm_dither_bits = 2

options.gpio_slowdown = 2 
matrix = RGBMatrix(options = options)
canvas = matrix
font = graphics.Font()
#font.CharacterWidth(60)
font.LoadFont("./fonts/10x20.bdf")
color = graphics.Color(125, 125, 255)

#canvas.Clear()
graphics.DrawText(canvas, font, 1, 20, color, '>:(')
input("Press Enter to continue...")

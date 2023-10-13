#!/usr/bin/env python
from samplebase import SampleBase
from rgbmatrix import graphics
from rgbmatrix import RGBMatrix, RGBMatrixOptions
import time
#import displayio
#import board
#import framebufferio

#bit_depth = 1
#base_width = 64
#base_height = 64
#chain_across = 2
#tile_down = 1
#serpentine = True
#width = base_width * chain_across
#height = base_height * tile_down


class GraphicsTest(SampleBase):
    def __init__(self, *args, **kwargs):
        super(GraphicsTest, self).__init__(*args, **kwargs)

    def run(self):
	options = RGBMatrixOptions()
	options.rows = 64
	options.cols = 64
	#options.chain_length = 1 
	#options.parallel = 1
	#options.hardware_mapping = 'adafruit-hat'  # If you have an Adafruit HAT: 'adafruit-hat'
	#options.gpio_slowdown = 2 
	matrix = RGBMatrix(options = options)

        canvas = self.matrix
        font = graphics.Font()
        font.LoadFont("fonts/7x13.bdf")

        red = graphics.Color(255, 0, 0)
        graphics.DrawLine(canvas, 5, 5, 22, 13, red)

        green = graphics.Color(0, 255, 0)
        graphics.DrawCircle(canvas, 15, 15, 10, green)

        blue = graphics.Color(0, 0, 255)
        graphics.DrawText(canvas, font, 2, 10, blue, "Text")

        time.sleep(10)   # show display for 10 seconds before exit


# Main function
if __name__ == "__main__":
    graphics_test = GraphicsTest()
    if (not graphics_test.process()):
        graphics_test.print_help()

#!/usr/bin/python

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np

# Read in .png and convert to 0,255 bytescale
image = (mpimg.imread('exit-ramp.png')*255).astype('uint8')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Define a kernel size for Gaussian smoothing / blurring
kernel_size = 3
blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size), 0)

low_threshold = 50
high_threshold = 150
edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

plt.imshow(edges, cmap='Greys_r')
plt.show()

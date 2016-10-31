#!/usr/bin/python

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np

# Read in .png and convert to 0,255 bytescale
image = (mpimg.imread('exit-ramp.png')*255).astype('uint8')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Define a kernel size for Gaussian smoothing / blurring
kernel_size = 5
blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size), 0)

# Define our params for Canny and apply
low_threshold = 50
high_threshold = 150
edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

# Create a masked edges image 
mask = np.zeros_like(edges)
ignore_mask_color = 255
# Define a four sided polygon mask
imshape = image.shape
#vertices = np.array([[(0,imshape[0]), (0,0), (imshape[1],0), (imshape[1],imshape[0])]], dtype=np.int32)
vertices = np.array([[(50,imshape[0]), (420,300), (520,300), (imshape[1],imshape[0])]], dtype=np.int32)
cv2.fillPoly(mask, vertices, ignore_mask_color)
masked_edges = cv2.bitwise_and(edges, mask)

show_mask = False
if show_mask:
	x = [vertices[0][0][0], vertices[0][1][0], vertices[0][2][0], vertices[0][3][0]]
	y = [vertices[0][0][1], vertices[0][1][1], vertices[0][2][1], vertices[0][3][1]]
	plt.plot(x, y, 'b--', lw=4)
	plt.imshow(masked_edges)
	plt.show()


# Define the Hough transform parameters
# Make a blank the same size as our image to draw on
rho = 1  # in pixels
theta = np.pi/180 # in radians
threshold = 30 # number of votes/intersections needed
min_line_length = 30 # pixels 
max_line_gap = 5 # pixels between segments
line_image = np.copy(image)*0 #creating a blank to draw lines on

# Run Hough on edge detected image
lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)
print("Found {} lines".format(lines.shape[0]))

# Iterate over the output "lines" and draw lines on the blank
for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)

# Create a "color" binary image to combine with line image
color_edges = np.dstack((edges, edges, edges)) 

# Draw the lines on the edge image
lines_edges = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0) 

plt.imshow(lines_edges)
plt.show()

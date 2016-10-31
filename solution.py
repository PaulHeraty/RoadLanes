#!/home/pedgrfx/anaconda3/bin/python

#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os

import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def get_slope(line):
    for x1,y1,x2,y2 in line:
        slope = (y2-y1) / (1.0 * (x2-x1))
    return slope

def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate

@static_vars(last_lx1=0,last_ly1=0,last_lx2=0,last_y2=0,last_rx1=0,last_ry1=0,last_rx2=0,last_ry2=0)
def draw_lines(img, lines, color=[255, 0, 0], thickness=4):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
# First let's divide the lines into left and right lane by slope
    left_lines = np.empty((0,4), dtype=np.int32)
    right_lines = np.empty((0,4), dtype=np.int32)

    annotate_lines = True    
    for line in lines:
        if get_slope(line) > 0.1:
            right_lines = np.append(right_lines, line, axis=0)
            if annotate_lines:
                for x1,y1,x2,y2 in line:
                    #print("Line {},{}, {},{}".format(x1,y1,x2,y2))
                    cv2.line(img, (x1, y1), (x2, y2), [0,255,0], thickness)
        elif get_slope(line) < -0.1:
            left_lines= np.append(left_lines, line, axis=0)
            if annotate_lines:
                for x1,y1,x2,y2 in line:
                    #print("Line {},{}, {},{}".format(x1,y1,x2,y2))
                    cv2.line(img, (x1, y1), (x2, y2), [0,0,255], thickness)
    
    
    draw_new_left = False
    draw_new_right = False
    if right_lines.shape[0] > 1:
        right_coeff = np.polyfit(np.append(right_lines[:,0], right_lines[:,2]), np.append(right_lines[:,1], right_lines[:,3]), 1)
        #print("Lines {}".format(right_lines))
        #print("Right lane num {}  slope {}".format(right_lines.shape[0], right_coeff[0]))
        if right_coeff[0] > 0.001 or right_coeff[0] < -0.001:
            draw_new_right = True
    if left_lines.shape[0] > 1:
        left_coeff = np.polyfit(np.append(left_lines[:,0], left_lines[:,2]), np.append(left_lines[:,1], left_lines[:,3]), 1)
        #print("Left lane num {}  slope {}".format(left_lines.shape[0], left_coeff[0]))
        if left_coeff[0] > 0.001 or left_coeff[0] < -0.001:
            draw_new_left = True
    
    # Draw right lane
    if draw_new_right:
        # Find bottom coord, we know Y = 540
        ry1 = 540
        r_c = right_coeff[1]
        r_m = right_coeff[0]
        rx1 = int((ry1 - r_c)/ r_m)
        #Find top coord, we know Y = 320
        ry2 = 320
        rx2 = int((ry2 - r_c)/ r_m)
        cv2.line(img, (rx1, ry1), (rx2, ry2), color, thickness)
        draw_lines.old_rx1 = rx1
        draw_lines.old_ry1 = ry1
        draw_lines.old_rx2 = rx2
        draw_lines.old_ry2 = ry2
    else:
        cv2.line(img, (draw_lines.old_rx1, draw_lines.old_ry1), (draw_lines.old_rx2, draw_lines.old_ry2), color, thickness)
    
    #Draw left lane
    if draw_new_left:
        # Find bottom coord, we know Y = 540
        ly1 = 540
        l_c = left_coeff[1]
        l_m = left_coeff[0]
        lx1 = int((ly1 - l_c)/ l_m)
        #Find top coord, we know Y = 320
        ly2 = 320
        lx2 = int((ly2 - l_c)/ l_m)
        #print("lx1 ly1 lx2 ly1 lm {} {} {} {} {}".format(lx1, ly1, lx2, ly2, l_m))
        cv2.line(img, (lx1, ly1), (lx2, ly2), color, thickness)
        draw_lines.old_lx1 = lx1
        draw_lines.old_ly1 = ly1
        draw_lines.old_lx2 = lx2
        draw_lines.old_ly2 = ly2
    else:
        cv2.line(img, (draw_lines.old_lx1, draw_lines.old_ly1), (draw_lines.old_lx2, draw_lines.old_ly2), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((*img.shape, 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

imageList = os.listdir("test_images/")
roi = np.array([[(140,540), (460,320), (510,320), (900,540)]])

def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image with lines are drawn on lanes)
    gray = grayscale(image)
    blur_gray = gaussian_blur(gray, 5)
    edges = canny(blur_gray, low_threshold=50, high_threshold=150)
    masked_img = region_of_interest(edges, roi)
    hough_img = hough_lines(masked_img, rho=1, theta=np.pi/180, threshold=40, min_line_len=5, max_line_gap=10)
    #hough_img = hough_lines(masked_img, rho=1, theta=np.pi/180, threshold=30, min_line_len=5, max_line_gap=1)
    result = weighted_img(hough_img, image)
    return result


proc_images = False
if proc_images:
    for img in imageList:
        imgFileName = "./test_images/" + img
        image = mpimg.imread(imgFileName)
        final_img = process_image(image)
        plt.imshow(final_img)
        plt.show()

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

#white_output = 'white.mp4'
#clip1 = VideoFileClip("solidWhiteRight.mp4")
#white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
#white_clip.write_videofile(white_output, audio=False)

#yellow_output = 'yellow.mp4'
#clip2 = VideoFileClip('solidYellowLeft.mp4')
#yellow_clip = clip2.fl_image(process_image)
#yellow_clip.write_videofile(yellow_output, audio=False)

challenge_output = 'extra.mp4'
clip2 = VideoFileClip('challenge.mp4')
challenge_clip = clip2.fl_image(process_image)
challenge_clip.write_videofile(challenge_output, audio=False)

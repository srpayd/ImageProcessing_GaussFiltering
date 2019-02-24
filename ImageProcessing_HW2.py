
"""
Created on Sun Feb 24 17:41:44 2019
@author: serap aydogdu
"""

#  NOTE: Please note that you should close the popped-up image window first to continue)


#Load Libraries
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import data
from skimage.filters import try_all_threshold
import scipy.ndimage

# Read the Image  
img=cv2.imread('lenna.png',1)
cv2.namedWindow("Colorful Image")
cv2.imshow('Colorful Image',img)
k = cv2.waitKey(0)
cv2.destroyAllWindows()  



# ==================================================================================================================
# # Q1: Add the following zero mean Gaussioan noises, seperately to red, green and blue channels of 256X256 lena image
# ==================================================================================================================

# Step 1: Obtain seperately red, green and blue channels of 256X256 lena image.

blue_image = img.copy() # Make a copy
blue_image[:,:,1] = 0   # Green channel=0
blue_image[:,:,2] = 0   # Red channel=0 
cv2.imshow('Blue Image',blue_image)

green_image = img.copy() # Make a copy
green_image[:,:,0] = 0   # Red channel=0
green_image[:,:,2] = 0   # Blue channel=0 
cv2.imshow('Green Image',green_image)

red_image = img.copy() # Make a copy
red_image[:,:,0] = 0   # Blue channel=0
red_image[:,:,1] = 0    # Green channel=0 
cv2.imshow('Red Image',red_image)

#Step 2: Add Gaussioan noises to those channels seperatly.

gauss1  = img + np.random.normal(0,1,img.shape)
gauss5  = img + np.random.normal(0,5,img.shape)
gauss10 = img + np.random.normal(0,10,img.shape)
gauss20 = img + np.random.normal(0,20,img.shape)

cv2.imshow('Colorful Image',img)     # Display original lenna.png
cv2.imwrite("gauss1.png", gauss1)    # Save the gauss1 as gauss1.png
gauss1 = cv2.imread("gauss1.png")    # Then read the gauss1.png
cv2.imshow('gauss1',gauss1)          # Display the gauss1.png

cv2.imwrite("gauss5.png", gauss5)    # Save the gauss5 as gauss5.png
gauss5 = cv2.imread("gauss5.png")    # Then read the gauss5.png
cv2.imshow('gauss5',gauss5)          # Display the gauss5.png

cv2.imwrite("gauss10.png", gauss10)  # Save the gauss10 as gauss10.png
gauss10 = cv2.imread("gauss10.png")  # Then read the gauss10.png
cv2.imshow('gauss10',gauss10)        # Display the gauss10.png

cv2.imwrite("gauss20.png", gauss20)  # Save the gauss20 as gauss20.png
gauss20 = cv2.imread("gauss20.png")  # Then read the gauss20.png  
cv2.imshow('gauss20',gauss20)        # Display the gauss20.png

# Gaussian noise is specified by its mean and its variance (σ2n) or its standard deviation (σn). 
# As we observe, whereas standard deviation increasing, the images become smoother.






# ==========================================================================================================================================
# Q2: Obtain gray scale images, I_1, I_5, I_10, I_20 by taking the average values of R,G,B channels corresponding to different noise levels.
# ==========================================================================================================================================

# Step 1: Obtain grayscale image by taking average values of RGB channels

I = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray Scale Image',I)

# Step 2: Add Gaussian noises to obtained grayscale image on Step1

I1  = I + np.random.normal(0,1,I.shape)
I5  = I + np.random.normal(0,5,I.shape)
I10 = I + np.random.normal(0,10,I.shape)
I20 = I + np.random.normal(0,20,I.shape)

cv2.imwrite("I1.png", I1)                        # Save the I1 level Gauss noised image as I1.png
gauss_I1 = cv2.imread("I1.png")                  # Then read the I1.png
cv2.imshow('I1 Grayscale Gauss ',gauss_I1)       # Display the I1.png

cv2.imwrite("I5.png", I5)                        # Save the I5 level Gauss noised image as I5.png
gauss_I5 = cv2.imread("I5.png")                  # Then read the I5.png
cv2.imshow('I5 Grayscale Gauss ',gauss_I5)       # Display the I5.png

cv2.imwrite("I10.png", I10)                      # Save the I10 level Gauss noised image as I10.png
gauss_I10 = cv2.imread("I10.png")                # Then read the I10.png
cv2.imshow('I10 Grayscale Gauss ',gauss_I10)     # Display the I10.png

cv2.imwrite("I20.png", I20)                      # Save the I20 level Gauss noised image as I20.png
gauss_I20 = cv2.imread("I20.png")                # Then read the I20.png
cv2.imshow('I20 Grayscale Gauss ',gauss_I20)     # Display the I20.png







# ==========================================================================================================================================
# Q3: Filter these images using low-pass filters with kernel.
# ==========================================================================================================================================

# Create a 3x3 kernel
kernel_3x3 = np.array([
    [-1, -1, -1],
    [-1,  9, -1],
    [-1, -1, -1]
])

# Apply the 3x3 low pass filter with kernel
lpf_1  = cv2.filter2D(gauss_I1,  -1, kernel_3x3)
lpf_5  = cv2.filter2D(gauss_I5,  -1, kernel_3x3)
lpf_10 = cv2.filter2D(gauss_I10, -1, kernel_3x3)
lpf_20 = cv2.filter2D(gauss_I20, -1, kernel_3x3)

# Display images
cv2.imshow("lpf I1",  lpf_1)
cv2.imshow("lpf I5",  lpf_5)
cv2.imshow("lpf I10", lpf_10)
cv2.imshow("lpf I20", lpf_20)


# Create a 5x5 kernel
kernel_5x5 = np.array([
    [0.04, 0.04, 0.04, 0.04, 0.04],
    [0.04, 0.04, 0.04, 0.04, 0.04],
    [0.04, 0.04, 0.04, 0.04, 0.04],
    [0.04, 0.04, 0.04, 0.04, 0.04],
    [0.04, 0.04, 0.04, 0.04, 0.04]
])

# Apply the 5x5 low pass filter with kernel
LPF_1  = cv2.filter2D(gauss_I1,  -1, kernel_5x5)
LPF_5  = cv2.filter2D(gauss_I5,  -1, kernel_5x5)
LPF_10 = cv2.filter2D(gauss_I10, -1, kernel_5x5)
LPF_20 = cv2.filter2D(gauss_I20, -1, kernel_5x5)

# Display images
cv2.imshow("LPF I1",  LPF_1)
cv2.imshow("LPF I5",  LPF_5)
cv2.imshow("LPF I10", LPF_10)
cv2.imshow("LPF I20", LPF_20)

# LPF (low-pass filtering) is usually used to remove noise, blur, smoothen an image.
# A kernel is a matrix contains weights, which always has an odd size (1,3,5,7,..).
# In case of LPF, all values in kernel sum up to 1. If the kernel contains both negative and positive weights, it’s probably used to sharpen (or smoothen) an image. 





# ==========================================================================================================================================
# Q4: Filter the same images using high-pass filters this time with kernel.
# ==========================================================================================================================================

# Create a 3x3 kernel
kernel_3x3 = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
])

# Apply the 3x3 high pass filter with kernel
hpf_1  = cv2.filter2D(gauss_I1,  -1, kernel_3x3)
hpf_5  = cv2.filter2D(gauss_I5,  -1, kernel_3x3)
hpf_10 = cv2.filter2D(gauss_I10, -1, kernel_3x3)
hpf_20 = cv2.filter2D(gauss_I20, -1, kernel_3x3)

# Display images
cv2.imshow("hpf I1",  hpf_1)
cv2.imshow("hpf I5",  hpf_5)
cv2.imshow("hpf I10", hpf_10)
cv2.imshow("hpf I20", hpf_20)


# Create a 5x5 kernel
kernel_5x5 = np.array([
    [-1, -1, -1, -1, -1],
    [-1,  1,  2,  1, -1],
    [-1,  2,  4,  2, -1],
    [-1,  1,  2,  1, -1],
    [-1, -1, -1, -1, -1]
])

# Apply the 5x5 high pass filter with kernel
HPF_1  = cv2.filter2D(gauss_I1,  -1, kernel_5x5)
HPF_5  = cv2.filter2D(gauss_I5,  -1, kernel_5x5)
HPF_10 = cv2.filter2D(gauss_I10, -1, kernel_5x5)
HPF_20 = cv2.filter2D(gauss_I20, -1, kernel_5x5)

# Display images
cv2.imshow("HPF I1",  HPF_1)
cv2.imshow("HPF I5",  HPF_5)
cv2.imshow("HPF I10", HPF_10)
cv2.imshow("HPF I20", HPF_20)

# HPF (high-pass filtering) is usually used to detect edges in an image.
# A kernel is a matrix contains weights, which always has an odd size (1,3,5,7,..).
# In case of HPS, the kernel contains both negative and positive weights, sum up to 0. 





# ==========================================================================================================================================
# Q5: Filter these images using low-pass filters with kernel.
# ==========================================================================================================================================

# Step 1: Read the noised lenna Image  
noised_img=cv2.imread('lenna_noised.png',1)
cv2.namedWindow("Noised Lenna Image")
cv2.imshow('Noised Lenna Image',noised_img)
k = cv2.waitKey(0)
cv2.destroyAllWindows()  

# Median filtering is going to applied to the image. Median filtering is a nonlinear method used to remove 
# noise from images. It is particularly effective at removing 'salt and pepper' type noises which is also
# taken place in this image.The median filter works by moving through the image pixel by pixel replacing each value with the median
# value of neighbouring pixels.

# Step 2: Median filter is applied.
median_filter = cv2.medianBlur(noised_img, 3)           # Median Filter is applied to reduce Salt & Pepper Effect
cv2.imwrite("lenna_median_filter.png", median_filter)   # Save the modified image to current directory
lenna_median = cv2.imread("lenna_median_filter.png")    # Read the image from current directory    
cv2.imshow('Median Lenna Image',lenna_median)           # Then finally display the image that median filter is applied

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle



def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0,255)): #thresh_min=0, thresh_max=255):
    
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if orient=='x':
        sobel=cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=sobel_kernel)
    else:
        sobel=cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=sobel_kernel)
        
    #print(sobel.shape)    
    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    #print(abs_sobel.shape)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    
    scaled_sobel=np.uint8(255*abs_sobel/np.max(abs_sobel))
    #print(type(scaled_sobel[0][0]))
    # 5) Create a mask of 1's where the scaled gradient magnitude 
            # is > thresh_min and < thresh_max
        
    binary_output=np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])]=1
    
    #(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)
    # 6) Return this mask as your binary_output image
    return binary_output

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelX=cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=sobel_kernel)
    sobelY=cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=sobel_kernel)
    
    # 3) Calculate the magnitude 
    abs_sobelXY=np.sqrt(np.square(sobelX)+np.square(sobelY))
    #print(abs_sobelXY.shape)
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel=np.uint8(255*abs_sobelXY/np.max(abs_sobelXY))
    # 5) Create a binary mask where mag thresholds are met
    binary_output=np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])]=1
    # 6) Return this mask as your binary_output image
    
    return binary_output

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelX=cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=sobel_kernel)
    sobelY=cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    abs_sobelX = np.absolute(sobelX)
    abs_sobelY = np.absolute(sobelY)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    dirGrad=np.arctan2(abs_sobelY,abs_sobelX)
    # 5) Create a binary mask where direction thresholds are met
    binary_output=np.zeros_like(dirGrad)
    binary_output[(dirGrad >= thresh[0]) & (dirGrad <= thresh[1])]=1
    # 6) Return this mask as your binary_output image
    return binary_output


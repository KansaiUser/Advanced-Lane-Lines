import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


#WARNING:  These imports have to be done with scripts because they are being called from code. 
from scripts.my_color_thresh import HLScolor_thresh

from scripts.my_sobel_thresh import abs_sobel_thresh
from scripts.my_sobel_thresh import mag_thresh
from scripts.my_sobel_thresh import dir_threshold

def pipeline(img, channel='s', color_thresh=(110, 255), orient='x',sobel_thresh=(20, 100)):
    
    # We are considering here only X and Y abs sobel threshold.     
    
    grad_sobel = abs_sobel_thresh(img, orient=orient, thresh=sobel_thresh)
    
    #binary_color=HLScolor_thresh(img,channel,color_thresh)
    binary_color=HLScolor_thresh(img,channel=channel,thresh=color_thresh)
    
    # Stack each channel 
    color_binary = np.dstack(( np.zeros_like(grad_sobel), grad_sobel, binary_color)) * 255
    #print(color_binary.shape)
    
    #Then combine them
    # Combine the two binary thresholds
    combined_binary = np.zeros_like(grad_sobel)
    combined_binary[(binary_color == 1) | (grad_sobel == 1)] = 1

    return color_binary, combined_binary


import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#import pickle
#import glob

def HLScolor_thresh(image, channel='s', thresh=(0,255)):
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    if channel=='s':
        channel_img=hls[:,:,2]
    elif channel=='l':
        channel_img=hls[:,:,1]
    elif channel=='h':
        channel_img=hls[:,:,0]
    else:
        print("Error in channel!")
        return
    
    binary_image = np.zeros_like(channel_img)
    binary_image[(channel_img > thresh[0]) & (channel_img <= thresh[1])] = 1
    
    return binary_image

    

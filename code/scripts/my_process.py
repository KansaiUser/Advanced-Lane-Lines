import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#import glob

from scripts.my_pipeline import pipeline
from scripts.my_camera_cal import image_perspective
from scripts.my_line_finding import fit_polynomial
from scripts.my_line_finding import search_around_poly

def process(img_name,mtx,dist,l_fit=np.array([]),r_fit=np.array([])):
    image = mpimg.imread(img_name)
    #undistort it
    dst = cv2.undistort(image, mtx, dist, None, mtx)
    color_binary,combined_binary= pipeline(dst,channel='s',color_thresh=(110,255), orient='x', sobel_thresh=(20, 100))
    bird_image,M=image_perspective(combined_binary)
    
    if l_fit.size==0 or r_fit.size==0:        
        #if we don't have the l_fit,r_fit
        l_fit,r_fit,polynomial_image=fit_polynomial(bird_image)
    else:
        l_fit,r_fit,polynomial_image=search_around_poly(bird_image,l_fit,r_fit)
        
    final=back_to_world(dst,bird_image,l_fit,r_fit)    
    
    return l_fit,r_fit,polynomial_image,final

def process2(image,mtx,dist,l_fit=np.array([]),r_fit=np.array([])):
    #image = mpimg.imread(img_name)
    #undistort it
    dst = cv2.undistort(image, mtx, dist, None, mtx)
    color_binary,combined_binary= pipeline(dst,channel='s',color_thresh=(110,255), orient='x', sobel_thresh=(20, 100))
    bird_image,M=image_perspective(combined_binary)
    
    if l_fit.size==0 or r_fit.size==0:        
        #if we don't have the l_fit,r_fit
        l_fit,r_fit,polynomial_image=fit_polynomial(bird_image)
    else:
        l_fit,r_fit,polynomial_image=search_around_poly(bird_image,l_fit,r_fit)
        
    final=back_to_world(dst,bird_image,l_fit,r_fit)    
    
    return l_fit,r_fit,polynomial_image,final

# back_to_world(dst,bird_image,l_fit,r_fit)


def back_to_world(real_image,bird_image,l_fit,r_fit,source=np.float32 ([(580, 460), (202, 720), (1110, 720), (703, 460)]), 
                  destino=np.float32([(336, 0), (336, 720), (976, 720), (976, 0)])):
    z = np.zeros_like(bird_image)
    bird_lane = np.dstack((z, z, z))
    
    kl, kr = l_fit, r_fit
    h=bird_lane.shape[0]
    ys = np.linspace(0, h - 1, h)
    lxs = kl[0] * (ys**2) + kl[1]* ys +  kl[2]
    rxs = kr[0] * (ys**2) + kr[1]* ys +  kr[2]
    
    pts_left = np.array([np.transpose(np.vstack([lxs, ys]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([rxs, ys])))])
    pts = np.hstack((pts_left, pts_right))
    
    cv2.fillPoly(bird_lane, np.int_(pts), (0,255,0))
    
    i_shape = (bird_lane.shape[1], bird_lane.shape[0])

    #source=np.float32 ([(580, 460), (202, 720), (1110, 720), (703, 460)])
    #destino=np.float32([(336, 0), (336, 720), (976, 720), (976, 0)])

    inv_M=cv2.getPerspectiveTransform(destino, source)


    ground_lane = cv2.warpPerspective(bird_lane, inv_M, i_shape)
    
    final_image = cv2.addWeighted(real_image, 1, ground_lane, 0.3, 0)
    return final_image



# left_fit right_fit comes from the data
#y_eval is the height of the pics

def measure_curvature_pixels(left_fit,right_fit,y_eval):
    '''
    Calculates the curvature of polynomial functions in pixels.
    '''    
    # Calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    
    return left_curverad, right_curverad
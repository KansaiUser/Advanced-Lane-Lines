import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from pathlib import Path
import pickle


# directory:  the directory where the callibration images are
# nx,ny the number of points inside the chess board
def find_points(directory,nx,ny):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((nx*ny,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.
    
    # Make a list of calibration images
    input_path=Path(directory)
    limages=list(input_path.glob('*.jpg'))
    #images = glob.glob('../camera_cal/*.jpg')
    #print(limages)
    for input_image in limages:
        #print(input_image)
        img = cv2.imread(str(input_image))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)
    
        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
            
            # Draw and display the corners
            #cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
            #write_name = 'corners_found'+str(idx)+'.jpg'
            #cv2.imwrite(write_name, img)
            #cv2.imshow('img', img)
            #cv2.waitKey(500)
            
    #cv2.destroyAllWindows()
    return objpoints,imgpoints    



def undistort(image,objpoints,imgpoints):
    img_size = (image.shape[1], image.shape[0])
    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
    dst = cv2.undistort(image, mtx, dist, None, mtx)
    
    return dst

def saveValuesToUndistort(objpoints,imgpoints,img_size,filename):
    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
    # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    filename=filename+".p"
    print(filename)
    pickle.dump( dist_pickle, open( filename, "wb" ) )
    
def readValuesToUndistort(filename):
    dist_pickle=pickle.load( open( filename, "rb" ) )
    mtx=dist_pickle["mtx"]
    dist=dist_pickle["dist"]
    return mtx,dist

def unwarp(undist,nx,ny):
    gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    if(ret==True):
        dest2=undist.copy()
        cv2.drawChessboardCorners(dest2, (nx, ny), corners, ret)
        offset = 100 # offset for dst points
        # Grab the image shape
        img_size = (gray.shape[1], gray.shape[0])
        src = np.float32([corners[0], corners[nx-1], corners[-1], corners[-nx]])
        #print(src)
        dst = np.float32([[offset, offset], [img_size[0]-offset, offset], 
                                     [img_size[0]-offset, img_size[1]-offset], 
                                     [offset, img_size[1]-offset]])
        #print(dst)
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(dest2, M, img_size)
        
    return warped,M

def image_perspective(image,source=np.float32 ([(580, 460), (202, 720), (1110, 720), (703, 460)]), destino=np.float32([(336, 0), (336, 720), (976, 720), (976, 0)])):
    img_size = (image.shape[1], image.shape[0])
    
    M = cv2.getPerspectiveTransform(source, destino)
    warped = cv2.warpPerspective(image, M, img_size)
    
    return warped,M

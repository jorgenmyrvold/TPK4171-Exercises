import os
import glob
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def calibrate(path='calibration_img/*.JPG', rows = 6, columns = 9, size = 0.0213):
    
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*7,3), np.float32)
    objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    images = glob.glob(path)
    num_imgs = len(images)
    counter = 0

    for idx, fname in enumerate(images):
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (columns ,rows), None)
        
        # If found, add object points, image points (after refining them)
        if ret == True:
            print('Corners found: %s\t%d/%d' % (os.path.basename(os.path.normpath(fname)), idx, num_imgs))
            objpoints.append(objp)
            imgpoints.append(cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria))

        else:
            print('Corners not found:', fname, str(counter)+'/'+str(num_imgs))
    
    print('Calculating...', end='')
    err, K, distCoeffs, Rs, ts = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print("Done.")    
    print("Err: ", err)
    print("K:\n", K)
    print("Distortion coefficients:\n", distCoeffs)

    

if __name__ == "__main__":
    calibrate()

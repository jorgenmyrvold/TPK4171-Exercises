import cv2 
import glob
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(precision=4, suppress=True)

def calibrate(path = 'calibration_img/*.JPG', rows = 6, columns = 9, size = 0.0213, verbose = 2):

    if not os.path.isabs(path):
        path = os.path.join(os.path.dirname(sys.argv[0]), path)
    
    images = glob.glob(path)

    try:
        assert rows > 0, "Wrong checkerboard dimensions."
        assert columns > 0, "Wrong checkerboard dimension."
        assert rows*columns >= 2, "Wrong checkerboard dimensions."
        assert len(images) >= 3, "Calibration requires more than three images."
        assert verbose >= 0 and verbose <= 2, "Wrong verbosity parameter."
    except AssertionError as error:
        print(error, " Quitting...")
        sys.exit(1)
  
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    worldFrame = np.zeros((rows*columns,3), np.float32)
    worldFrame[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2) * size 

    # Arrays to store object points and image points from all the images.
    worldPoints = [] # 3d point in real world space
    imagePoints = [] # 2d points in image plane.

    if verbose > 0:
        print("Finding chessboard corners...")

    for idx, filename in enumerate(images):
        img = cv2.imread(filename)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, (rows, columns), None)

        if found == True:
            worldPoints.append(worldFrame)
            imagePoints.append(cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria))

            if verbose > 0:
                print('Corners found: %s\t%d/%d' % (os.path.basename(filename), idx, len(images)))
            if verbose > 1:
                img = cv2.drawChessboardCorners(img, (rows, columns), corners, True)
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                plt.show()
        else:
            if verbose > 0:
                print('Corners not found: %s\t%d/%d' % (os.path.basename(os.path.normpath(filename)), idx, len(images)))

    err, K, distCoeffs, Rs, ts = cv2.calibrateCamera(worldPoints, imagePoints, gray.shape[::-1], None, None)
    
    if verbose > 0:
        print("\nErr: ", err)
        print("K:\n", K)
        print("Distortion coefficients:\n", distCoeffs, "\n")

    return (K, distCoeffs, Rs, ts)


if __name__ == "__main__":
    calibrate(verbose=1)
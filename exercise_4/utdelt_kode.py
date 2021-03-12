import cv2 
import glob
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(precision=4, suppress=True)

def calibrate(path = 'calibration_img/*.JPG', rows = 6, columns = 9, size = 0.0213, verbose = 2):
    '''
        Parameters:
            path (str):
                The relative or absolute path to the calibration images that is supported by
                glob.glob(). https://docs.python.org/3/library/glob.html
                Default: './Calibration Images/*.jpg' will assume that every
                jpg-file in the folder 'Calibration Images' is a picture of 
                the calibration board.
            rows & columns (int): 
                Number of _inner_ corners of the calibration checkerboard.
            size (float):
                Size of the checkerboard squares in _meters_. 
            verbose (int):
                If 0 - Silent.
                If 1 - Prints progress and final calibration.
                If 2 - Same as 1, but will also display the calibration
                       images annotated with the found corners.
                    
        Output: 
            err (float):
                I.e. the mean L2 norm between the imagePoints and the projected 
                worldPoints.
            K (3x3, float):
                Intrinsic camera matrix.
            distCoeffs (list of 5 floats):
                Lens distortion coefficients [k1, k2, p1, p2, k3]
                k_i are radial distrotion coefficients              
                p_i are tangential distortion coeffeicients
                https://docs.opencv.org/4.2.0/d9/d0c/group__calib3d.html#ga3207604e4b1a1758aa66acb6ed5aa65d
            Rs (list of n (3,1) vectors):
                Each element in the list is a Rodrigues vector 
            ts (list of n (3,1) vectors):
                Each element is a translation vector.

    '''
    # Here we make sure that the path becomes an absolute path. 
    # glob.glob() searches relative to the location from where the script is 
    # launched, not from where it is located. 
    # We solve this by always making the path into an absolute path.
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
  
    
    # OpenCV optimizes the camera calibration using the Levenberg-Marquardt 
    # algorithm (Simply put, a combination of Gauss-Newton and Gradient Descent)
    # This defines the stopping criteria.
    # The first parameter states that we limit the algorithm both in terms of
    # desired accuracy, as well as number of iterations. 
    # Here we limit it to 30 iterations or an accuracy of 0.001
    # The criteria is also used by the sub-pixel estimator.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # The calibration board specifies a world frame where the inside corners 
    # lie on the z=0 plane. Here we create a a list of these points.
    
    # First we initialize a list with (rows*columns) number of elements, each 
    # with three parameters, namely: x, y, z.
    worldFrame = np.zeros((rows*columns,3), np.float32)
    
    # Then we use some tensor magic. 
    # np.mgrid -- returns a (2, rows, columns) sized tensor. A tensor is simply 
    # a matrix with more than two dimensions. 
    # .T -- transposes the tensor to the shape (columns, rows, 2). I.e. numpy 
    # defines transposition as switching i and j indices. 
    # .reshape(-1, 2) -- Specifies that we want to remove a dimension by  
    # concatenating the elements of the two first dimensions.
    # We finally scale the coordinates so that they match with the size of the
    # actual checkerboard.
    worldFrame[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2) * size 


    # Arrays to store object points and image points from all the images.
    worldPoints = [] # 3d point in real world space
    imagePoints = [] # 2d points in image plane.

    if verbose > 0:
        print("Finding chessboard corners...")

    for idx, filename in enumerate(images):
        img = cv2.imread(filename)
        # We need a gray scale image for sub-pixel corner localization.
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        found, corners = cv2.findChessboardCorners(gray, (rows, columns), None)

        # If the above function takes a long time, or returns false you have 
        # probably inserted the wrong number of rows and columns. 
        # Remember: It's the inside corners that count. 
        # And: It depends if the pattern is photographed in landscape or 
        # portrait mode. 

        # If corners are found, we refine them by estimating their sub-pixel 
        # location. 
        if found == True:
            worldPoints.append(worldFrame)
            imagePoints.append(cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria))

            if verbose > 0:
                print("Found corners. %d / %d" % (idx+1, len(images)))
            if verbose > 1:
                img = cv2.drawChessboardCorners(img, (rows, columns), corners, True)
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                plt.show()
        else:
            if verbose > 0:
                print("Could not find corners in image %s. Image %d / %d" % (filename, idx+1, len(images)))
    ### end for 
    
    # This performes the actual calibration.
    # It returns:
    #   err (float):
    #        I.e. the mean L2 norm between the imagePoints and the projected 
    #        worldPoints.
    #   K (3x3, float):
    #        Intrinsic camera matrix.
    #   distCoeffs (list of 5 floats):
    #        Lens distortion coefficients [k1, k2, p1, p2, k3]
    #        k_i are radial distrotion coefficients              
    #        p_i are tangential distortion coeffeicients
    #       https://docs.opencv.org/4.2.0/d9/d0c/group__calib3d.html#ga3207604e4b1a1758aa66acb6ed5aa65d
    #    Rs (list of n (3,1) vectors):
    #       Each element in the list is a Rodrigues vector 
    #    ts (list of n (3,1) vectors):
    #       Each element is a translation vector.
    # 
    if verbose > 0:
        print("Calibrating...", end=' ')
    err, K, distCoeffs, Rs, ts = cv2.calibrateCamera(worldPoints, imagePoints, gray.shape[::-1], None, None)
    
    if verbose > 0:
        print("Done.")    
        print("Err: ", err)
        print("K:\n", K)
        print("Distortion coefficients:\n", distCoeffs)

    return (K, distCoeffs, Rs, ts)


if __name__ == "__main__":
    calibrate(verbose=1)
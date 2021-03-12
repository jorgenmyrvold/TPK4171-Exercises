import numpy as np
import cv2 
import glob
import sys
import os
np.set_printoptions(formatter={'float': '{: 0.2f}'.format})


def skewm(r):
	return np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])

def expso3(u):
    R = np.identity(3) + np.sinc(np.linalg.norm(u)/np.pi)*skewm(u)\
        + 0.5*(np.sinc(np.linalg.norm(u)/(2*np.pi)))**2 * skewm(u) @ skewm(u)
    return R

def calc_h1h2_rows(u,v):
    A1 = [u[0]*v[0],  
          u[0]*v[1]+u[1]*v[0],
          u[1]*v[1], 
          u[0]*v[2]+u[2]*v[0],
          u[1]*v[2]+u[2]*v[1], 
          u[2]*v[2]]
    A2 = [u[0]*u[0]-v[0]*v[0], 
          2*u[0]*u[1]-2*v[0]*v[1],
          u[1]*u[1]-v[1]*v[1], 
          2*u[0]*u[2]-2*v[0]*v[2],
          2*u[1]*u[2]-2*v[1]*v[2], 
          u[2]*u[2]-v[2]*v[2]]
    return A1, A2

def rc2H(x,xp):
    # Homography H = [r1,r2,t] from point is a plane
    A1 = np.hstack([x[0][0]*skewm(xp[0].T), x[0][1]*skewm(xp[0].T), skewm(xp[0].T)])
    A2 = np.hstack([x[1][0]*skewm(xp[1].T), x[1][1]*skewm(xp[1].T), skewm(xp[1].T)])
    A3 = np.hstack([x[2][0]*skewm(xp[2].T), x[2][1]*skewm(xp[2].T), skewm(xp[2].T)])
    A4 = np.hstack([x[3][0]*skewm(xp[3].T), x[3][1]*skewm(xp[3].T), skewm(xp[3].T)])
    A = np.vstack((A1, A2, A3, A4))
    # Singlar value decomposition
    U,S,V = np.linalg.svd(A)
    # Nullspace solution
    scale = np.sign(V[8,8])*np.linalg.norm(V[8,0:3])
    h1 = V[8,0:3]/scale
    h2 = V[8,3:6]/scale
    h3 = V[8,6:9]/scale
    H = np.zeros((3,3))
    H[:,0] = h1; H[:,1] = h2; H[:,2] = h3
    return H

def pose_estimation(x,xp):
    A1 = np.hstack([x[0][0]*skewm(xp[0].T), x[0][1]*skewm(xp[0].T), skewm(xp[0].T)])
    A2 = np.hstack([x[1][0]*skewm(xp[1].T), x[1][1]*skewm(xp[1].T), skewm(xp[1].T)])
    A3 = np.hstack([x[2][0]*skewm(xp[2].T), x[2][1]*skewm(xp[2].T), skewm(xp[2].T)])
    A4 = np.hstack([x[3][0]*skewm(xp[3].T), x[3][1]*skewm(xp[3].T), skewm(xp[3].T)])
    
    A = np.vstack((A1, A2, A3, A4))
    u,s,v = np.linalg.svd(A)
    h = v[8] 
    scale = np.sign(h[8])*np.linalg.norm(h[0:3])
    r1 = h[0:3]/scale
    r2 = h[3:6]/scale
    r3 = np.cross(r1,r2)
    t = h[6:9]/scale
    H = np.identity(4)
    H[0:3,0] = r1; H[0:3,1] = r2; H[0:3,2] = r3; H[0:3,3] = t
    return H

def Hom2B(H1, H2, H3):
    A1, A2 = calc_h1h2_rows(H1[:,0],H1[:,1])
    A3, A4 = calc_h1h2_rows(H2[:,0],H2[:,1])
    A5, A6 = calc_h1h2_rows(H3[:,0],H3[:,1])
  
    A = np.vstack((A1, A2, A3, A4, A5, A6))
    U,Sigma,Vt = np.linalg.svd(A)
    V = Vt.T
    B = np.zeros((3,3))
    B[0,0] = V[0,5]
    B[0,1] = V[1,5]
    B[1,1] = V[2,5]
    B[0,2] = V[3,5]
    B[1,2] = V[4,5]
    B[2,2] = V[5,5]
    B[1,0] = B[0,1]
    B[2,0] = B[0,2]
    B[2,1] = B[1,2]
    
    B = B/B[2,2];
    return B



def calibrate(path = '../calib_imgs/*.JPG', rows = 6, columns = 9, size = 0.0213, verbose = 2):
   
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


  
    worldPoints = [] # 3d point in real world space
    imagePoints = [] # 2d points in image plane.

    if verbose > 0:
        print("Finding chessboard corners...")
    k = 0
    imgs = []
    for idx, filename in enumerate(images):
        img = cv2.imread(filename)
        if k<3:
            imgs.append(img)
            k += 1

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        
        found, corners = cv2.findChessboardCorners(gray, (rows, columns), None)
       
        if found == True:
            worldPoints.append(worldFrame)
            imagePoints.append(cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria))

            if verbose > 0:
                print("Found corners. %d / %d" % (idx+1, len(images)))
            if verbose > 1:
                img = cv2.drawChessboardCorners(img, (rows, columns), corners, True)
                # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                # plt.show()
        else:
            if verbose > 0:
                print("Could not find corners in image %s. Image %d / %d" % (filename, idx+1, len(images)))


    x1 = []
    xp1 = []
    x2 = []
    xp2 = []
    x3 = []
    xp3 = []
    temp = np.array([0,0,0,1],dtype=float)
    temp2 = np.array([0,0,1],dtype=float)
    itt = np.array([0,6,42,48])
    counter = 0 
    for j in (itt):
        temp2[0] = imagePoints[0][j][0][0]
        temp2[1] = imagePoints[0][j][0][1]
        xp1.append(temp2.copy())

        temp[0] = worldPoints[0][j][0]
        temp[1] = worldPoints[0][j][1]
        x1.append(temp.copy())
        

    for j in (itt):
        temp2[0] = imagePoints[1][j][0][0]
        temp2[1] = imagePoints[1][j][0][1]
        xp2.append(temp2.copy())

        temp[0] = worldPoints[1][j][0]
        temp[1] = worldPoints[1][j][1]
        x2.append(temp.copy())
    for j in (itt):
        temp2[0] = imagePoints[2][j][0][0]
        temp2[1] = imagePoints[2][j][0][1]
        xp3.append(temp2.copy())

        temp[0] = worldPoints[2][j][0]
        temp[1] = worldPoints[2][j][1]
        x3.append(temp.copy())
 
    
    H_1 = rc2H(x1,xp1)
    H_2 = rc2H(x2,xp2)
    H_3 = rc2H(x3,xp3)
 
    B = Hom2B(H_1,H_2,H_3)
    K = np.linalg.inv(np.linalg.cholesky(B))
    # Normalize
    K = K.transpose()/K[2,2]

    #finding pose for the first picture    
    #finding the normalized image coordinates
    for i in range(len(xp1)):
        xp1[i] = np.dot(np.linalg.inv(K),xp1[i].T).copy()
        
    print ('Estiamted kamera parameter matrix:')
    print (K)
    
    T_co_1 = pose_estimation(x1,xp1)
    print("T_co_1: ")
    print(T_co_1)


if __name__ == "__main__":
    calibrate(verbose=1)
import numpy as np


PI = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,0]])

def rot(angle, axis):
    if axis.lower() == 'x':
        return np.array([[1, 0, 0],
                         [0, np.cos(angle), -np.sin(angle)],
                         [0, np.sin(angle), np.cos(angle)]])
    elif axis.lower() == 'y':
        return np.array([[np.cos(angle), 0, np.sin(angle)],
                         [0, 1, 0],
                         [-np.sin(angle), 0, np.cos(angle)]])
    elif axis.lower() == 'z':
        return np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0],
                         [0, 0, 1]])
    else:
        print('INVALID INPUT ROTATION AXIS')
        return np.eye(3)
        

def rotx(angle):
    return np.array([[1, 0, 0],
                     [0, np.cos(angle), -np.sin(angle)],
                     [0, np.sin(angle), np.cos(angle)]])
    
def roty(angle):
    return np.array([[np.cos(angle), 0, np.sin(angle)],
                     [0, 1, 0],
                     [-np.sin(angle), 0, np.cos(angle)]])
    
def rotz(angle):
    return np.array([[np.cos(angle), -np.sin(angle), 0],
                     [np.sin(angle), np.cos(angle), 0],
                     [0, 0, 1]])

def rot2D(angle):
    return np.array([[np.cos(angle), -np.sin(angle)],
                     [np.sin(angle), np.cos(angle)]])
    
def skew(v):
    v = v.flat
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])
    
def expso3(u):
    S = skew(u)
    un = np.linalg.norm(u)
    return np.eye(3) + np.sinc(un/np.pi)*S + 0.5*(np.sinc(un/(2*np.pi)))**2 * S@S
    
def pixl2img(p, K):
    s = np.zeros(p.shape())
    k_inv = np.linalg.inv(K)
    for i in range(len(p)):
        s[i] = k_inv @ p[i]
    return s

def img2pixl(s, K):
    p = np.zeros(s.shape())
    for i in range(len(p)):
        p[i] = K @ s[i]
    return p


def print_image_and_pixel_coordinates(x, K):
    s = (1/x[2])*x
    p = K@s
    print("s:", s,"\np:", p)
    
def matrix_eq(m1, m2, margin=0.0001):
    for i in range(len(m1)):
        for j in range(len(m1[i])):
            if m1[i,j] < m2[i,j] - margin or m1[i,j] > m2[i,j] + margin:
                return False
    return True

def matrix_max_diff(m1, m2):  # Naive estimation of precision
    max_diff = 0              # returns the largest element-wise diff
    for i in range(len(m1)):
        for j in range(len(m1[i])):
            if abs(m1[i,j] - m2[i,j]) > max_diff:
                max_diff = abs(m1[i,j] - m2[i,j])
    return max_diff

def matrix_mse(m1, m2): # returns the mean-square error between two matrices
    mse = (np.square(m1 - m2)).mean()
    return mse


def sdv_homogenous_line(A):
    '''
    Takes a matrix A on the form A=[[x1, y1, 1], ..., [xn, yn,1]] and returns a line on homogenous form
    @param:
        A = [[x1, x2, ..., xn],
             [y1, y2, ..., yn],
             [1,   1, ..., 1 ]].T
    returns:
        line, on the form line=[a, b, c] which corresponds to ax+by+c=0
    '''
    _, _, vt = np.linalg.svd(A)
    return vt[-1, :]


def find_inliers_outliers(line, homogenous_points, delta):
    '''
    Takes a matrix of homogenous points and a homogenous line and returns the inliers and outliers.
    Used to implement a RANSAC algorithm
    param:
        homogenous_points: matrix on the form A = [[x1, x2, ..., xn],
                                                   [y1, y2, ..., yn],
                                                   [1,   1, ..., 1 ]].T
        line: a line on the form [a, b, c] which corresponds to ax+by+c=0
        delta: maximum distance between line and point
    returns:
        inliers: points inside delta on the form of homogenous_points
        outliers: points outside delta on the form of homogenous_points
    '''
    inliers = np.array(np.zeros(3))
    outliers = np.array(np.zeros(3))

    for i in range(len(homogenous_points)):
        if ((line @ homogenous_points[i]) / np.linalg.norm(line[:2])) <= delta:
            inliers = np.block([[inliers], [homogenous_points[i]]])
        else:
            outliers = np.block([[outliers], [homogenous_points[i]]])
    
    return inliers[1:], outliers[1:]



if __name__ == "__main__":
    k = 1500
    u0 = 640
    v0 = 512

    K = np.array([[k, 0, u0],
                  [0, k, v0],
                  [0, 0, 1]])
    
    x1 = np.array([0.1,0.2,0.5])
    print_image_and_pixel_coordinates(x1, K)
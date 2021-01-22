import numpy as np

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
    
def scew(M):
    return np.array([[0, -M[2], M[1]],
                     [M[2], 0, -M[0]],
                     [-M[1], M[0], 0]])

def sdv_homogenous_line(A):
    '''
    Takes a matrix A on the form A=[[x1, y1, 1], ..., [xn, yn,1]] and returns a line on homogenous form
    param:
        A = [[x1, x2, ..., xn],
             [y1, y2, ..., yn],
             [1,   1, ..., 1 ]].T
    returns:
        line, on the form line=[a, b, c] which corresponds to ax+by+c=0
    '''
    u, s, vt = np.linalg.svd(A)
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

    for i in range(len(A)):
        if ((line @ A[i]) / np.linalg.norm(line[:2])) <= delta:
            inliers = np.block([[inliers], [A[i]]])
        else:
            outliers = np.block([[outliers], [A[i]]])
    
    return inliers[1:], outliers[1:]
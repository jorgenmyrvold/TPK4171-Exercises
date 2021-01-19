import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=8, suppress=True)
# np.set_printoptions(formatter={'float': '{: 0.2f}'.format})


A = np.array([[0, 1, 2, 3, 5, 5.5],
              [1, 2, 3.1, 4.1, 5.5, 6.4],
              [1, 1, 1, 1, 1, 1]]).T

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
    Takes a matrix of homogenous points and a homogenous line and returns the inliers and outliers
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

def task_a():    
    u, s, vt = np.linalg.svd(A)
    line = vt[-1, :]
    
    x = np.linspace(0, 6, 100)
    y = -1/line[1] * (line[0] * x + line[2])
    
    plt.scatter(A[:,0], A[:,1], label='Data points', color='C0')
    plt.plot(x, y, label='Regression', color='C1')
    plt.legend()
    plt.grid(ls='--')
    plt.show()


def task_b(delta=0.1):
    line = sdv_homogenous_line(A[:2])
    x = np.linspace(0, 6, 100)
    y = -1/line[1] * (line[0] * x + line[2])
    
    inliers, outliers = find_inliers_outliers(line, A, delta)

    plt.scatter(inliers[:,0], inliers[:,1], label='inliers', color='C2')
    plt.scatter(outliers[:,0], outliers[:,1], label='outliers', color='C1', marker='X')
    plt.plot(x, y, label='Regression', color='C0')
    plt.legend()
    plt.grid(ls='--')
    plt.show()

def task_c(delta=0.1):
    line_all = sdv_homogenous_line(A)
    x = np.linspace(0, 6, 100)
    y_all = -1/line_all[1] * (line_all[0] * x + line_all[2])
    
    inliers, outliers = find_inliers_outliers(line_all, A, delta)
    line_innliers = sdv_homogenous_line(inliers)
    y_inliers = -1/line_innliers[1] * (line_innliers[0] * x + line_innliers[2])
    
    plt.scatter(inliers[:,0], inliers[:,1], label='inliers', color='C2')
    plt.scatter(outliers[:,0], outliers[:,1], label='outliers', color='C3', marker='X')
    plt.plot(x, y_all, label='Regression - All points', color='C0')
    plt.plot(x, y_inliers, label='Regression - Inliers', color='C1')
    plt.legend()
    plt.grid(ls='--')
    plt.show()


if __name__ == "__main__":
    # task_a()
    # task_b()
    task_c()
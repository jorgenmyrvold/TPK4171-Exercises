import numpy as np
np.set_printoptions(formatter={'float': '{: 0.0f}'.format})

def matrix_eq(m1, m2, margin=0.0000000000001):
    for i in range(len(m1)):
        for j in range(len(m1[i])):
            if m1[i,j] < m2[i,j] - margin or m1[i,j] > m2[i,j] + margin:
                return False
    return True

x = np.array([[0, 0, 1],
              [0.1, 0, 1],
              [0.1, 0.1, 1],
              [0, 0.1, 1],
              [0.2, -0.1, 1],
              [0.15, 0.1, 1],
              [-0.1, 0.3, 1],
              [-0.2, 0.1, 1],
              [-0.2, 0, 1]]).T

xp = np.array([[960, 540, 1],
               [1320, 540, 1],
               [1320, 900, 1],
               [960, 900, 1],
               [1680, 180, 1],
               [1500, 900, 1],
               [600, 1620, 1],
               [240, 900, 1],
               [240, 540, 1]]).T

def find_homographie(x, xp):
    num_points = x.shape[1]
    zzz = np.zeros((num_points,3))
    
    A = np.block([[x.T, zzz, -x.T*xp[0].reshape(num_points,1)], 
                [zzz, x.T, -x.T*xp[1].reshape(num_points,1)]])
    _, _, v = np.linalg.svd(A)
    
    h = v[-1]/v[-1,-1]
    H = np.block([[h[0:3]], [h[3:6]], [h[6:9]]]) 
    return H

if __name__ == '__main__':
    H = find_homographie(x, xp)
    print('\n------------ Problem 2 ------------ \nH = \n{}'.format(H))
    
    # Test to see if the homographie is correct
    print('xp and H @ x, are equal: {}'.format(matrix_eq(xp, H @ x, margin=0.000001)))
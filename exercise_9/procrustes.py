import numpy as np
np.set_printoptions(formatter={'float': '{: 0.4f}'.format})

def rotz(angle):
    return np.array([[np.cos(angle), -np.sin(angle), 0],
                     [np.sin(angle), np.cos(angle), 0],
                     [0, 0, 1]])

def procrustes_book(A, B): # Such that A = R @ B
    H = B @ A.T
    U, _, Vt = np.linalg.svd(H)
    # R = Vt.T @ U.T   # Usual algorithm
    R = Vt.T @ np.diag([1,1,np.linalg.det( Vt.T @ U.T)]) @ U.T  # Prevents reflection matix
    
    if np.linalg.det(R) < 0: # Notify if reflection
        print('Reflection')
    return R


if __name__ == '__main__':
    # Task 1a
    b = np.array([[1,0,0], [1,1,0]]).T
    a = np.array([[0.5, 0.866, 0], [-0.25, 1.299, 0.5]]).T
    R = procrustes_book(a,b)
    # print(R)
    
    
    # Task 1b
    b = np.array([[1,0,0.001], [0,1,0]]).T
    Rz = rotz(np.pi/2)
    n = np.array([[0,0,-0.01],[0,0,0]]).T
    a = Rz@b + n
    R = procrustes_book(a, b)
    print(R)
    

import numpy as np
np.set_printoptions(formatter={'float': '{: 0.4f}'.format})


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

def rotz(angle):
    return np.array([[np.cos(angle), -np.sin(angle), 0],
                     [np.sin(angle), np.cos(angle), 0],
                     [0, 0, 1]])

#From book
def procrustes(A, B, allow_reflect=False):
    '''
    Returns the optimal orthogonal or special orthogonal matrix R such that A = R @ B
    param:
        A: Measured coordinates
        B: World frame coordinates
        allow_reflect: If false it returns R in SO(3) if false it might ret
    Algorithm from book ch. 
    '''
    H = B @ A.T
    u, _, v = np.linalg.svd(H)
    R = v.T @ np.diag([1,1,np.linalg.det(v.T @ u.T)]) @ u.T

    if allow_reflect:
        Rr = v.T @ u.T
        if matrix_mse(A, Rr@B) < matrix_mse(A, R@B):
            print('Reflection matrix')
            return Rr
    return R




if __name__ == '__main__':
    B = np.array([[1,0,0.01],
                  [1,1,0],
                  [0,1,0]]).T
    A = np.array([[0.5, 0.866, -0.001],
                  [-0.366, 1.366, 0],
                  [-0.866, 0.5, 0]]).T
    
    R_reflect = procrustes(A, B, allow_reflect=True)
    print('det(R) = {:.4f}\n{}'.format(np.linalg.det(R_reflect), R_reflect))
    
    R = procrustes(A, B, allow_reflect=False)
    print('\ndet(R) = {:.4f}\n{}\n'.format(np.linalg.det(R), R))
    
    new_A_reflect = R_reflect @ B
    new_A = R @ B
    
    print('A     ', A.flatten())
    print('new_Ar', new_A_reflect.flatten())
    print('new_A ', new_A.flatten())
    
    print('\nA = new_A         error: {:.6f}'.format(matrix_max_diff(A, new_A)))
    print('A = new_A_reflect error: {:.6f}'.format(matrix_max_diff(A, new_A_reflect)))
    
    print('\nA = new_A         mse: {:.10f}'.format(matrix_mse(A, new_A)))
    print('A = new_A_reflect mse: {:.10f}'.format(matrix_mse(A, new_A_reflect)))
    

import numpy as np
np.set_printoptions(formatter={'float': '{: 0.0f}'.format})

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

# https://stackoverflow.com/questions/18925181/procrustes-analysis-with-numpy
def procrustes(X, Y, scaling=False, reflection='best'): # Rotation and translation
    """
    A port of MATLAB's `procrustes` function to Numpy.

    Procrustes analysis determines a linear transformation (translation,
    reflection, orthogonal rotation and scaling) of the points in Y to best
    conform them to the points in matrix X, using the sum of squared errors
    as the goodness of fit criterion.

        d, Z, [tform] = procrustes(X, Y)

    Inputs:
    ------------
    X, Y    
        matrices of target and input coordinates. they must have equal
        numbers of  points (rows), but Y may have fewer dimensions
        (columns) than X.

    scaling 
        if False, the scaling component of the transformation is forced
        to 1

    reflection
        if 'best' (default), the transformation solution may or may not
        include a reflection component, depending on which fits the data
        best. setting reflection to True or False forces a solution with
        reflection or no reflection respectively.

    Outputs
    ------------
    d       
        the residual sum of squared errors, normalized according to a
        measure of the scale of X, ((X - X.mean(0))**2).sum()

    Z
        the matrix of transformed Y-values

    tform   
        a dict specifying the rotation, translation and scaling that
        maps X --> Y

    """

    n,m = X.shape
    ny,my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0**2.).sum()
    ssY = (Y0**2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m-my)),0)

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U,s,Vt = np.linalg.svd(A,full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection != 'best':

        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0

        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:,-1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)

    traceTA = s.sum()

    if scaling:

        # optimum scaling of Y
        b = traceTA * normX / normY

        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA**2

        # transformed coords
        Z = normX*traceTA*np.dot(Y0, T) + muX

    else:
        b = 1
        d = 1 + ssY/ssX - 2 * traceTA * normY / normX
        Z = normY*np.dot(Y0, T) + muX

    # transformation matrix
    if my < m:
        T = T[:my,:]
    c = muX - b*np.dot(muY, T)
    
    # Notify if reflection
    if np.linalg.det(T) < 0:
        print('Reflection')
    
    #transformation values 
    tform = {'rotation':T, 'scale':b, 'translation':c}
   
    return d, Z, tform


#From book - pure rotation
def procrustes_book(A, B, allow_reflect=False):
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
                  [0,1,0]])
    A = np.array([[0.5, 0.866, -0.001],
                  [-0.366, 1.366, 0],
                  [-0.866, 0.5, 0]])
    
    d, Z, tform = procrustes(A, B)
    
    R = tform['rotation']
    t = tform['translation']
    b = tform['scale']
    
    print('scaling: {}'.format(b))
    print('translation: {}'.format(t))
    print('Rotation: \n{}'.format(R))
    
    T = np.block([[R, t.reshape((3,1))],
                  [0,0,0,1]])
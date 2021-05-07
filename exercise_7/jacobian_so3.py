import numpy as np
np.set_printoptions(suppress=True, precision=8)

def skew(M):
    return np.array([[0, -M[2], M[1]],
                     [M[2], 0, -M[0]],
                     [-M[1], M[0], 0]])
    

def weks(M): # From skew symetric matrix to vector
    if M[0,1] != -M[1,0] or M[0,2] != -M[2,0] or M[1,2] != -M[2,1]:
        print('ERROR: Matrix is not skew symetric')
    return np.array([M[2,1], M[0,2], M[1,0]])


def jacobian(u, t, left): # Eq. 1301, 1302
    u = skew(u)
    if left:
        return np.eye(3) + ((1-np.cos(t))/t**2)*u + ((t-np.sin(t))/t**3)*u**2
    else:
        return np.eye(3) - ((1-np.cos(t))/t**2)*u + ((t-np.sin(t))/t**3)*u**2


def jacobian_inv(u, t, left): # Eq. 1303, 1304
    u = skew(u)
    if left:
        return np.eye(3) - (1/2)*u + ((1-(t/2)*(1/np.tan(t/2)))/(t**2))*u**2
    else:
        return np.eye(3) + (1/2)*u + ((1-(t/2)*(1/np.tan(t/2)))/(t**2))*u**2
    
    
    
    
if __name__ == "__main__":
    u = np.array([1,0,0])
    t = np.pi/4
    left_j = jacobian(u, t, left=True)
    right_j = jacobian(u, t, left=False)
    left_j_inv = jacobian_inv(u, t, left=True)
    right_j_inv = jacobian_inv(u, t, left=False)

    print('Left jacobian: \n{}\n\nRight jacobian: \n{}'. format(left_j, right_j))
    print('\nLeft inverse jacobian: \n{}\n\nRight inverse jacbian: \n{}'.format(left_j_inv, right_j_inv))
    
    print("test:\n{}\n{}".format(left_j@left_j_inv, right_j@right_j_inv))
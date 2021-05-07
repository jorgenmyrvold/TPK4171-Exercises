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


def log2rot(o, t): # o: omega [x,y,z], t: theta int
    o = skew(o)
    return np.eye(3) + np.sin(t)*o + (1-np.cos(t))*(o @ o) # Rodrigues formula

    
def rot2log(R): # Rotation matrix R
    if (R == np.eye(3)).all():
        print('No rotation: Rotation axis is undefined')
        return
    if np.trace(R) == -1:
        o = 1/(np.sqrt(2*(1+R[2,2]))) * np.array([R[0,2], R[1,2], 1+R[2,2]])
        return np.pi, o
    else:
        t = np.arccos(1/2 * (np.trace(R)-1))
        o_skew = 1/(2*np.sin(t)) * (R - R.T)
        o = weks(o_skew)
        return t, o


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
    # Logarithm and exponetial 
    print("------ LOG AND EXP ------")
    t = 0.001
    o = np.array([1,0,0])
    R = log2rot(o, t)
    t_new, o_new = rot2log(R)
    print('R:\n',R)
    print('\nt: {} \no: {}'.format(t_new, o_new))
    
    # Calculation of left and right jacobian and their inverse
    print("\n\n------ JACOBIAN ------")
    u = np.array([1,0,0])
    t = np.pi/4
    left_j = jacobian(u, t, left=True)
    right_j = jacobian(u, t, left=False)
    left_j_inv = jacobian_inv(u, t, left=True)
    right_j_inv = jacobian_inv(u, t, left=False)

    print('Left jacobian: \n{}\n\nRight jacobian: \n{}'. format(left_j, right_j))
    print('\nLeft inverse jacobian: \n{}\n\nRight inverse jacbian: \n{}'.format(left_j_inv, right_j_inv))
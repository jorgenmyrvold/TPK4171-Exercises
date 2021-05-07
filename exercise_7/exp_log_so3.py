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



if __name__ == "__main__":
    t = 0.001
    o = np.array([1,0,0])
    R = log2rot(o, t)
    t_new, o_new = rot2log(R)
    print('R:\n',R)
    print('\nt: {} \no: {}'.format(t_new, o_new))
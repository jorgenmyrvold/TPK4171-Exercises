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
    

def log2rot(k, t): # o: omega [x,y,z], t: theta int
    k = skew(k)
    return np.eye(3) + np.sin(t)*k + (1-np.cos(t))*(k @ k) # Rodrigues formula

def expSO3(u): # From book
    return np.identity(3) + np.sinc(np.linalg.norm(u)/np.pi)*skew(u) + 0.5*(np.sinc(np.linalg.norm(u)/(2*np.pi)))**2*skew(u)@skew(u)


def rot2log(R, margin=0.0000001): # Rotation matrix R
    if (R == np.eye(3)).all():
        print('No rotation: Rotation axis is undefined')
        return
    if np.trace(R)-margin < -1 < np.trace(R)+margin:
        k = 1/(np.sqrt(2*(1+R[2,2]))) * np.array([R[0,2], R[1,2], 1+R[2,2]])
        return np.pi, k
    else:
        t = np.arccos(1/2 * (np.trace(R)-1))
        k_skew = 1/(2*np.sin(t)) * (R - R.T)
        k = weks(k_skew)
        return t, k

def logSO3(R):  # From LF, does not separate theta and k
    # The vector form of the logarithm in SO(3) (Iserles, 2006)
    eh = 0.5*(R-R.T) # eh = sin(theta)k^skew
    en = np.linalg.norm(weks(eh)) # en = |sin(theta)|
    if en < 0.000001:
        g = 1+ (en**2)/6
    else:
        g = (np.arcsin(en)/en)
    return weks(g*eh)


if __name__ == "__main__":
    t = 0.32
    k = np.array([np.sqrt(2),np.sqrt(2),0])
    R = log2rot(k, t)
    t_new, k_new = rot2log(R)
    u = logSO3(R)
    print('R:\n',R)
    print('\nt: {:.13f} \nk: {}'.format(t_new, k_new))
    print('u: {}'.format(u))
    
    R = rot(0.32, 'x')
    t, k = rot2log(R)
    new_R = log2rot(k, t)
    print('\n\nR:\n{}'.format(R))
    print('\nt: {:.13f} \nk: {}'.format(t, k))
    print('new_R:\n{}'.format(new_R))
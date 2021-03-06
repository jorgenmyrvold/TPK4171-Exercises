import numpy as np
from scipy.linalg.lapack import HAS_ILP64
np.set_printoptions(formatter={'float': '{: 0.4f}'.format})

def rotx(angle):
    return np.array([[1, 0, 0],
                     [0, np.cos(angle), -np.sin(angle)],
                     [0, np.sin(angle), np.cos(angle)]])
    
def rotz(angle):
    return np.array([[np.cos(angle), -np.sin(angle), 0],
                     [np.sin(angle), np.cos(angle), 0],
                     [0, 0, 1]])
    
def skew(v):
    return np.array([[0, -v[2], v[1]],
                    [v[2], 0, -v[0]],
                    [-v[1], v[0], 0]])

def ro2rc(ro_h, Toc): # Task 1a: Transform ro to rc
    rc_h = np.zeros(ro_h.shape)
    for i in range(len(ro_h)):
        rc_h[i] = Toc @ ro_h[i]
        
    print("\n--------- Problem 1a ---------\nrc = \n{}".format(rc_h[:, :-1]))
    return rc_h[:, :-1] # Returns non-homogenous rc

def rc2s(rc): # Task 1b: Transform rc to normalized image coordinates s
    s = np.zeros(rc.shape)
    for i in range(len(rc)):
        s[i] = rc[i]/rc[i,-1]
        
    print("\n--------- Problem 1b ---------\ns = \n{}".format(s))
    return s
    
def pose_estimation(ro, s):
    A1 = np.hstack([ro[0][0]*skew(s[0]), ro[0][1]*skew(s[0]), skew(s[0])]);
    A2 = np.hstack([ro[1][0]*skew(s[1]), ro[1][1]*skew(s[1]), skew(s[1])]);
    A3 = np.hstack([ro[2][0]*skew(s[2]), ro[2][1]*skew(s[2]), skew(s[2])]);
    A4 = np.hstack([ro[3][0]*skew(s[3]), ro[3][1]*skew(s[3]), skew(s[3])]);
    A = np.vstack((A1, A2, A3, A4))

    _, _, v = np.linalg.svd(A)
    h = v[8]

    scale = np.sign(h[8])*np.linalg.norm(h[0:3])
    r1 = h[0:3]/scale
    r2 = h[3:6]/scale
    r3 = np.cross(r1,r2)
    t = h[6:9]/scale

    H = np.block([[r1.reshape((3,1)), r2.reshape((3,1)), r3.reshape((3,1)), t.reshape((3,1))],
                  [0,0,0,1]])

    print ('Estimated pose:')
    print("\n--------- Problem 1c ---------\nEstimated pose:\nH = \n{}".format(H))
    return H
    
if __name__ == '__main__':
    PI = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,0]])
    
    t = np.array([[0, 0.2, 3]]).T
    Toc = np.block([[rotx((2*np.pi)/3)@rotz(np.pi/6), t],
                    [0,0,0,1]])
    
    ro = np.array([[0,0,0],
                   [1,0,0],
                   [1,1,0],
                   [0,1,0]])
    
    ro_h = np.block([ro, np.ones((len(ro),1))])  # Create homogenous coordinates
    
    print("\nro = \n{}".format(ro))
    rc= ro2rc(ro_h, Toc)    
    s = rc2s(rc)
    
    # Task c: new s
    s = np.array([[0,0,1],
                  [0.5, 0, 1],
                  [0.34, -0.116, 1],
                  [0, -0.116, 1]])
    H = pose_estimation(ro_h, s) # With pixelcoordinates and calibrated camera s = np.linalg.inv(K) @ p
    
    
    # Check that new_s calculated with the Homographie
    # is equal to the original s
    new_s = np.zeros(ro_h.shape)
    for i in range(len(ro_h)):
        new_s[i] = PI @ H @ ro_h[i]
        new_s[i] = new_s[i]/new_s[i,2]
    new_s = new_s[:,:-1]
    print('\n{}'.format(s.flatten()))
    print(new_s.flatten())
    
    
    
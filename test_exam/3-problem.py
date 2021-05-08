import numpy as np
np.set_printoptions(formatter={'float': '{: 0.4f}'.format})

def skew(v):
    v = v.flat
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

def roty(angle):
    return np.array([[np.cos(angle), 0, np.sin(angle)],
                     [0, 1, 0],
                     [-np.sin(angle), 0, np.cos(angle)]])


def find_essential_matrix(t, R):
    E = skew(t) @ R
    print('E = \n{}'.format(E))
    return E
    

if __name__ == '__main__':
    R01 = roty(np.pi/12)
    t0_01 = np.array([[-0.5,0,0]]).T
    T01 = np.block([[R01, t0_01],
                    [0,0,0,1]])
    
    R02 = roty(-np.pi/12)
    t0_02 = np.array([[0.5,0,0]]).T
    T02 = np.block([[R02, t0_02],
                    [0,0,0,1]])
    
    R = R02.T @ R01
    t2_01 = R02.T @ t0_01
    t2_02 = R02.T @ t0_02
    t2_21 = t2_01 - t2_02
    
    print('R = \n', R)
    print('t2_21 = {}'.format(t2_21.flatten()))
    
    E = find_essential_matrix(t2_21, R)
    
    r00 = np.array([[0.1, 0.1, 1/(2*np.tan(np.pi/12)), 1]]).T
    r2_2p = np.linalg.inv(T02) @ r00
    s2_2p = r2_2p[:3]/r2_2p[2]
    print('\nr2_00 = {}'.format(r2_2p.flatten()))
    print('s2_00 = {}'.format(s2_2p.flatten()))

    r1_1p = np.linalg.inv(T01) @ r00
    s1_1p = r1_1p[:3]/r1_1p[2]
    print('r1_00 = {}'.format(r1_1p.flatten()))
    print('s1_00 = {}'.format(s1_1p.flatten()))
    
    l2 = E @ s1_1p
    l1 = E.T @ s2_2p
    print('\nl1 = {}'.format(l1.flatten()))
    print('l2 = {}'.format(l2.flatten()))
    
    
    

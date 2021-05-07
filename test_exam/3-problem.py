import numpy as np
np.set_printoptions(formatter={'float': '{: 0.4f}'.format})

def roty(angle):
    return np.array([[np.cos(angle), 0, np.sin(angle)],
                     [0, 1, 0],
                     [-np.sin(angle), 0, np.cos(angle)]])


if __name__ == '__main__':
    R01 = roty(np.pi/12)
    t0_01 = np.array([[-0.5,0,0]]).T
    T01 = np.block([[R01, t0_01],[0,0,0,1]])
    
    R02 = roty(-np.pi/12)
    t0_02 = np.array([[0.5,0,0]]).T
    T02 = np.block([[R02, t0_02],[0,0,0,1]])
    
    R = R02.T @ R01
    t2_01 = R02 @ t0_01
    t2_02 = R02 @ t0_02
    t2_21 = t2_02-t2_01
    
    print(R)
    print(t2_21)

    
    

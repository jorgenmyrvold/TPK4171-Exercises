import numpy as np
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})


def K_from_B(B): # Using algorithm in 14.6
    v0 = (B[0,1]*B[0,2] - B[0,0]*B[1,2]) / (B[0,0]*B[1,1] - B[0,1]**2)
    lambda_ = B[2,2] - (B[0,2]**2 + v0*B[0,1]*B[0,2] - B[0,0]*B[1,2])/(B[0,0])
    k1 = np.sqrt(lambda_/B[0,0])
    k2 = np.sqrt((lambda_*B[0,0]) / (B[0,0]*B[1,1] - B[0,1]**2))
    s = -(B[0,1]*k1*k1*k2)/lambda_
    u0 = (s*v0)/k1 - (B[0,2]*k1*k1)/lambda_
    
    K = np.array([[k1, s, u0], [0, k2, v0], [0, 0, 1]])
    return K

if __name__ == '__main__':
    K = np.array([[1500, 0.3, 640], [0, 1490, 512], [0, 0, 1]])
    B = np.linalg.inv(K @ K.T);
    Kc = K_from_B(B)
    print(K.flatten())
    print(Kc.flatten())
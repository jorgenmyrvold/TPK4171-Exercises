import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})


def rotx(angle):
    return np.array([[1, 0, 0],
                     [0, np.cos(angle), -np.sin(angle)],
                     [0, np.sin(angle), np.cos(angle)]])


def residual_transfer_error(hh): # Used for optimization
    Hh = hh.reshape(3,3)
    xph = Hh @ x
    ph = xph[0:2]/xph[2,:]
    residual = p - ph
    return residual.flatten()


def find_homographie_dlt(x, xp):
    num_points = x.shape[1]
    zzz = np.zeros((num_points,3))
        
    A = np.block([[x.T, zzz, -x.T*xp[0].reshape(num_points,1)], 
                  [zzz, x.T, -x.T*xp[1].reshape(num_points,1)]])
    _, _, v = np.linalg.svd(A)
    
    h = v[-1]/v[-1,-1]
    H = np.block([[h[0:3]], [h[3:6]], [h[6:9]]]) 
    return H


def find_homographie_opt(x,xp):
    # Optimization of transfer error with DLT solution as initial value
    Hdlt = find_homographie_dlt(x,xp)

    h0 = Hdlt.copy().flatten()
    res_1 = least_squares(residual_transfer_error, h0)
    hopt = res_1.x
    Hopt = hopt.reshape(3,3)
    Hopt = Hopt/Hopt[2,2]
    return Hopt


# def find_homographie_opt(x,xp): # With some optimality measurements etc...
#     # Optimization of transfer error with DLT solution as initial value
#     Hdlt = find_homographie_dlt(x,xp)

#     H0 = Hdlt.copy()
#     h0 = H0.flatten()
#     r0 = residual_transfer_error(h0)
#     d_DLT = np.dot(r0,r0)
#     res_1 = least_squares(residual_transfer_error, h0)
#     hopt = res_1.x
#     r_opt = residual_transfer_error(hopt)
#     d_opt = np.dot(r_opt,r_opt)
#     Hopt = hopt.reshape(3,3)
#     Hopt = Hopt/Hopt[2,2]
#     return Hopt, r_opt, d_opt, res_1, d_DLT


if __name__ == '__main__':
    K = np.array([[1500, 0, 640], 
                [0, 1500, 512], 
                [0, 0, 1]])
    t = np.array([[0,0,2]]).T
    R = rotx(np.deg2rad(120))
    T = np.block([[R, t], 
                [0,0,0,1]])
    
    Ha = K @ np.block([R[:,0:2], t])  # Actual H from LF

    # Points in object frame                        
    x = np.array([[0, 0, 1],
                [0, 1, 1],
                [0, 2, 1],
                [0, 3, 1],
                [0.5, 0.5, 1],
                [-0.5, 0.5, 1],
                [-1, 1, 1],
                [-1, 2, 1],
                [-1, 3, 1]]).T
    # Points in pixel coordinates
    xp = np.array([[637, 509, 1],
                   [639, 251, 1],
                   [641, 108, 1],
                   [638, 21, 1],
                   [951, 361, 1],
                   [331, 358, 1],
                   [119, 252, 1],
                   [240, 110, 1],
                   [314, 25, 1]]).T
    p = xp[:2]

    Hdlt = find_homographie_dlt(x,xp)
    Hopt = find_homographie_opt(x, xp)  
    print('H:    {}'.format((Ha/2).flatten()))
    print('Hdlt: {}'.format(Hdlt.flatten()))
    print('Hopt: {}'.format(Hopt.flatten()))
    
    
    # Hdlt = find_homographie_dlt(x,xp)
    # Hopt, r_opt, d_opt, res_1, d_DLT = find_homographie_opt(x, xp)  # Dependent on p = xp[:2]
    # print('\n optimality = {:.9f}'.format(res_1.optimality))
    # print('\n Hopt = \n {} \n\n Hdlt = \n {}\n\n Ha = \n {} \n\n d_DLT = {:.9f} \n d_opt = {:.9f}'.format(Hopt/Hopt[2,2], Hdlt, Ha/Ha[2,2], d_DLT, d_opt))

    print('\n\n---- TEST ----')
    print('xp\t\t Hdlt@x\t\t   Hopt@x')
    for i in range(x.shape[1]):
        print('{} | {}\t| {}'.format(xp[:,i], Hdlt@x[:,i], Hopt@x[:,i]))


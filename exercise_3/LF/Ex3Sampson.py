import numpy as np
from scipy.optimize import least_squares

def residual_Sampson(h):
    # Calculate residual array
    H = h.reshape(3,3)
    J = Sampson_jacobian(H)
    E = algebraic_error(H)
    N = x.shape[1]   # x is a global variable
    res = np.zeros((4, N))
    for i in range(0, N):  # i iterates over point correspondences
        Ei = np.block([[E[i]], [E[i+N]]])
        Ji = np.block([[J[i]], [J[i+N]]])
        Mi = np.linalg.inv(Ji@Ji.T)
        res[:,i] = (Ji.T @ Mi @ Ei).reshape(4,)
    return res.flatten()
    
def Sampson_jacobian(H):
    h3 = H[2]; hv3 = h3.reshape(3,1)
    N = x.shape[1]  # x and xp are global variables
    one_v = np.ones((N,1))
    J = np.block([[H[0,0]*one_v - xp[0].reshape(N,1)*H[2,0] , 
                  H[0,1]*one_v - xp[0].reshape(N,1)*H[2,1] , 
                  - x.T@hv3, np.zeros((N,1))],
                  [H[1,0]*one_v - xp[1].reshape(N,1)*H[2,0] , 
                  H[1,1]*one_v - xp[1].reshape(N,1)*H[2,1] , 
                  np.zeros((N,1)), - x.T@hv3]])
    return J

def algebraic_error(H):
    h1 = H[0]; h2 = H[1]; h3 = H[2]
    hv3 = h3.reshape(3,1)
    N = x.shape[1]  # x and xp are global variables
    eps = np.block([[(x.T@h1).reshape(N,1) - xp[0].reshape(N,1)*x.T@hv3],
                     [(x.T@h2).reshape(N,1) - xp[1].reshape(N,1)*x.T@hv3]])
    return eps

# True homography
H0 = np.array([1., 0., 1., 0, 2., 1, 1, 1, 2.]).reshape(3,3)
#H0 = np.array([1500., 0., 640., 0, 1500., 512, 0, 0, 1.]).reshape(3,3)
x = np.array([[0.,0.,1.], [1.,0.,1.], [1.,1.,1.], [0.,1.,1.],
              [2.,-1.,1.], [-1.,3.,1.], [4.,1.,1.]]).T
    # Point correspondences
xp = H0 @ x
xp = xp/xp[2,:] # Normalization of xp

# Initial homography
h0_Sampson = np.array([1, 0., 0., 0, 1., 0, 0, 0, 1.])
# Optimization of H
res_1 = least_squares(residual_Sampson, h0_Sampson)

np.set_printoptions(formatter={'float': '{: 0.9f}'.format})
H = res_1.x.reshape(3,3); H = H/H[2,2]*H0[2,2]
print('\n H = \n', H)
print('\n cost = ', res_1.cost)
print('\n optimality = ', res_1.optimality)
print('\n H0 = \n', H0)




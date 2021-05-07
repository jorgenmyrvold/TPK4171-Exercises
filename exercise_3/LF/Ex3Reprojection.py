import numpy as np
from scipy.optimize import least_squares

def fun_reprojection(z):
    # Calculate residual vector
    nz = z.shape[0]; nx = int((nz - 9)/2); 
    hhat = z[0:9]; xhat = z[9:nz].reshape(2,nx)
    Hhat = hhat.reshape(3,3)
    xphat_h = Hhat @ np.block([[xhat], [np.ones(nx)]]); 
    xphat_h = xphat_h / xphat_h[2,:];  xphat = xphat_h[0:2,:]
    xx = xa - xhat; xxhat = xap - xphat
    return np.block([xx.flatten(), xxhat.flatten()])

# True homography
ha = np.array([1., 0., 1., 0, 2., 1, 0, 0, 2.])
Ha = ha.reshape(3,3)
xa_h = np.array([[0.,0.,1.], [1.,0.,1.], [1.,1.,1.], [0.,1.,1.],
              [2,-1,1], [-1,3,1], [4,1,1]]).T
# Point correspondences
xap_h = Ha @ xa_h
xap_h = xap_h/xap_h[2,:] # Normalization of xp
xa = xa_h[0:2,:]; xap = xap_h[0:2,:]

# Initial homography and point
x0 = xa; h0 = np.eye(3,3).flatten()
z0_reprojection = np.block([h0,  x0.T.flatten()])

# Optimization of H
res_1 = least_squares(fun_reprojection, z0_reprojection)

np.set_printoptions(formatter={'float': '{: 0.4f}'.format})
z = res_1.x
H = z[0:9].reshape(3,3); H = H/H[2,2]*Ha[2,2]
print('\n H = \n', H)
xh = z[9:z.shape[0]].reshape(xa.shape[0], xa.shape[1])
print('\n xh =\n', xh)
print('\n optimality = ', res_1.optimality)
print('\n Ha = \n', Ha)




import numpy as np
from scipy.optimize import least_squares
from numpy import random
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})


def Skew(r):
	S = np.array([[ 0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0],  0]])
	return S

def fun_reprojection(z):
    # Calculate residual vector
    nz = z.shape[0]; nx = int((nz - 9)/2); 
    hhat = z[0:9]; xhat = z[9:nz].reshape(2,nx)
    Hhat = hhat.reshape(3,3)
    xphat_h = Hhat @ np.block([[xhat], [np.ones(nx)]]); 
    xphat_h = xphat_h / xphat_h[2,:];  xphat = xphat_h[0:2,:]
    xx = xn[0:2] - xhat; xxhat = xpn[0:2] - xphat
    return np.block([xx.flatten(), xxhat.flatten()])


# Input: Points
x = np.block([[-1.5, -1.5, 1], 
              [-2,  2, 1], 
              [2,  2, 1], 
              [2, -2, 1],
              [1,1,1], 
              [1,0,1], 
              [0,1,1]]).T

H = np.array([[1, 0, 0], 
              [0, 1, 0], 
              [0.2, 0.2, 0.8]])
xp = H @ x
print(xp)
xp = xp/xp[2,:]
print(xp)

nx = x.shape[1]
xn = x; xn[0:2] += random.normal(0, 0.1, (2,nx))
xpn = xp; xpn[0:2] += random.normal(0, 0.1, (2,nx))

zzz = np.zeros((nx,3))
A = np.block([[xn.T, zzz, -xn.T*xpn[0].reshape(nx,1)], 
              [zzz, xn.T, -xn.T*xpn[1].reshape(nx,1)] ])
print("--------\n", A)
ua, sa, vat = np.linalg.svd(A)
ha = vat[8]/vat[8,8]*0.8
Ha = np.block([[ha[0:3]], [ha[3:6]], [ha[6:9]]]) 

# Reprojection error
xh = xn; xph = Ha @ xn; xph = xph/ xph[2]
xt = xn - xh; xpt = xpn - xph
xt = xt[0:2].T.flatten(); xpt = xpt[0:2].T.flatten()
r = np.block([xt, xpt])
dg = np.inner(r,r)

print('Ha = \n {} \n dg = {}'.format(Ha, dg))

# Optimization of H
h0 = ha.copy(); x0 = xn.copy(); x0 = x0[0:2] 
z0_reprojection = np.block([ha.copy(),  x0.T.flatten()])
res_1 = least_squares(fun_reprojection, z0_reprojection)

np.set_printoptions(formatter={'float': '{: 0.4f}'.format})
z = res_1.x
H = z[0:9].reshape(3,3); H = H/H[2,2]*Ha[2,2]
print('\n H = \n', H)
xh = z[9:].reshape(2, nx)
print('\n xh =\n', xh)
print('\n optimality = ', res_1.optimality)
print('\n Ha = \n', Ha)

xph = Ha @ np.block([[xh], [np.ones(nx)]]); xph = xph/ xph[2]
xt = xn - np.block([[xh], [np.ones(nx)]]); xpt = xpn - xph
xt = xt[0:2].T.flatten(); xpt = xpt[0:2].T.flatten()
r = np.block([xt, xpt])
dg = np.inner(r,r)
print('\n xn =\n', xn)
print('dg = {}'.format(dg))

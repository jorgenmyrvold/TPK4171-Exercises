import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

def skew(r):
	return np.array([[0,-r[2],r[1]], [r[2],0,-r[0]], [-r[1],r[0],0]])
def expso3(u):
    S = skew(u)
    un = np.linalg.norm(u)
    return np.identity(3) + np.sinc(un/np.pi)*S + 0.5*(np.sinc(un/(2*np.pi)))**2 * S@S 

def rotx(angle):
    return np.array([[1, 0, 0],
                     [0, np.cos(angle), -np.sin(angle)],
                     [0, np.sin(angle), np.cos(angle)]])

def residual_transfer_error(hh):
    Hh = hh.reshape(3,3)
    xph = Hh @ x
    ph = xph[0:2]/xph[2,:]
    residual = p - ph
    return residual.flatten()
                        
K = np.array([[1500, 0, 640], 
              [0, 1500, 512], 
              [0, 0, 1]])
t = np.array([[0,0,2]]).T
R = rotx(np.deg2rad(120))
T = np.block([[R, t], 
              [0,0,0,1]])
Ha = K @ np.block([R[:,0:2], t])

# Points in object frame                        
roh = np.array([[0,0,0,1], [0,1,0,1], [0,2,0,1], [0,3,0,1],
               [0.5,0.5,0,1], [-0.5,0.5,0,1], [-1,1,0,1], 
               [-1,2,0,1], [-1,3,0,1]]).T
nx = roh.shape[1]
x =  np.block([[roh[0:2,:]], [np.ones(nx)]])
# Transformation from object plane to pixel coordinates
xp = Ha @ x
xp = xp/xp[2,:]        # Scaling
xp = np.floor(xp+0.5)  # Truncates to closest integer pixel position
xp += np.array([[-3,-3,0], [-1,1,0], [1,-2,0], [-2,-2,0],
               [3,3,0], [-1,0,0], [2,2,0], 
               [2,0,0], [0,2,0]]).T # Noise
p = xp[0:2]

# DLT solution with vectorized A
zzz = np.zeros((nx,3))
A = np.block([[x.T, zzz, -x.T*p[0].reshape(nx,1)], 
              [zzz, x.T, -x.T*p[1].reshape(nx,1)] ])
u, s, vt = np.linalg.svd(A)
h = vt[8]/vt[8,8]
Hdlt = np.block([[h[0:3]], [h[3:6]], [h[6:9]]]) 

# Optimization of transfer error with DLT solution as initial value
H0 = Hdlt.copy()
h0 = H0.flatten()
r0 = residual_transfer_error(h0); d_DLT = np.dot(r0,r0)
res_1 = least_squares(residual_transfer_error, h0)

# Printing
np.set_printoptions(formatter={'float': '{: 0.0f}'.format})
hopt = res_1.x
r_opt = residual_transfer_error(hopt); d_opt = np.dot(r_opt,r_opt)
Hopt = hopt.reshape(3,3); Hopt = Hopt/Hopt[2,2]*Ha[2,2]
print('\n optimality = ', res_1.optimality)
print('\n Hopt = \n {} \n\n Hdlt = \n {}\n\n Ha = \n {} \
      \n\n d_DLT = {} \n d_opt = {}'
      .format(Hopt/Hopt[2,2], Hdlt, Ha/Ha[2,2], d_DLT, d_opt))

print('\n\n------- THIS:')
print('xp\n{}'.format(xp))
print('Hdlt @ x \n{}'.format(Hdlt @ x))





 



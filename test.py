import numpy as np
np.set_printoptions(formatter={'float': '{: 0.0f}'.format})


x = np.array([[0, 0, 1],
              [0.1, 0, 1],
              [0.1, 0.1, 1],
              [0, 0.1, 1],
              [0.2, -0.1, 1],
              [0.15, 0.1, 1],
              [-0.1, 0.3, 1],
              [-0.2, 0.1, 1],
              [-0.2, 0, 1]]).T

xp = np.array([[960, 540, 1],
                  [1320, 540, 1],
                  [1320, 900, 1],
                  [960, 900, 1],
                  [1680, 180, 1],
                  [1500, 900, 1],
                  [600, 1620, 1],
                  [240, 900, 1],
                  [240, 540, 1]]).T

nx = x.shape[1]
xn = x
xpn = xp

zzz = np.zeros((nx,3))
A = np.block([[xn.T, zzz, -xn.T*xpn[0].reshape(nx,1)], 
              [zzz, xn.T, -xn.T*xpn[1].reshape(nx,1)]])
_, _, v = np.linalg.svd(A)
h = v[-1]/v[-1,-1]
H = np.block([[h[0:3]], [h[3:6]], [h[6:9]]]) 

print(H)
new_xp = H @ x
print(xp)
print(new_xp)

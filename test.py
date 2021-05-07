import numpy as np
np.set_printoptions(formatter={'float': '{: 0.4f}'.format})


import numpy as np
def skewm(r):
    return np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])

# r[0] = np.array([0,0,0])
# r[1] = np.array([1,0,0])
# r[2] = np.array([1,1,0])
# r[3] = np.array([0,1,0])

r = np.array([[0,0,0],
                   [1,0,0],
                   [1,1,0],
                   [0,1,0]])

# s[0] = np.array([0,0,1])
# s[1] = np.array([0.5, 0, 1])
# s[2] = np.array([0.34, -0.116, 1])
# s[3] = np.array([0, -0.116, 1])

s = np.array([[0,0,1],
                  [0.5, 0, 1],
                  [0.34, -0.116, 1],
                  [0, -0.116, 1]])

A1 = np.hstack([r[0][0]*skewm(s[0]), r[0][1]*skewm(s[0]), skewm(s[0])]);
A2 = np.hstack([r[1][0]*skewm(s[1]), r[1][1]*skewm(s[1]), skewm(s[1])]);
A3 = np.hstack([r[2][0]*skewm(s[2]), r[2][1]*skewm(s[2]), skewm(s[2])]);
A4 = np.hstack([r[3][0]*skewm(s[3]), r[3][1]*skewm(s[3]), skewm(s[3])]);
A = np.vstack((A1, A2, A3, A4))
u,s,v = np.linalg.svd(A)
h = v[8]
scale = np.sign(h[8])*np.linalg.norm(h[0:3])
r1 = h[0:3]/scale
r2 = h[3:6]/scale
r3 = np.cross(r1,r2)
t = h[6:9]/scale
H = np.identity(4)
H[0:3,0] = r1; H[0:3,1] = r2; H[0:3,2] = r3; H[0:3,3] = t # Result: Homography

print ('Estimated pose:')
print(H)
import numpy as np
def skewm(r):
	return np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])
def expso3(u):
    S = skewm(u)
    return np.eye(3) + np.sinc(np.linalg.norm(u)/np.pi)*S \
        + 0.5*(np.sinc(np.linalg.norm(u)/(2*np.pi)))**2 * S @ S


# Input: Homogeneous points in object frame
x1 = np.array([0, 0, 0, 1])
x2 = np.array([0.1, 0, 0, 1])
x3 = np.array([0.1, 0.1, 0, 1])
x4 = np.array([0, 0.1, 0, 1])

Rco = expso3(np.pi/4*np.array([0,1,0])) @ expso3(120/180*np.pi*np.array([1,0,0]))
ococ = np.array([0.1, 0, 0.5]) 
Tco = np.eye(4)
Tco[0:3,0:3] = Rco
Tco[0:3,3] = ococ[0:3]

# Homogeneous points in camera frame
xp1 = Tco @ x1
xp2 = Tco @ x2
xp3 = Tco @ x3
xp4 = Tco @ x4

# Sensor readings: Homogeneous normalized image coordinates
xp1 = xp1[0:3]/xp1[2]
xp2 = xp2[0:3]/xp2[2]
xp3 = xp3[0:3]/xp3[2]
xp4 = xp4[0:3]/xp4[2]

A1 = np.hstack([x1[0]*skewm(xp1), x1[1]*skewm(xp1), skewm(xp1)]);
A2 = np.hstack([x2[0]*skewm(xp2), x2[1]*skewm(xp2), skewm(xp2)]);
A3 = np.hstack([x3[0]*skewm(xp3), x3[1]*skewm(xp3), skewm(xp3)]);
A4 = np.hstack([x4[0]*skewm(xp4), x4[1]*skewm(xp4), skewm(xp4)]);
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

np.set_printoptions(formatter={'float': '{: 0.4f}'.format})
print ('Estimated pose:')
print(H)
print ('Actual pose:')
print(Tco)

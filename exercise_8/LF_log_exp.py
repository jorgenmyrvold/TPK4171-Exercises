import numpy as np
np.set_printoptions(formatter={'float': '{: 0.4f}'.format})


def skewm(r):
    return np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])

def expSO3(u):
    return np.identity(3) + np.sinc(np.linalg.norm(u)/np.pi)*skewm(u) + 0.5*(np.sinc(np.linalg.norm(u)/(2*np.pi)))**2 * skewm(u) @ skewm(u)

def vex(u):
    return np.array([u[2,1], u[0,2], u[1,0]])

def logSO3(R, margin=0.000001):
    # The vector form of the logarithm in SO(3) (Iserles, 2006)
    eh = 0.5*(R-R.T) # eh = sin(theta)k^skew
    en = np.linalg.norm(vex(eh)) # en = |sin(theta)|
    if en < margin:
        g = 1 + (en**2)/6
    else:
        g = (np.arcsin(en)/en)
    return vex(g*eh)

# Unit vectors
ex = np.array([1, 0, 0])
ey = np.array([0, 1, 0])
ez = np.array([0, 0, 1]);

# Problem 1a)
R1 = expSO3(ex*np.pi/12)
R2 = expSO3(-ex*np.pi/12)
Ra = R1.T @ R2
loga = logSO3(Ra)
print('\n loga =\n {}'. format(loga))

# Problem 1b)
R0 = expSO3(ey*np.pi/6)
R3 = R0 @ R1; R4 = R0 @ R2
Rb = R3.T @ R4
logb = logSO3(Rb)
print('\n logb =\n {}'. format(logb))

# Problem 1c)
R0 = expSO3(ey*np.pi/6)
R5 = R1 @ R0; R6 = R2 @ R0
Rc = R5.T @ R6
logc = logSO3(Rc)
print('\n logc =\n {} \n R0.T @ loga = \n {}'. format(logc, R0.T @ loga))
R0 @ loga


# Problem 2
c = np.pi/180
Rd = np.stack((
expSO3(ex*c*5), expSO3(-ex*c*5), expSO3(ex*c*7.5),
expSO3(-ex*c*2.5), expSO3(ey*c*5), expSO3(-ey*c*5),
expSO3(ey*c*2.5), expSO3(-ey*c*7.5), expSO3(ez*c*3),
expSO3(-ez*c*4)
))
N = Rd.shape[0]

# Minimization of angular distance with gradient search
R = Rd[0,:,:]; r = np.ones(3);
while np.linalg.norm(r) > 0.001:
    r = np.zeros(3)
    for i in range(0,N):
        r = r + logSO3(R.T@Rd[i,:,:])
    R = R@expSO3(r/N)
    R_grad_search = R
    
# Minimization of chordal distance with Procrustes minimization
H = np.zeros((3,3));
for i in range(0,N):
    H = H + Rd[i,:,:].T
U,S,Vt = np.linalg.svd(H); V = Vt.T
Rsvd = V @ np.diag([1,1,np.linalg.det(V @U.T)])@U.T
print('\n R_grad_search =\n {} \n\n Rsvd =\n {}\n'.format(R_grad_search, Rsvd))
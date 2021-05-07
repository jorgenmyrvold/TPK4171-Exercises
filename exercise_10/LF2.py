import numpy as np

def weks(M): # From skew symetric matrix to vector
    if M[0,1] != -M[1,0] or M[0,2] != -M[2,0] or M[1,2] != -M[2,1]:
        print('ERROR: Matrix is not skew symetric')
    return np.array([M[2,1], M[0,2], M[1,0]])

def skewm(r):
    return np.array([[0,-r[2],r[1]], [r[2],0,-r[0]], [-r[1],r[0],0]])

def expso3(u):
    S = skewm(u); un = np.linalg.norm(u)
    return np.identity(3) + np.sinc(un/np.pi)*S + 0.5*(np.sinc(un/(2*np.pi)))**2 * S@S

def logSO3(R):
    # The vector form of the logarithm in SO(3) (Iserles, 2006)
    eh = 0.5*(R-R.T) # eh = sin(theta)k^skew
    en = np.linalg.norm(weks(eh)) # en = |sin(theta)|
    if en < 0.000001:
        g = 1 + (en**2)/6
    else:
        g = (np.arcsin(en)/en)
    return weks(g*eh)

def unit_quat_to_rotation(q):
    s = q[0]; v = q[1:4]
    V = skewm(v)
    return np.eye(3) + 2*s*V + 2*V@V

def expquat(u):
    phi = np.linalg.norm(u)
    return np.block([np.cos(phi), np.sinc(phi/np.pi)*u])

def shepperd(R):
    z00 = np.trace(R)
    z11 = R[0,0] + R[0,0] - z00
    z22 = R[1,1] + R[1,1] - z00
    z33 = R[2,2] + R[2,2] - z00
    #Find a large zii to avoid division by zero
    if z00 >= 0.5:
        w = np.sqrt(1.0 + z00)
        wInv = 1.0/w
        x = (R[2,1] - R[1,2])*wInv;
        y = (R[0,2] - R[2,0])*wInv;
        z = (R[1,0] - R[0,1])*wInv;
    elif z11 >= 0.5:
        x = np.sqrt(1.0 + z11);
        xInv = 1.0/x;
        w = (R[2,1] - R[1,2])*xInv;
        y = (R[1,0] + R[0,1])*xInv;
        z = (R[2,0] + R[0,2])*xInv;
    elif z22 >= 0.5:
        y = np.sqrt(1.0 + z22);
        yInv = 1.0/y;
        w = (R[0,2] - R[2,0])*yInv;
        x = (R[1,0] + R[0,1])*yInv;
        z = (R[2,1] + R[1,2])*yInv;
    else:
        z = np.sqrt(1.0 + z33);
        zInv = 1.0/z;
        w = (R[1,0] - R[0,1])*zInv;
        x = (R[2,0] + R[0,2])*zInv;
        y = (R[2,1] + R[1,2])*zInv;
    s = 0.5*w;
    v = 0.5*np.array([x, y, z])
    if s < 0:
        s = - s; v = - v;
    return np.block([s, v])

ex = np.array([1,0,0]); ey = np.array([0,1,0]); ez = np.array([0,0,1])
R = np.eye(3)
u = logSO3(R)
qL = expquat(u/2)
qs = shepperd(R)
print('\n qL = {}\n qs = {}'.format(qL, qs))

R = expso3(np.pi/6*ex)
u = logSO3(R)
qL = expquat(u/2)
qs = shepperd(R)
print('\n qL = {}\n qs = {}'.format(qL, qs))

R = expso3(np.pi/2*ez)
u = logSO3(R)
qL = expquat(u/2)
qs = shepperd(R)
print('\n qL = {}\n qs = {}'.format(qL, qs))

R = expso3(np.pi*ey)
u = logSO3(R)
qL = expquat(u/2)
qs = shepperd(R)
print('\n qL = {}\n qs = {}'.format(qL, qs))

RL = unit_quat_to_rotation(qL)
Rs = unit_quat_to_rotation(qs)
print('\n R =\n {}\n \n RL =\n {}\n Rs =\n {}'.format(R, RL, Rs))

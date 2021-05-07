import numpy as np
np.set_printoptions(formatter={'float': '{: 0.4f}'.format})


def skewm(r):
    return np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])

def vex(u):
    return np.array([u[2,1], u[0,2], u[1,0]])

def expSO3(u):
    return np.identity(3) + np.sinc(np.linalg.norm(u)/np.pi)*skewm(u) + 0.5*(np.sinc(np.linalg.norm(u)/(2*np.pi)))**2*skewm(u)@skewm(u)

def logSO3(R):
# The vector form of the logarithm in SO(3) (Iserles, 2006)
    eh = 0.5*(R-R.T) # eh = sin(theta)k^skew
    en = np.linalg.norm(vex(eh)) # en = |sin(theta)|
    if en < 0.000001:
        g = 1+ (en**2)/6
    else:
        g = (np.arcsin(en)/en)
    return vex(g*eh)

# Unit vectors
ex = np.array([1, 0, 0]) 
ey = np.array([0, 1, 0]) 
ez = np.array([0, 0, 1])

# Problem 1
R1 = expSO3(ex*0)
R2 = expSO3(-ex*1.e-10)
R3 = expSO3(ey*1.e-4)
R4 = expSO3(ey*np.pi)
print('\n R1 =\n {}\n R2 =\n {}\n R3 =\n {}\n R4 =\n {}'.format(R1, R2, R3, R4))

L1 = logSO3(R1)
L2 = logSO3(R2)
L3 = logSO3(R3)
L4 = logSO3(expSO3(ey*0.99*np.pi))
print('\n L1 = {}\n L2 = {}\n L3 = {}\n L4 = {}'.format(L1, L2, L3, L4))


# Problem 2
def J_L(u):
    theta = np.linalg.norm(u); uh = skewm(u)
    if theta > 0.000001:
        a = (1-np.cos(theta))/(theta**2)
        b = (theta - np.sin(theta))/(theta**3)
    else:
        a = 0.5 - theta**2/24
        b = 1/3 - theta**2/120
    return np.eye(3) + a*uh + b*uh@uh

def J_L_inv(u):
    theta = np.linalg.norm(u); thetahalf = theta/2
    uh = skewm(u)
    if theta > 0.000001:
        a = (1 - thetahalf/np.tan(thetahalf))/(theta**2)
    else:
        a = 1/12 - 1/180*theta**2
    return np.eye(3) - 0.5*uh + a*uh@uh

def J_R(u):
    return J_L(-u)

def J_R_inv(u):
    return J_L_inv(-u)

u1 = ex*np.pi/3
u2 = -ey*np.pi/6
u3 = ez*0.00000001
u4 = ex + ey - ez

JL1 = J_L(u1)
JL2 = J_L(u2)
JL3 = J_L(u3)
JL4 = J_L(u4)

JLinv1 = J_L_inv(u1)
JLinv2 = J_L_inv(u2)
JLinv3 = J_L_inv(u3)
JLinv4 = J_L_inv(u4)

print('\n JL1 = \n {}\n\n JL2 = \n {}\n\n JL3 = \n {}\n\n JL4 = \n {}'.format(JL1, JL2, JL3, JL4))
print('\n JL1@JLinv1 = \n {}\n\n JL2@JLinv2 = \n {}\n\n JL3@JLinv3 = \n {}\n\n JL4@JLinv4 = \n {}'.format(JL1@JLinv1, JL2@JLinv2, JL3@JLinv3, JL4@JLinv4))
JR1 = J_R(u1)
JRinv1 = J_R_inv(u1)
print('\n JR1@JRinv1 = \n{}\n'.format(JR1@JRinv1))
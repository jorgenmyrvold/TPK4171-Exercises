import numpy as np
np.set_printoptions(formatter={'float': '{: 0.4f}'.format})

def skewm(r):
    return np.array([[0,-r[2],r[1]], [r[2],0,-r[0]], [-r[1],r[0],0]])

def expso3(u):
    S = skewm(u); un = np.linalg.norm(u)
    return np.eye(3) + np.sinc(un/np.pi)*S + 0.5*(np.sinc(un/(2*np.pi)))**2 * S@S

ez = np.array([0,0,1])
# Problem 1a
B = np.block([[1., 0., 0.], [1., 1., 0.]]).T
A = np.block([[0.5, 0.866, 0.], [-0.25, 1.299, 0.5]]).T
H = B @ A.T
U, S, Vt = np.linalg.svd(H)
R = Vt.T @ np.diag([1,1,np.linalg.det( Vt.T @ U.T)]) @ U.T
print('\n Ra =\n {}\n det(Ra) = {:.4f}'.format(R, np.linalg.det(R)))


# Problem 1b
B = np.block([[1., 0., 0.001], [0., 1., 0.]]).T
Ra = expso3(np.pi/2*ez)
A =Ra@B + np.block([[0., 0., -0.01], [0., 0., 0.]]).T
H = B @ A.T
U, S, Vt = np.linalg.svd(H)
R = Vt.T @ U.T
print('\n Rb =\n {}\n det(Rb) = {:.4f}'.format(R, np.linalg.det(R)))
R = Vt.T @ np.diag([1,1,np.linalg.det( Vt.T @ U.T)]) @ U.T
print('\n Rr =\n {}\n det(Rr) = {:.4f}'.format(R, np.linalg.det(R)))
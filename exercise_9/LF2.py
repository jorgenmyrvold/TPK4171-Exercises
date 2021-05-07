import numpy as np

def skewm(r):
    return np.array([[0,-r[2],r[1]], [r[2],0,-r[0]], [-r[1],r[0],0]])

def expso3(u):
    S = skewm(u); un = np.linalg.norm(u)
    return np.eye(3) + np.sinc(un/np.pi)*S + 0.5*(np.sinc(un/(2*np.pi)))**2 * S@S

U = np.block([[0.,0.,0.5], 
              [0.2,0.,0.5], 
              [-0.1,0.,0.5], 
              [0.,0.05,0.5], 
              [0.,-0.15,0.5]])

Rd = np.stack((expso3(U[0]), 
               expso3(U[1]),
               expso3(U[2]),
               expso3(U[3]),
               expso3(U[4])
))

H = np.sum(Rd,0).T
U, S, Vt = np.linalg.svd(H)
R = Vt.T @ np.diag([1,1,np.linalg.det(Vt.T @ U.T)]) @ U.T
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
print('\n Rr =\n {}\n det(Rr) = {:.4f}'.format(R, np.linalg.det(R)))
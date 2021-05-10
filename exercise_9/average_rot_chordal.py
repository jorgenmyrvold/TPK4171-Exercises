import numpy as np
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

def skew(v):
    v = v.flat
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

def expso3(u):
    S = skew(u)
    un = np.linalg.norm(u)
    return np.eye(3) + np.sinc(un/np.pi)*S + 0.5*(np.sinc(un/(2*np.pi)))**2 * S@S

def chordal_distance(angle):
    return np.sqrt(8 * np.sin(angle/2)**2)
    
def decompose_exp(u): # takes a vector u=θk and returns θ and k
    u = u.flat
    return np.linalg.norm(u), (1/np.linalg.norm(u))*u

def average_chordal_dist(u):
    '''
    Given a list of u, calculate the average rotation using chordal distance
    param:
        u: list of exponential rotations u on the form
            u = [[x1, x2, ..., xn],
                 [y1, y2, ..., yn]
                 [z1, z2, ..., zn]]
    return:
        R: The average rotation matrix
    '''
    R = np.zeros((u.shape[1], 3, 3))
    for i in range(R.shape[0]):
        R[i] = expso3(u[:,i])
        
    H = np.sum(R, axis=0).T
    u, _, v = np.linalg.svd(H)
    R = v.T @ np.diag([1,1,np.linalg.det(v.T @ u.T)]) @ u.T # Second part is to not reflect
    return R


if __name__ == '__main__':
    
    u = np.array([[0, 0, 0.5],
                [0.2, 0, 0.5],
                [-0.1, 0, 0.5],
                [0, 0.05, 0.5],
                [0, -0.15, 0.5]]).T

    t = np.zeros(u.shape[1])
    k = np.zeros(u.shape)
    for i in range(u.shape[1]):
        t[i], k[:,i] = decompose_exp(u[:,i])
    
    chordal_dist = np.zeros(t.shape)
    for i in range(t.shape[0]):
        chordal_dist = chordal_distance(t[i])
    
    avg_chordal_dist = np.average(chordal_dist)
    
    print(average_chordal_dist(u))


